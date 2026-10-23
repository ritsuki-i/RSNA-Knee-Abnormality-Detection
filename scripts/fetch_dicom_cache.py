# -*- coding: utf-8 -*-
"""配布 zip に HTTP Range で直接アクセスし、DICOM をその場で PNG キャッシュへ変換する。

■ なぜこの方式か

Kaggle 配布データをローカルで扱うには 3 つの壁がある。

1. 容量      : DICOM を展開すると約450GB。C: の空きは254GB なので一括展開は不可能。
2. パス長    : LongPathsEnabled=0 のため、配布時の構成
               train_series/<UID63>/<UID63>/<UID63>.dcm (約284文字) は作成できない。
3. レート制限: ファイル単位 API (competition_download_file) は約550ファイルで
               429/404 になり、復帰まで約27分かかる。実測の持続レートは約0.34 files/s
               で、819,640 ファイルなら約32日。使い物にならない。

一方、一括ダウンロード URL は Google Cloud Storage の署名 URL にリダイレクトされ、
そこは `Accept-Ranges: bytes` に対応している。つまり **265GB の zip 全体を落とさずに、
必要なメンバーだけを Range リクエストで抜き出せる**。ここは Kaggle の API を通らない
ので上記のレート制限もかからない。実測 18.1 MB/s。

    zip の中央ディレクトリを一度だけ取得 (233MB)
      -> 各メンバーの位置とサイズが分かる
      -> series 単位で連続領域を Range 取得 -> inflate -> PNG 化 -> 生データは捨てる

ピーク時のディスク使用量は PNG キャッシュだけ (384px で約41GB)。zip は一切保存しない。

■ 使い方

    python scripts/fetch_dicom_cache.py --index-only      # 索引だけ作る (233MB, 数分)
    python scripts/fetch_dicom_cache.py --studies 20      # 動作確認
    python scripts/fetch_dicom_cache.py --all             # 全部 (約4-5時間, 中断可)
    python scripts/fetch_dicom_cache.py --split test --all

    --every 2   スライスを 1 枚おきに間引く (容量と時間が半分になる)
    --crop-mm 140   元画像の中央 140mm 四方を切り出してから縮小する（img_<split>_c140_<size>、状態・CSV も別名）

■ 出力

    data/processed/img_<split>_<size>/<study_idx>/<series_idx>/<slice_idx>.png
    data/interim/zip_index.csv.gz            zip 内の全メンバーの位置 (再利用可)
    data/interim/dicom_meta_<split>.csv      スライスごとの DICOM ヘッダ
    data/interim/dicom_manifest_<split>.csv  連番 <-> UID の対応
    data/interim/fetch_state_<split>.json    進捗 (再開用)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import queue
import struct
import sys
import threading
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geo3d.geometry import phys_box  # noqa: E402

COMP = "rsna-knee-abnormality-detection"
URL_EP = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{COMP}"

HDR_TAGS = [
    "SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
    "RepetitionTime", "EchoTime", "Laterality", "ImageLaterality",
    "PixelSpacing", "SliceThickness", "Rows", "Columns",
    "RescaleSlope", "RescaleIntercept",
    "ImagePositionPatient", "ImageOrientationPatient",
    "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
    "BitsStored", "PhotometricInterpretation", "InstanceNumber",
]
META_COLS = ["study_idx", "series_idx", "slice_idx", "StudyInstanceUID",
             "SeriesInstanceUID", "src_rows", "src_cols", "win_lo", "win_hi"] + HDR_TAGS


def project_root() -> Path:
    here = Path(__file__).resolve()
    for cand in [here.parent, *here.parents]:
        if (cand / "src" / "utils" / "paths.py").is_file() and (cand / "config" / "config.yaml").is_file():
            return cand
    raise RuntimeError("プロジェクトルートが見つからない")


ROOT = project_root()
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
INDEX_PATH = INTERIM / "zip_index.csv.gz"


# ============================================================ 署名 URL / Range
class ZipRemote:
    """署名 URL 越しに zip を読む。URL は期限切れするので 403 で取り直す。"""

    def __init__(self):
        self.sess = requests.Session()
        self.url = None
        self.size = None
        self._refresh()

    def _refresh(self):
        # 認証情報は KAGGLE_API_TOKEN か KAGGLE_CONFIG_DIR 配下のみ参照する（~/.kaggle は見ない）
        cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "/mnt/home/ishikawa/kaggle/.kaggle"))
        tokp = cfg_dir / "access_token"
        s = requests.Session()
        if os.environ.get("KAGGLE_API_TOKEN"):
            s.headers["Authorization"] = "Bearer " + os.environ["KAGGLE_API_TOKEN"].strip()
        elif tokp.exists():
            s.headers["Authorization"] = "Bearer " + tokp.read_text().strip()
        else:
            raise RuntimeError(f"Kaggle の認証情報が見つからない ({cfg_dir})")
        r = s.get(URL_EP, stream=True, allow_redirects=True, timeout=120)
        r.close()
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers["Content-Length"])

    def read(self, start, length):
        """[start, start+length) を返す。失敗時は URL 再取得を挟んで再試行。"""
        end = start + length - 1
        last = None
        for attempt in range(5):
            try:
                r = self.sess.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                  timeout=300)
                if r.status_code in (403, 401):
                    self._refresh()
                    continue
                r.raise_for_status()
                if len(r.content) != length:
                    raise IOError(f"短い応答 {len(r.content)} != {length}")
                return r.content
            except Exception as e:
                last = e
                time.sleep(2 * (attempt + 1))
                if attempt == 2:
                    self._refresh()
        raise IOError(f"Range 取得に失敗: {start}+{length}: {last}")


# ====================================================== 中央ディレクトリの解析
def _cd_location(remote: ZipRemote):
    tail = remote.read(remote.size - 70000, 70000)
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        (_sig, _sz, _vmb, _vn, _dsk, _cdd, _nd, ntot,
         cds, cdo) = struct.unpack("<IQHHIIQQQQ", tail[j:j + 56])
        return cdo, cds, ntot
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("zip の EOCD が見つからない")
    (_s, _d1, _d2, _n1, ntot, cds, cdo, _c) = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    return cdo, cds, ntot


def _parse_cd(buf):
    """中央ディレクトリを走査して (name, method, csize, local_header_offset) を返す。"""
    p, n = 0, len(buf)
    while p + 46 <= n:
        if buf[p:p + 4] != b"PK\x01\x02":
            break
        (_sig, _vmb, _vn, _flg, method, _mt, _md, _crc, csize, usize,
         nlen, elen, clen, _dn, _ia, _ea, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", buf[p:p + 46])
        name = buf[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = buf[p + 46 + nlen:p + 46 + nlen + elen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or lho == 0xFFFFFFFF:
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    v, k = extra[q + 4:q + 4 + hsz], 0
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", v[k:k + 8])[0]; k += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", v[k:k + 8])[0]; k += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", v[k:k + 8])[0]; k += 8
                    break
                q += 4 + hsz
        yield name, method, csize, lho
        p += 46 + nlen + elen + clen


def build_index(remote: ZipRemote):
    """中央ディレクトリ (233MB) を取得して csv.gz に保存する。一度だけ。"""
    cd_off, cd_size, n_ent = _cd_location(remote)
    print(f"zip {remote.size/1e9:.1f} GB / エントリ {n_ent:,} / "
          f"中央ディレクトリ {cd_size/1e6:.1f} MB")
    parts, got = [], 0
    step = 32 << 20
    t0 = time.time()
    while got < cd_size:
        m = min(step, cd_size - got)
        parts.append(remote.read(cd_off + got, m))
        got += m
        print(f"\r  索引取得 {got/1e6:6.1f} / {cd_size/1e6:.1f} MB "
              f"({got/1e6/max(time.time()-t0,.01):.1f} MB/s)", end="", flush=True)
    print()
    buf = b"".join(parts)
    n = 0
    INTERIM.mkdir(parents=True, exist_ok=True)
    tmp = INDEX_PATH.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "method", "csize", "offset"])
        for name, method, csize, lho in _parse_cd(buf):
            w.writerow([name, method, csize, lho])
            n += 1
    tmp.replace(INDEX_PATH)
    print(f"索引を書き出した: {INDEX_PATH.name} ({n:,} エントリ, "
          f"{INDEX_PATH.stat().st_size/1e6:.1f} MB)")


def load_index(split):
    """split の .dcm メンバーを study -> series -> [(name, method, csize, off)] で返す。"""
    prefix = f"{split}_series/"
    studies = OrderedDict()
    with gzip.open(INDEX_PATH, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["name"]
            if not name.startswith(prefix) or not name.endswith(".dcm"):
                continue
            _, st, se, _fn = name.split("/", 3)
            studies.setdefault(st, OrderedDict()).setdefault(se, []).append(
                (name, int(row["method"]), int(row["csize"]), int(row["offset"])))
    return studies


# ================================================================== 変換
def inflate_member(remote: ZipRemote, method, csize, lho, blob=None, base=None):
    """ローカルヘッダを読み飛ばしてメンバー本体を展開する。"""
    if blob is None:
        hdr = remote.read(lho, 30)
    else:
        hdr = blob[lho - base:lho - base + 30]
    (_sig, _vn, _flg, _mth, _mt, _md, _crc, _c, _u,
     nlen, elen) = struct.unpack("<IHHHHHIIIHH", hdr[:30])
    doff = lho + 30 + nlen + elen
    if blob is None:
        data = remote.read(doff, csize)
    else:
        data = blob[doff - base:doff - base + csize]
    if method == 8:
        return zlib.decompressobj(-zlib.MAX_WBITS).decompress(data)
    if method == 0:
        return data
    raise RuntimeError(f"未対応の圧縮方式 {method}")


def to_uint8(arr):
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8), float(lo), float(hi)


def hdr_value(ds, tag):
    v = getattr(ds, tag, None)
    if v is None:
        return ""
    try:
        if isinstance(v, (list, tuple)) or v.__class__.__name__ == "MultiValue":
            return "|".join(str(x) for x in v)
    except Exception:
        pass
    return str(v)


MAX_SPAN = 192 << 20   # 1 回の Range で読む最大バイト数


def _series_blob(remote, members):
    """series 分の連続領域をまとめて 1 回で読む。大きすぎる場合は None。"""
    lo = members[0][3]
    hi = members[-1][3] + 30 + 512 + members[-1][2]
    span = hi - lo
    if span > MAX_SPAN:
        return None, None
    try:
        return remote.read(lo, min(span, remote.size - lo)), lo
    except Exception:
        return None, None


def convert_study(remote, study_uid, series_map, study_idx, size, img_root,
                  every, pool, png_level, crop_mm=0.0):
    """1 study を変換する。

    ダウンロードは 1 本の先読みスレッドが担当し、その裏で変換をスレッドプールに
    流す。こうしないと 20 コアが遊んだまま回線待ちになる (実測 19.5h -> 4-5h)。
    """
    rows = []
    n_png = png_bytes = 0
    dl_bytes = 0
    t0 = time.time()

    plan = []
    for si, suid in enumerate(sorted(series_map)):
        members = sorted(series_map[suid], key=lambda m: m[3])
        if every > 1:
            members = members[::every]
        if members:
            plan.append((si, suid, members))

    q = queue.Queue(maxsize=2)

    def producer():
        for si, suid, members in plan:
            blob, base = _series_blob(remote, members)
            q.put((si, suid, members, blob, base))
        q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()

    def one_slice(job):
        li, name, method, csize, off, blob, base, out = job
        try:
            raw = inflate_member(remote, method, csize, off, blob, base)
            ds = pydicom.dcmread(io.BytesIO(raw))
            px = ds.pixel_array
        except Exception as e:
            print(f"\n    skip ({type(e).__name__}) {name[-40:]}", file=sys.stderr)
            return None
        u8, wlo, whi = to_uint8(px)
        if crop_mm > 0 and u8.ndim == 2:
            # 正規化は画像全体で（従来のキャッシュと同じ）、そのあと中央 crop_mm 四方を切り出す
            ps = getattr(ds, "PixelSpacing", None)
            y0, x0, h, w = phys_box(ps, u8.shape[0], u8.shape[1], crop_mm)
            u8 = u8[y0:y0 + h, x0:x0 + w]
        dest = out / f"{li:03d}.png"
        Image.fromarray(u8).resize((size, size), Image.BILINEAR).save(
            dest, format="PNG", compress_level=png_level)
        return dest, ds, wlo, whi, len(raw)

    while True:
        item = q.get()
        if item is None:
            break
        si, suid, members, blob, base = item
        if blob is not None:
            dl_bytes += len(blob)
        out = img_root / f"{study_idx:05d}" / f"{si:02d}"
        out.mkdir(parents=True, exist_ok=True)

        jobs = [(li, m[0], m[1], m[2], m[3], blob, base, out)
                for li, m in enumerate(members)]
        for li, res in enumerate(pool.map(one_slice, jobs)):
            if res is None:
                continue
            dest, ds, wlo, whi, raw_len = res
            if blob is None:
                dl_bytes += raw_len
            n_png += 1
            png_bytes += dest.stat().st_size
            row = {"study_idx": study_idx, "series_idx": si, "slice_idx": li,
                   "StudyInstanceUID": study_uid, "SeriesInstanceUID": suid,
                   "src_rows": int(getattr(ds, "Rows", 0)),
                   "src_cols": int(getattr(ds, "Columns", 0)),
                   "win_lo": f"{wlo:.1f}", "win_hi": f"{whi:.1f}"}
            for t in HDR_TAGS:
                row[t] = hdr_value(ds, t)
            rows.append(row)

    th.join(timeout=5)
    return rows, dict(dt=time.time() - t0, dl_mb=dl_bytes / 1e6, n_png=n_png,
                      png_mb=png_bytes / 1e6, n_series=len(series_map))


# ================================================================== 永続化
def append_csv(path: Path, rows, cols):
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def load_state(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"done": {}, "next_idx": 0}


def save_state(p: Path, st):
    t = p.with_suffix(".json.tmp")
    t.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    t.replace(p)


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--studies", type=int, default=20)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--every", type=int, default=1, help="スライスを N 枚おきに間引く")
    ap.add_argument("--jobs", type=int, default=8, help="変換の並列数 (既定 8)")
    ap.add_argument("--png-level", type=int, default=1,
                    help="PNG 圧縮レベル 0-9 (既定 1: 最速。9 でも容量は15%%しか変わらない)")
    ap.add_argument("--index-only", action="store_true", help="zip 索引だけ作って終了")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="並列取得用: このプロセスの担当番号 (0 始まり)")
    ap.add_argument("--nshards", type=int, default=1, help="並列取得用: 全体の分割数")
    ap.add_argument("--crop-mm", type=float, default=0.0, help="元画像の中央 N mm 四方を切り出してから縮小（0 = 全体）")
    args = ap.parse_args()

    INTERIM.mkdir(parents=True, exist_ok=True)
    remote = ZipRemote()

    if args.rebuild_index or not INDEX_PATH.exists():
        build_index(remote)
    if args.index_only:
        return

    tag = f"_c{args.crop_mm:g}" if args.crop_mm > 0 else ""
    img_root = PROCESSED / f"img_{args.split}{tag}_{args.size}"
    img_root.mkdir(parents=True, exist_ok=True)
    sfx = tag + (f"_shard{args.shard}of{args.nshards}" if args.nshards > 1 else "")
    state_path = INTERIM / f"fetch_state_{args.split}{sfx}.json"
    meta_path = INTERIM / f"dicom_meta_{args.split}{sfx}.csv"
    manifest_path = INTERIM / f"dicom_manifest_{args.split}{sfx}.csv"

    print("索引を読み込み中...")
    studies = load_index(args.split)
    print(f"{args.split}: {len(studies):,} study")

    state = load_state(state_path)
    done = state["done"]
    budget = float("inf") if args.all else args.studies

    print(f"cache    : {img_root}")
    print(f"size     : {args.size}px   every={args.every}   jobs={args.jobs}   png_level={args.png_level}")
    print(f"処理済み : {len(done):,} study / 今回上限 {'全部' if args.all else args.studies}")
    print("-" * 78)

    n = 0
    tot_dl = tot_png = 0.0
    t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.jobs)
    try:
        for pos, (study_uid, series_map) in enumerate(studies.items()):
            # study_idx は索引内の位置で決める（逐次実行時の next_idx と同じ値）
            if pos % args.nshards != args.shard or study_uid in done:
                continue
            if n >= budget:
                break
            idx = pos
            rows, st = convert_study(remote, study_uid, series_map, idx,
                                     args.size, img_root, args.every,
                                     pool, args.png_level, args.crop_mm)
            append_csv(meta_path, rows, META_COLS)
            append_csv(manifest_path, [{"study_idx": idx, "StudyInstanceUID": study_uid,
                                        "n_series": st["n_series"], "n_slices": st["n_png"]}],
                       ["study_idx", "StudyInstanceUID", "n_series", "n_slices"])
            done[study_uid] = idx
            state["next_idx"] = max(state["next_idx"], idx + 1)
            save_state(state_path, state)

            n += 1
            tot_dl += st["dl_mb"]
            tot_png += st["png_mb"]
            el = time.time() - t_start
            eta = (len(studies) // args.nshards - len(done)) * el / n / 3600
            print(f"[{n:5d}/{len(studies)}] study {idx:05d} "
                  f"series={st['n_series']:2d} slices={st['n_png']:4d} "
                  f"dl={st['dl_mb']:6.1f}MB png={st['png_mb']:5.1f}MB "
                  f"{st['dt']:5.1f}s  {st['dl_mb']/max(st['dt'],.01):5.1f}MB/s  "
                  f"ETA {eta:5.1f}h", flush=True)
    except KeyboardInterrupt:
        print("\n中断しました (状態は保存済み。同じコマンドで再開できます)")
    finally:
        save_state(state_path, state)
        pool.shutdown(wait=True)

    el = time.time() - t_start
    print("-" * 78)
    print(f"今回 {n} study / {el/60:.1f} 分")
    if n:
        print(f"転送 {tot_dl/1e3:.2f} GB -> PNG {tot_png/1e3:.2f} GB "
              f"({tot_dl/max(tot_png,1e-9):.1f}x)")
        rem = len(studies) - len(done)
        print(f"1 study: dl {tot_dl/n:.1f}MB / png {tot_png/n:.1f}MB / {el/n:.1f}s")
        if rem > 0:
            print(f"残り {rem:,} study: 約 {rem*el/n/3600:.1f} 時間 / "
                  f"+{rem*tot_png/n/1e3:.1f} GB")
    print(f"累計 {len(done):,} study")


if __name__ == "__main__":
    main()
