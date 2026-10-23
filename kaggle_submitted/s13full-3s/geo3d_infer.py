# -*- coding: utf-8 -*-
"""RSNA Knee: geo3d (GeoQueryNet) 推論。Kaggle Models の重みと src を読み、DICOM を直接読んで推論する。"""
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image

t_start = time.time()


def find_marker(name, roots=("/kaggle/input",), max_depth=7, skip=("rsna-knee-abnormality-detection",)):
    """/kaggle/input 全体を再帰で探さない（学習の DICOM が 80 万枚ある）。深さを限って目印のファイルを探す。"""
    for root in roots:
        frontier = [root]
        for _ in range(max_depth):
            nxt = []
            for d in frontier:
                try:
                    entries = os.listdir(d)
                except OSError:
                    continue
                if name in entries:
                    return os.path.join(d, name)
                for e in entries:
                    p = os.path.join(d, e)
                    if e in skip or not os.path.isdir(p):
                        continue
                    nxt.append(p)
            frontier = nxt
    raise SystemExit(f"{name} が見つからない: {os.listdir('/kaggle/input')}")


W = os.path.dirname(find_marker("geo3d_manifest.json"))
sys.path.insert(0, W)
from geo3d.data import infer_study  # noqa: E402
from geo3d.geometry import (build_series, parse_vec, phys_box, plane_from_normal, series_type,  # noqa: E402
                            study_frame, to_uint8, CACHE)
from geo3d.model import GeoQueryNet  # noqa: E402

COMP = next(p for p in ["/kaggle/input/rsna-knee-abnormality-detection",
                        "/kaggle/input/competitions/rsna-knee-abnormality-detection"]
            if os.path.exists(os.path.join(p, "test.csv")))
manifest = json.load(open(os.path.join(W, "geo3d_manifest.json")))
print("weights", W, os.listdir(W), "| comp", COMP, flush=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 崎さんの o2（v13 の SWA 10 本）と混ぜる場合（2026-10-15）。崎さんの infer.py を別プロセスで動かす（崎さんの提出と同じ前処理・枚数・推論）。
# 10 本・k30 を 1 枚の GPU で動かすと隠しテストで約 6 時間かかり、geo3d（約 3 時間）の後に順番に動かすと 9 時間を超えた。
# そこで 10 本を 2 枚の GPU に分ける: 2 枚目で最初から o2 の split 本、geo3d が終わったら 1 枚目で残り。確率は本数で重み付けして平均する
# （= 10 本の平均と同じ）。GPU が 1 枚なら geo3d の後に 10 本。
o2 = manifest.get("o2")
o2_jobs, t_o2 = [], time.time()


def o2_dir():
    OW = os.path.dirname(find_marker("fold9_swa.pt"))
    have = sorted(f for f in os.listdir(OW) if f.endswith("_swa.pt"))
    print("o2 weights", OW, have, flush=True)
    if have != [f"fold{i}_swa.pt" for i in range(10)]:
        raise SystemExit(f"o2 の重みがそろわない（v13 の 10 本のはず）: {have}")
    return OW


def start_o2(gpu, idx, tag):
    """o2 の重みのうち idx の本だけを使う infer.py を起動する。重みは /kaggle/working に貼ったリンクから読ませる。"""
    import subprocess
    OW = o2_dir()
    wd = os.path.abspath(f"o2w_{tag}")
    os.makedirs(wd, exist_ok=True)
    for i in idx:
        dst = os.path.join(wd, f"fold{i}_swa.pt")
        if not os.path.exists(dst):
            os.symlink(os.path.join(OW, f"fold{i}_swa.pt"), dst)
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"o2 {tag}: 重み {list(idx)} を GPU {gpu} で開始（{time.time() - t_start:.0f}s）", flush=True)
    pr = subprocess.Popen([sys.executable, os.path.join(OW, "src", "infer.py"), "--comp", COMP, "--weights", wd,
                           "--out", f"o2_{tag}.csv", "--k", str(int(o2["k"]))]
                          + (["--res", str(int(o2["res"]))] if int(o2.get("res", 384)) != 384 else []), env=env)
    return dict(proc=pr, n=len(idx), out=f"o2_{tag}.csv", tag=tag)


TWO_GPU = torch.cuda.device_count() > 1

# nartaa の公開 CoAtNet（2026-10-19、ユーザー指示「滑膜炎対策」）。公開 notebook の推論スクリプトをそのまま別プロセスで動かす
# （GPU を全部使うので、o2・geo3d より先に）。出力 /kaggle/working/raptor_raw.csv（確率、失敗した検査は 12 列とも 0.5）。
# 失敗や時間切れなら、manifest の所見も s13 + o2 のまま。
rap = manifest.get("raptor")
rap_ok = False
if rap:
    import signal
    import subprocess
    t_r = time.time()
    pr = subprocess.Popen([sys.executable, "-u", os.path.join(W, "raptor_run.py")], cwd="/kaggle/working", start_new_session=True)
    try:
        rc = pr.wait(timeout=float(rap.get("timeout_s", 7200)))
        rap_ok = rc == 0 and os.path.isfile("/kaggle/working/raptor_raw.csv")
        print(f"raptor 終わり rc {rc}（{time.time() - t_r:.0f}s）", flush=True)
    except subprocess.TimeoutExpired:
        os.killpg(pr.pid, signal.SIGKILL)       # 前処理の子プロセスごと止める
        print(f"raptor が時間内に終わらない（{time.time() - t_r:.0f}s）。滑膜炎は s13 + o2 のまま", flush=True)
    json.dump({"ok": bool(rap_ok), "seconds": round(time.time() - t_r, 1)}, open("raptor_status.json", "w"))

if o2 and TWO_GPU:
    o2_jobs.append(start_o2("1", range(int(o2.get("split", 7))), "a"))


models, cfg = [], None
for f in manifest["weights"]:
    ck = torch.load(os.path.join(W, f), map_location="cpu", weights_only=False)
    a = ck["args"]
    m = GeoQueryNet(backbone=a.get("backbone", "convnext_tiny.in12k_ft_in1k_384"), pretrained=False,
                    use_coords=not a.get("no_coords", False), head=a.get("head", "dec"),
                    gap=a.get("gap", False), pool=a.get("pool", 1), xattn=a.get("xattn", 0))
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    m.to(dev).eval()
    m.enc.to(memory_format=torch.channels_last)
    models.append((m, a))
    cfg = a
    T = ck["targets"]
# 入力の種類: 0 = 視野全体を 384 に、>0 = 元画像の中央 N mm 四方を 384 に（学習時のキャッシュと同じ）
CROP_MMS = sorted({float(a.get("crop_mm", 0.0)) for _, a in models})
print("models", len(models), "cfg", {k: cfg[k] for k in ("size", "per_series", "max_series")}, "crop_mm", CROP_MMS,
      flush=True)

test = pd.read_csv(os.path.join(COMP, "test.csv"), dtype=str)
sser = pd.read_csv(os.path.join(COMP, "test_series.csv"), dtype=str)
sample = pd.read_csv(os.path.join(COMP, "sample_submission.csv"), dtype={"StudyInstanceUID": str})
prior = np.array(manifest["prior"], dtype=np.float64)


def read_series(d):
    imgs, p0, o, ps, rows, cols, lat = {mm: [] for mm in CROP_MMS}, [], [], [], [], [], None
    boxes = {mm: [] for mm in CROP_MMS if mm > 0}
    for fp in sorted(glob.glob(os.path.join(d, "*.dcm"))):
        try:
            ds = pydicom.dcmread(fp, force=True)
            px = ds.pixel_array
            if px.ndim != 2:
                continue
            u8 = to_uint8(px)
            sp = parse_vec(list(ds.PixelSpacing), 2)
            ims, bxs = {}, {}
            for mm in CROP_MMS:
                if mm > 0:
                    y0, x0, h, w = phys_box(sp, u8.shape[0], u8.shape[1], mm)
                    bxs[mm] = (y0, x0, h, w)
                    src = u8[y0:y0 + h, x0:x0 + w]
                else:
                    src = u8
                ims[mm] = np.asarray(Image.fromarray(np.ascontiguousarray(src)).resize((CACHE, CACHE), Image.BILINEAR))
            p0.append(parse_vec(list(ds.ImagePositionPatient), 3))
            o.append(parse_vec(list(ds.ImageOrientationPatient), 6))
            ps.append(sp)
            rows.append(float(ds.Rows))
            cols.append(float(ds.Columns))
            for mm in CROP_MMS:
                imgs[mm].append(ims[mm])
                if mm > 0:
                    boxes[mm].append(bxs[mm])
            if lat is None and getattr(ds, "Laterality", None):
                lat = str(ds.Laterality)
        except Exception as e:  # noqa: BLE001
            print("skip slice", fp[-50:], type(e).__name__, flush=True)
    if not p0:
        return None, None
    p0, o, ps = np.stack(p0), np.stack(o), np.stack(ps)
    rows, cols = np.array(rows), np.array(cols)
    info = build_series(p0, o[:, :3], o[:, 3:], ps, rows, cols)
    k = info["order"]
    s = dict(raw_by_mm={mm: np.stack(v)[k] for mm, v in imgs.items()},
             box_by_mm={mm: np.array(v)[k] for mm, v in boxes.items()},
             p0=p0[k], r=o[k, :3], c=o[k, 3:], ps=ps[k], rows=rows[k], cols=cols[k],
             center=info["center"], normal=info["normal"])
    return s, lat


class StudySet(torch.utils.data.Dataset):
    def __init__(self, uids):
        self.uids = uids

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        uid = self.uids[i]
        try:
            series, lats = [], []
            for _, r in sser[sser.StudyInstanceUID == uid].iterrows():
                s, lat = read_series(os.path.join(COMP, "test_series", uid, r.SeriesInstanceUID))
                if s is None:
                    continue
                t = series_type(r.Anatomical_Plane, int(float(r.Fluid_Sensitive)) if pd.notna(r.Fluid_Sensitive) else 0)
                if t is None:
                    t = series_type(plane_from_normal(s["normal"]), 0)
                s["type"] = t
                series.append(s)
                if lat:
                    lats.append(lat)
            if not series:
                return uid, None
            cen, side = study_frame([s["center"] for s in series], lats[0] if lats else None)
            return uid, (series, cen, side)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return uid, None


uids = test.StudyInstanceUID.tolist()
def identity(x):
    return x


dl = torch.utils.data.DataLoader(StudySet(uids), batch_size=None, num_workers=4, prefetch_factor=2, collate_fn=identity)
preds = {}
BUDGET = float(manifest.get("budget_s", 8 * 3600))     # 実行の上限は 9 時間。これを過ぎたら残りは事前確率で埋めて、必ず提出ファイルを書く
for i, (uid, item) in enumerate(dl):
    p = prior.copy()
    if item is not None and time.time() - t_start < BUDGET:
        series, cen, side = item
        try:
            mix = float(manifest.get("mix", 0.0))
            crops = tuple(manifest.get("crops", [1.0]))
            outs = []
            for m, c in models:
                mm = float(c.get("crop_mm", 0.0))
                ser = [dict(s, raw=s["raw_by_mm"][mm], box=s["box_by_mm"].get(mm)) for s in series]
                outs.append(infer_study(m, ser, cen, side, c["size"], c["per_series"], c["max_series"], dev, crops=crops))
            lg = np.mean([(1 - mix) * o["study"] + mix * o["inst"] for o in outs], 0)
            p = 1.0 / (1.0 + np.exp(-lg))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    preds[uid] = p
    if i < 3 or (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(uids)} {time.time() - t_start:.0f}s", flush=True)

sub = pd.DataFrame([[u] + list(preds[u]) for u in uids], columns=["StudyInstanceUID"] + T)
cols = list(sample.columns)
sub = sub[cols]

q, q_n = None, 0
if o2:
    import gc
    import subprocess
    models.clear()
    m = outs = None
    gc.collect()
    torch.cuda.empty_cache()
    rest = range(int(o2.get("split", 7)), 10) if TWO_GPU else range(10)
    o2_jobs.append(start_o2("0" if TWO_GPU else None, rest, "b"))
    acc, n = 0.0, 0
    for jb in o2_jobs:
        try:
            rc = jb["proc"].wait(timeout=max(60.0, BUDGET + 900 - (time.time() - t_start)))
        except subprocess.TimeoutExpired:     # 時間内に終わらない分は使わない（予定どおりなら起きない）
            jb["proc"].kill()
            print(f"o2 {jb['tag']} が時間内に終わらない（{time.time() - t_start:.0f}s）。この分は使わない", flush=True)
            continue
        if rc != 0:
            raise SystemExit(f"o2 {jb['tag']} の推論が失敗した: {rc}")
        r = pd.read_csv(jb["out"], dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[sub.StudyInstanceUID, T]
        acc, n = acc + jb["n"] * r.values, n + jb["n"]
        print(f"o2 {jb['tag']} 終わり（{jb['n']} 本、{time.time() - t_start:.0f}s）", flush=True)
    if n:
        q, q_n = pd.DataFrame(acc / n, index=sub.StudyInstanceUID, columns=T), n
        print(f"o2 は {n} 本の平均", flush=True)
    else:
        print("o2 が 1 本も間に合わない。geo3d だけで提出ファイルを書く", flush=True)
if o2 and q is not None:
    if not np.isfinite(q.values).all():
        raise SystemExit("o2 の予測に NaN がある")
    for u, v in o2.get("check", {}).items():      # 見える test は手元の o2 の予測（10 本の平均）と照合する（環境の違いで重みが正しく読めていない場合に止める）
        if u in q.index and q_n == 10:
            d = float(np.abs(q.loc[u].values - np.array(v, dtype=np.float64)).max())
            print(f"o2 照合 {u[-12:]}: 手元との差の最大 {d:.4f}", flush=True)
            if d > 0.02:
                raise SystemExit(f"o2 の予測が手元と合わない: {u} {d:.4f}")
    w = o2.get("w", 0.5)       # 数なら全所見で同じ重み、{所見: 重み} なら所見ごと
    w = np.array([float(w[t]) for t in T]) if isinstance(w, dict) else float(w)
    sub[T] = (1 - w) * sub[T].values + w * q.values
    print(f"o2 を重み {w} で平均（k {o2['k']}、o2 の開始から {time.time() - t_o2:.0f}s）", flush=True)

if rap_ok:
    # manifest の所見だけ: (1 - w) × test の中の順位（ここまでの予測）+ w × 順位（nartaa）。nartaa が失敗した検査はここまでの順位だけ
    r = pd.read_csv("/kaggle/working/raptor_raw.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").reindex(sub.StudyInstanceUID)[T]
    fail = (r.isna() | (r == 0.5)).all(axis=1).values
    print(f"raptor: {len(r)} 件のうち失敗 {int(fail.sum())}", flush=True)
    for t, wt in rap["w"].items():
        ours = sub[t].rank(pct=True).values
        rk = np.full(len(sub), np.nan)
        rk[~fail] = pd.Series(r[t].values[~fail]).rank(pct=True).values
        sub[t] = np.where(fail, ours, (1 - float(wt)) * ours + float(wt) * rk)
        print(f"raptor を {t} に重み {wt} で順位平均", flush=True)

sub.to_csv("submission.csv", index=False)
print("submission", sub.shape, f"{time.time() - t_start:.0f}s", flush=True)
print(sub.head())
