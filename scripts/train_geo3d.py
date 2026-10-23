# -*- coding: utf-8 -*-
"""GeoQueryNet の学習（1 fold）。検証 fold の検査は疑似ラベルごと学習から外す。

  python scripts/train_geo3d.py --name g1 --fold 0
  python scripts/train_geo3d.py --name g1 --fold -1      # 全データ（検証なし）

出力: outputs/geo3d/<name>/fold<k>.pt（EMA 重み）, oof_fold<k>.csv（全スライス推論）, log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geo3d.data import crop_resize, infer_study, prepare_series_images, series_coords, stack3  # noqa: E402
from geo3d.geometry import CACHE, pass_subsets, phys_box, select_series  # noqa: E402
from geo3d.model import GeoQueryNet  # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IMG_ROOT = ROOT / "data/processed/img_train_384"
cv2.setNumThreads(1)


def instance_weight(epoch, epochs, initial, final=None, hold=4):
    """Keep the auxiliary warmup signal, then linearly decay to the final epoch."""
    if final is None or epoch < hold:
        return initial
    return initial + (final - initial) * min(1.0, (epoch - hold + 1) / (epochs - hold))


_POOL = None


def _imread(path):
    a = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return a if a is not None else np.zeros((CACHE, CACHE), np.uint8)


def load_png_series(study_idx, s):
    """NFS は 1 ファイル開くのに約 50ms 待つので、スレッドで並列に読む（cv2 は GIL を離す）。"""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(16)
    d = f"{IMG_ROOT}/{study_idx:05d}/{s['series_idx']:02d}/"
    return np.stack(list(_POOL.map(_imread, [f"{d}{li:03d}.png" for li in s["slice_idx"]])))


def use_cache(index, split, crop_mm):
    """crop_mm > 0 なら物理切り出しのキャッシュ img_<split>_c<mm>_384 を読み、各シリーズに
    384 画像が元 DICOM のどの範囲かを表す box (n, 4) を付ける（座標の計算に使う）。"""
    global IMG_ROOT
    tag = f"_c{crop_mm:g}" if crop_mm > 0 else ""
    IMG_ROOT = ROOT / f"data/processed/img_{split}{tag}_384"
    if crop_mm > 0:
        for st in index.values():
            for s in st["series"]:
                s["box"] = np.array([phys_box(s["ps"][i], s["rows"][i], s["cols"][i], crop_mm) for i in range(s["n"])])


class TrainSet(Dataset):
    def __init__(self, index, uids, y, w, a, site=None):
        self.index, self.uids, self.y, self.w, self.a = index, uids, y, w, a
        self.site = site or {}          # --site_csv: 検査 → 施設の番号（Gold は「ずれなし」の行）。無ければ -1

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        a, uid = self.a, self.uids[i]
        st = self.index[uid]
        rng = np.random.default_rng()
        ser = st["series"]
        keep = select_series([s["type"] for s in ser], [s["n"] for s in ser], a.max_series)
        if len(keep) >= 4 and rng.random() < a.series_drop:
            keep.pop(rng.integers(len(keep)))
        grid = a.size // 32
        shift = rng.normal(0, a.coord_shift_mm, 3) / 80.0
        scale = rng.uniform(1 - a.coord_scale, 1 + a.coord_scale)
        imgs, coords, types = [], [], []
        for k in keep:
            s = ser[k]
            sub = pass_subsets(s["n"], a.per_series)
            idx = sub[rng.integers(len(sub))]
            # 必要なスライスとその前後だけ読む
            need = np.unique(np.clip(np.concatenate([idx - 1, idx, idx + 1]), 0, s["n"] - 1))
            raw = np.zeros((s["n"], CACHE, CACHE), np.uint8)
            raw[need] = load_png_series(st["study_idx"], {"series_idx": s["series_idx"], "slice_idx": s["slice_idx"][need]})
            raw, fh, fv = prepare_series_images(raw, st["side"], s["r"][0], s["c"][0])
            sc = rng.uniform(a.crop_min, 1.0)
            h = w = int(round(CACHE * math.sqrt(sc)))
            y0, x0 = rng.integers(0, CACHE - h + 1), rng.integers(0, CACHE - w + 1)
            crop = (int(y0), int(x0), h, w)
            x = crop_resize(stack3(raw, idx), crop, a.size).astype(np.float32)
            # 明るさ・コントラスト・ガンマ（シリーズ単位）
            g = rng.uniform(0.8, 1.25)
            x = 255.0 * (x / 255.0) ** g
            x = x * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10)
            imgs.append(np.clip(x, 0, 255).astype(np.uint8))
            c = series_coords(s, idx, grid, crop, fh, fv, st["center"], st["side"])
            coords.append((c * scale + shift).astype(np.float32))
            types.append(np.full(len(idx), s["type"], np.int64))
        return (torch.from_numpy(np.concatenate(imgs)), torch.from_numpy(np.concatenate(coords)),
                torch.from_numpy(np.concatenate(types)), torch.from_numpy(self.y[uid]), torch.from_numpy(self.w[uid]),
                torch.tensor(self.site.get(uid, -1)))


class SliceSet(Dataset):
    """予備学習用: 1 検査から 1 枚（選んだシリーズからランダム、中央 70% の範囲）を [前, 中, 後] で返す。"""

    def __init__(self, index, uids, y, w, a):
        self.index, self.uids, self.y, self.w, self.a = index, uids, y, w, a

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        a, uid = self.a, self.uids[i]
        st = self.index[uid]
        rng = np.random.default_rng()
        ser = st["series"]
        keep = select_series([s["type"] for s in ser], [s["n"] for s in ser], a.max_series)
        s = ser[keep[rng.integers(len(keep))]]
        n = s["n"]
        lo, hi = int(0.15 * n), max(int(0.85 * n), int(0.15 * n) + 1)
        j = int(rng.integers(lo, hi))
        need = np.clip(np.array([j - 1, j, j + 1]), 0, n - 1)
        raw = load_png_series(st["study_idx"], {"series_idx": s["series_idx"], "slice_idx": s["slice_idx"][need]})
        raw, _, _ = prepare_series_images(raw, st["side"], s["r"][0], s["c"][0])
        sc = rng.uniform(a.crop_min, 1.0)
        h = w = int(round(CACHE * math.sqrt(sc)))
        y0, x0 = rng.integers(0, CACHE - h + 1), rng.integers(0, CACHE - w + 1)
        x = crop_resize(raw[None], (int(y0), int(x0), h, w), a.size)[0].astype(np.float32)
        x = 255.0 * (x / 255.0) ** rng.uniform(0.8, 1.25)
        x = x * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10)
        return torch.from_numpy(np.clip(x, 0, 255).astype(np.uint8)), torch.from_numpy(self.y[uid]), torch.from_numpy(self.w[uid])


def collate(batch):
    n = max(b[0].shape[0] for b in batch)
    B = len(batch)
    s, g = batch[0][0].shape[-1], batch[0][1].shape[1]
    imgs = torch.zeros(B, n, 3, s, s, dtype=torch.uint8)
    coords = torch.zeros(B, n, g, g, 3)
    types = torch.zeros(B, n, dtype=torch.long)
    mask = torch.zeros(B, n, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b[0].shape[0]
        imgs[i, :k], coords[i, :k], types[i, :k], mask[i, :k] = b[0], b[1], b[2], True
    y = torch.stack([b[3] for b in batch]).float()
    w = torch.stack([b[4] for b in batch]).float()   # (B, 所見) の重み
    site = torch.stack([b[5] for b in batch])
    return imgs, coords, types, mask, y, w, site


def load_labels():
    cod = pd.concat([pd.read_csv(ROOT / "outputs/working/codex_report_labels/non_gold_codex_labels.csv"),
                     pd.read_csv(ROOT / "outputs/working/codex_report_labels/gold_codex_labels.csv")])
    cod = cod.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T].astype(float)
    gold = pd.read_csv(ROOT / "data/interim/labeled_studies.csv").set_index("StudyInstanceUID")[T].astype(float)
    folds = pd.read_csv(ROOT / "data/interim/folds_jackee.csv").set_index("StudyInstanceUID")["fold"]
    return cod, gold, folds


def series_list_png(st, max_series):
    ser = st["series"]
    keep = select_series([s["type"] for s in ser], [s["n"] for s in ser], max_series)
    out = []
    for s in (ser[k] for k in keep):
        d = dict(s)
        d["raw"] = load_png_series(st["study_idx"], s)
        out.append(d)
    return out


class PredSet(Dataset):
    def __init__(self, index, uids, max_series):
        self.index, self.uids, self.max_series = index, uids, max_series

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        st = self.index[self.uids[i]]
        return self.uids[i], series_list_png(st, self.max_series), st["center"], st["side"]


def _identity(x):
    return x


def predict(model, index, uids, a, device, log=None):
    model.eval()
    rows, t0 = [], time.time()
    dl = DataLoader(PredSet(index, uids, a.max_series), batch_size=None, num_workers=4, prefetch_factor=2,
                    collate_fn=_identity)
    for i, (uid, ser, center, side) in enumerate(dl):
        r = infer_study(model, ser, center, side, a.size, a.per_series, a.max_series, device,
                        amp_dtype=torch.bfloat16)
        rows.append([uid] + list(r["study"]) + list(r["inst"]))
        if log and (i + 1) % 200 == 0:
            log(f"  predict {i + 1}/{len(uids)} {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows, columns=["StudyInstanceUID"] + T + [t + "__inst" for t in T])
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    out = {}
    for name, mix in (("study", 0.0), ("inst", 1.0), ("mix", 0.5)):
        o = df[["StudyInstanceUID"]].copy()
        for t in T:
            o[t] = sig((1 - mix) * df[t] + mix * df[t + "__inst"])
        out[name] = o
    return out


def auc_report(pred, cod, gold):
    p = pred.set_index("StudyInstanceUID")
    out = {}
    g = p.index.intersection(gold.index)
    if len(g) >= 5:
        a = [roc_auc_score(gold.loc[g, t], p.loc[g, t]) for t in T if gold.loc[g, t].nunique() > 1]
        out["gold"] = float(np.mean(a)) if a else float("nan")
        out["n_gold"] = len(g)
    ng = p.index.difference(gold.index).intersection(cod.index)
    aa = []
    for t in T:
        c = cod.loc[ng, t]
        sel = (c >= 0.8) | (c <= 0.2)
        if (c[sel] >= 0.8).sum() >= 5 and (c[sel] <= 0.2).sum() >= 5:
            aa.append(roc_auc_score(c[sel] >= 0.8, p.loc[ng[sel.values], t]))
    out["clear"] = float(np.mean(aa))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--per_series", type=int, default=8)
    ap.add_argument("--max_series", type=int, default=7)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--gold_w", type=float, default=5.0)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--series_drop", type=float, default=0.25)
    ap.add_argument("--crop_min", type=float, default=0.75)
    ap.add_argument("--coord_shift_mm", type=float, default=4.0)
    ap.add_argument("--coord_scale", type=float, default=0.05)
    ap.add_argument("--no_coords", action="store_true")
    ap.add_argument("--backbone", default="convnext_tiny.in12k_ft_in1k_384")
    ap.add_argument("--crop_mm", type=float, default=0.0, help="物理切り出しのキャッシュを使う（例 140）。0 = 視野全体")
    ap.add_argument("--head", default="mil", choices=["dec", "mil"])
    ap.add_argument("--gap", action="store_true")
    ap.add_argument("--pool", type=int, default=1)
    ap.add_argument("--xattn", type=int, default=0, help="MIL の前のトークン間自己注意の層数")
    ap.add_argument("--inst_w", type=float, default=1.0, help="スライスごとの補助損失の重み")
    ap.add_argument("--inst_w_final", type=float, default=None, help="補助損失を最終epochでこの重みまで線形減衰。未指定は従来の固定値")
    ap.add_argument("--inst_hold_epochs", type=int, default=4, help="補助損失の初期重みを保つepoch数")
    ap.add_argument("--warm_steps", type=int, default=0, help="本学習の前のスライス単体の予備学習のステップ数")
    ap.add_argument("--unk_w", type=float, default=1.0, help="所見文に記載なし（GPT v2 で UNK）の欄の損失の重み")
    ap.add_argument("--unk_teacher", default="", help="記載なし（UNK）の欄の目標値を置き換える教師の予測 CSV（OOF）")
    ap.add_argument("--unk_mix", type=float, default=1.0, help="UNK の欄の目標 = (1-mix)*Codex + mix*教師")
    ap.add_argument("--site_csv", default="", help="s13: 検査 → 施設（outputs/geo3d/sites.csv）。Gold 以外の損失に施設 × 所見のずれ b を足す（推論では使わない）")
    ap.add_argument("--site_sd", type=float, default=0.5, help="b の事前分布 N(0, sd²) に相当する L2")
    ap.add_argument("--label_csv", default="", help="Gold 以外の目標値と欄ごとの重み（列 T と T__w）。geo3d_make_consensus_labels.py が作る")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="動作確認用: 学習に使う検査数の上限")
    ap.add_argument("--val_limit", type=int, default=0)
    ap.add_argument("--quick_val", type=int, default=150, help="簡易検証に使う検査数")
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--rank_w", type=float, default=0.0, help="> 0: 検査ヘッドに所見ごとの対の順位損失を足す（2026-10-22）")
    ap.add_argument("--rank_margin", type=float, default=0.5, help="目標の差がこれを超える対だけ順序を課す")
    ap.add_argument("--rank_queue", type=int, default=256, help="対の相手に使う直近の検査の数（ロジットは勾配を切って保持）")
    ap.add_argument("--init_ckpt", default="", help="この重み（EMA）から学習を始める。隣の *_site_b.npy があれば施設のずれも引き継ぐ")
    ap.add_argument("--seed", type=int, default=None, help="全データ学習の seed 違い用（torch の初期値・シャッフル順）。指定なしは今までどおり")
    a = ap.parse_args()
    if a.inst_w < 0 or (a.inst_w_final is not None and a.inst_w_final < 0):
        ap.error("instance loss weights must be nonnegative")
    if a.inst_w_final is not None and not 0 <= a.inst_hold_epochs < a.epochs:
        ap.error("--inst_hold_epochs must be >= 0 and less than --epochs")
    if a.seed is not None:
        torch.manual_seed(a.seed)

    out = ROOT / "outputs/geo3d" / a.name
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"fold{a.fold}.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    log(f"args {json.dumps(vars(a))}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    index = pickle.load(open(ROOT / "data/interim/geo3d_index_train.pkl", "rb"))
    use_cache(index, "train", a.crop_mm)
    log(f"画像 {IMG_ROOT}")
    cod, gold, folds = load_labels()
    uids = [u for u in index if u in folds.index and (u in cod.index or u in gold.index)]
    tr = [u for u in uids if folds[u] != a.fold]
    va = [u for u in uids if folds[u] == a.fold]
    if a.limit:
        tr = list(np.random.default_rng(0).permutation(tr)[:a.limit])
    if a.val_limit:
        va = va[:a.val_limit]
    y = {}
    w = {}
    v2 = pd.read_csv(ROOT / "data/external/llm_labels/report_labels_v2.csv").set_index("StudyInstanceUID")
    teacher = pd.read_csv(ROOT / a.unk_teacher).set_index("StudyInstanceUID")[T] if a.unk_teacher else None
    lab = pd.read_csv(ROOT / a.label_csv).set_index("StudyInstanceUID") if a.label_csv else None
    n_rep = n_lab = 0
    for u in uids:
        if u in gold.index:
            y[u], w[u] = gold.loc[u, T].values.astype(np.float32), np.full(len(T), a.gold_w, np.float32)
        else:
            y[u], w[u] = cod.loc[u, T].values.astype(np.float32), np.ones(len(T), np.float32)
            if lab is not None and u in lab.index:
                # 複数の所見文ラベルの合意（目標）と食い違い（重み）。記載なしの欄は Codex のまま
                y[u] = lab.loc[u, T].values.astype(np.float32)
                w[u] = lab.loc[u, [t + "__w" for t in T]].values.astype(np.float32)
                n_lab += 1
            if a.unk_w != 1.0 and u in v2.index:
                # 所見文に記載なし（GPT v2 の判定が UNK）の欄は、Codex が低い確率を付けていても陰性とは限らない
                # （Gold58 で滑膜炎の記載なしの 34% が陽性）。値は変えず、重みだけ下げる。
                w[u][v2.loc[u, [t + "__verdict" for t in T]].values == "UNK"] = a.unk_w
            if teacher is not None and u in v2.index and u in teacher.index:
                # 記載なしの欄は、Codex のほぼ 0 の代わりに自前モデルの OOF 予測を目標にする（自己蒸留）。
                # 教師の OOF は別 fold のモデルの予測なので、その fold の Gold も間接的に効く（Gold OOF はやや甘くなりうる）。
                m = v2.loc[u, [t + "__verdict" for t in T]].values == "UNK"
                y[u][m] = (1 - a.unk_mix) * y[u][m] + a.unk_mix * teacher.loc[u, T].values.astype(np.float32)[m]
                n_rep += int(m.sum())
    if teacher is not None:
        log(f"UNK の欄を教師で置き換え: {n_rep} 欄（mix {a.unk_mix}）")
    if lab is not None:
        log(f"目標と重みを {a.label_csv} から: {n_lab} 検査（重みの平均 {np.mean([w[u] for u in uids if u not in gold.index]):.3f}）")
    log(f"train {len(tr)} (gold {sum(u in gold.index for u in tr)}) | val {len(va)} (gold {sum(u in gold.index for u in va)})")

    site = None
    if a.site_csv:
        sc = pd.read_csv(ROOT / a.site_csv).set_index("StudyInstanceUID")["site"]
        n_sites = int(sc.max()) + 1
        # Gold58 は b の最後の行（常に 0 = テストの読影者の基準）を使う
        site = {u: (n_sites if u in gold.index else int(sc[u])) for u in tr}
        log(f"施設のずれ: {n_sites} 施設 × {len(T)} 所見、事前分布の sd {a.site_sd}（Gold {sum(v == n_sites for v in site.values())} 検査はずれなし）")
    ds = TrainSet(index, tr, y, w, a, site)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                    drop_last=True, persistent_workers=True, prefetch_factor=4)
    mk = dict(backbone=a.backbone, use_coords=not a.no_coords, head=a.head, gap=a.gap, pool=a.pool, xattn=a.xattn)
    model = GeoQueryNet(**mk).to(device)
    model.enc.to(memory_format=torch.channels_last)
    prior = np.clip(np.mean([y[u] for u in tr], 0), 0.02, 0.98)
    with torch.no_grad():
        model.head_b.copy_(torch.from_numpy(np.log(prior / (1 - prior))).float())
        model.inst_head.bias.copy_(torch.from_numpy(np.log(prior / (1 - prior))).float())
    model.enc.set_grad_checkpointing(True)
    if a.init_ckpt:
        ck0 = torch.load(ROOT / a.init_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict({k: v.float() for k, v in ck0["model"].items()})
        log(f"重みを {a.init_ckpt} から読み込んだ")
    if a.warm_steps > 0:
        # 予備学習: 32 検査から 1 枚ずつのスライスで、エンコーダとスライスヘッドだけを学習する。
        # 検査単位の学習は立ち上がる時期が実行ごとに大きくばらつく（2 エポック後に clear 0.53〜0.81）ため、
        # 立ち上がった状態から本学習を始める。
        sdl = DataLoader(SliceSet(index, tr, y, w, a), batch_size=32, shuffle=True, num_workers=a.workers,
                         drop_last=True, persistent_workers=True, prefetch_factor=4)
        wp = list(model.inst_norm.parameters()) + list(model.inst_head.parameters())
        wopt = torch.optim.AdamW([{"params": list(model.enc.parameters()), "lr": a.lr},
                                  {"params": wp, "lr": 1e-3}], weight_decay=0.01)
        model.train()
        it, run, t0 = 0, [], time.time()
        while it < a.warm_steps:
            for xb, yb, wb in sdl:
                f = min(1.0, (it + 1) / 50)
                wopt.param_groups[0]["lr"], wopt.param_groups[1]["lr"] = a.lr * f, 1e-3 * f
                xb, yb, wb = xb.to(device, non_blocking=True), yb.to(device).float(), wb.to(device).float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, inst = model.encode(xb)
                loss = (F.binary_cross_entropy_with_logits(inst.float(), yb, reduction="none") * wb).sum() / wb.sum()
                wopt.zero_grad(set_to_none=True)
                loss.backward()
                wopt.step()
                run.append(loss.item())
                it += 1
                if it % 100 == 0:
                    log(f"warm {it}/{a.warm_steps} loss {np.mean(run[-100:]):.4f} {time.time() - t0:.0f}s")
                if it >= a.warm_steps:
                    break
        del wopt, sdl
    enc_p = list(model.enc.parameters())
    enc_ids = {id(p) for p in enc_p}
    head_p = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": enc_p, "lr": a.lr}, {"params": head_p, "lr": a.lr_head}]
    site_b = None
    if site is not None:
        site_b = torch.nn.Parameter(torch.zeros(n_sites, len(T), device=device))
        groups.append({"params": [site_b], "lr": a.lr_head, "weight_decay": 0.0})
        site_lam = 1.0 / (2 * a.site_sd ** 2 * len(tr))      # 1 検査あたりの平均の損失に足す L2（MAP 推定の事前分布）
        sb_p = (ROOT / a.init_ckpt).with_name(Path(a.init_ckpt).stem + "_site_b.npy") if a.init_ckpt else None
        if sb_p is not None and sb_p.exists():
            b0 = np.load(sb_p)
            if b0.shape == tuple(site_b.shape):
                with torch.no_grad():
                    site_b.copy_(torch.from_numpy(b0).to(device))
                log(f"施設のずれ b を {sb_p.name} から引き継いだ")
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    steps = a.epochs * (len(dl) // a.accum)
    warm = max(1, int(0.05 * steps))
    base = [a.lr, a.lr_head] + ([a.lr_head] if site_b is not None else [])

    def set_lr(step):
        f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f

    ema = GeoQueryNet(pretrained=False, **mk).to(device)
    ema.load_state_dict(model.state_dict())
    ema.enc.to(memory_format=torch.channels_last)
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(1)
    qv = list(rng.permutation(va)[:a.quick_val]) if va else []
    qv = list(dict.fromkeys([u for u in va if u in gold.index] + qv))   # 検証 fold の Gold は必ず含める

    from collections import deque
    rq = deque()          # 順位損失の相手: 直近の検査の (ロジット（勾配なし）, 目標, 重み)

    def rank_term(sc, yy, ww):
        """所見ごとに、目標の差が rank_margin を超える対 (i, j) で softplus(-(s_i - s_j))。重み w_i w_j。
        対はバッチの中どうし（上の側だけ数える）と、バッチ × 直近の検査（勾配はバッチ側だけ）。所見ごとに平均し、所見で平均。"""
        num = torch.zeros(sc.shape[1], device=sc.device)
        den = torch.zeros(sc.shape[1], device=sc.device)
        d = yy[:, None, :] - yy[None, :, :]
        m = (d > a.rank_margin).float() * ww[:, None, :] * ww[None, :, :]
        num = num + (F.softplus(-(sc[:, None, :] - sc[None, :, :])) * m).sum((0, 1))
        den = den + m.sum((0, 1))
        if rq:
            qs = torch.cat([q[0] for q in rq])
            qy = torch.cat([q[1] for q in rq])
            qw = torch.cat([q[2] for q in rq])
            d = yy[:, None, :] - qy[None]
            wp = ww[:, None, :] * qw[None]
            df = sc[:, None, :] - qs[None]
            mp, mn = (d > a.rank_margin).float() * wp, (d < -a.rank_margin).float() * wp
            num = num + (F.softplus(-df) * mp + F.softplus(df) * mn).sum((0, 1))
            den = den + (mp + mn).sum((0, 1))
        ok = den > 0
        return (num[ok] / den[ok]).mean() if ok.any() else sc.sum() * 0.0

    step = 0
    rank_tot = 0.0
    for ep in range(a.epochs):
        inst_w = instance_weight(ep, a.epochs, a.inst_w, a.inst_w_final, a.inst_hold_epochs)
        log(f"ep{ep} instance_weight {inst_w:.6f}")
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, (imgs, coords, types, mask, yb, wb, sb) in enumerate(dl):
            imgs, coords, types, mask = imgs.to(device, non_blocking=True), coords.to(device), types.to(device), mask.to(device)
            yb, wb = yb.to(device), wb.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, inst = model(imgs, coords, types, mask)
            lg, inst = lg.float(), inst.float()
            if site_b is not None:
                # 所見文の施設ごとの線引きのずれ（ロジットの切片）。Gold は最後の 0 の行
                off = F.pad(site_b, (0, 0, 0, 1))[sb.to(device)]
                lg, inst = lg + off, inst + off[:, None, :]
            loss = (F.binary_cross_entropy_with_logits(lg, yb, reduction="none") * wb).sum() / wb.sum()
            if site_b is not None:
                loss = loss + site_lam * (site_b ** 2).sum()
            if inst_w > 0:
                li = F.binary_cross_entropy_with_logits(inst, yb[:, None, :].expand_as(inst), reduction="none")
                li = (li * mask[..., None]).sum(1) / mask.sum(1)[:, None]          # (B, 所見): スライス平均
                loss = loss + inst_w * (li * wb).sum() / wb.sum()
            if a.rank_w > 0:
                lr_ = rank_term(lg, yb, wb)
                loss = loss + a.rank_w * lr_
                rank_tot += lr_.item()
                rq.append((lg.detach(), yb, wb))
                while sum(q[0].shape[0] for q in rq) > a.rank_queue:
                    rq.popleft()
            (loss / a.accum).backward()
            tot += loss.item()
            nb += 1
            if (it + 1) % a.accum == 0:
                set_lr(step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                d = min(a.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1 - d)
            if (it + 1) % 200 == 0:
                log(f"ep{ep} it{it + 1}/{len(dl)} loss {tot / nb:.4f} {time.time() - t0:.0f}s "
                    f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
        log(f"ep{ep} done loss {tot / max(nb, 1):.4f} {time.time() - t0:.0f}s" + (f" rank {rank_tot / max(nb, 1):.4f}" if a.rank_w > 0 else ""))
        rank_tot = 0.0
        if qv and ((ep + 1) % a.val_every == 0) and ep + 1 < a.epochs:
            pv = predict(ema, index, qv, a, device)
            r = {k: auc_report(v, cod, gold) for k, v in pv.items()}
            log(f"ep{ep} quick-val(EMA, n={len(qv)}) {json.dumps(r)}")
        torch.save({"model": {k: v.half() if v.is_floating_point() else v for k, v in ema.state_dict().items()},
                    "args": vars(a), "targets": T}, out / f"fold{a.fold}.pt")
        if site_b is not None:
            b = site_b.detach().cpu().numpy()
            np.save(out / f"fold{a.fold}_site_b.npy", b)
            log(f"ep{ep} 施設のずれ b: 所見ごとの SD {dict(zip(T, np.round(b.std(0), 3).tolist()))}")

    if va:
        log("OOF 推論（全スライス）")
        oof = predict(ema, index, va, a, device, log)
        for k, v in oof.items():
            v.to_csv(out / (f"oof_fold{a.fold}.csv" if k == "study" else f"oof_{k}_fold{a.fold}.csv"), index=False)
        log(f"OOF fold{a.fold} {json.dumps({k: auc_report(v, cod, gold) for k, v in oof.items()})}")
    log("done")


if __name__ == "__main__":
    main()
