# -*- coding: utf-8 -*-
"""geo3d の入力づくり（PNG キャッシュ / DICOM 直読みの共通部分）と、全スライスでの推論。"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .geometry import CACHE, flip_axes, pass_subsets, select_series, token_coords


def prepare_series_images(raw, side, r0, c0):
    """raw: (n, 384, 384) uint8（位置順）→ 右膝なら左膝の見え方へ反転したもの。"""
    fh, fv = flip_axes(r0, c0, side)
    if fh:
        raw = raw[:, :, ::-1]
    if fv:
        raw = raw[:, ::-1, :]
    return np.ascontiguousarray(raw), fh, fv


def stack3(raw, idx):
    """位置順の画像 raw (n, H, W) から、スライス idx の [前, 中, 後] を (len, 3, H, W) で返す。"""
    n = raw.shape[0]
    return np.stack([raw[np.clip(idx - 1, 0, n - 1)], raw[idx], raw[np.clip(idx + 1, 0, n - 1)]], 1)


def crop_resize(x, crop, size):
    """x: (k, 3, 384, 384) → (k, 3, size, size)"""
    y0, x0, h, w = crop
    if (y0, x0, h, w) == (0, 0, CACHE, CACHE) and size == CACHE:
        return x
    k = x.shape[0]
    flat = x.reshape(k * 3, CACHE, CACHE)[:, y0:y0 + h, x0:x0 + w]
    out = np.stack([cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR) for a in flat])
    return out.reshape(k, 3, size, size)


def series_coords(s, idx, grid, crop, fh, fv, center, side):
    """s に "box"（(n, 4)、物理切り出しのキャッシュ）があれば、各スライスの元画像上の範囲として使う。"""
    box = s.get("box")
    return np.stack([token_coords(s["p0"][i], s["r"][i], s["c"][i], s["ps"][i], s["rows"][i], s["cols"][i],
                                  grid, crop, fh, fv, center, side, None if box is None else box[i]) for i in idx])


def center_crop(scale):
    """中央を一辺 scale 倍で切り出す範囲（TTA 用）。"""
    h = int(round(CACHE * scale))
    o = (CACHE - h) // 2
    return (o, o, h, h)


@torch.no_grad()
def infer_study(model, series_list, center, side, size, per_series, max_series, device, chunk=32,
                amp_dtype=torch.float16, crops=(1.0,)):
    """1 検査の全スライスで推論する。

    series_list: [dict(type, raw=(n,384,384) uint8 位置順, p0, r, c, ps, rows, cols)]
    学習と同じ「シリーズごとに per_series 枚」の組を、刻みをずらして全スライスが 1 回ずつ入るように作り、
    各組のロジットを平均する。エンコーダは全スライスで 1 回だけ回す。
    crops: 中央切り出しの倍率の組（TTA）。各倍率のロジットを平均する。
    戻り値: dict(study=検査ヘッドのロジット, inst=スライスの補助ヘッドのロジットの全スライス平均)
    """
    outs = [_infer_one(model, series_list, center, side, size, per_series, max_series, device, chunk, amp_dtype,
                       center_crop(c)) for c in crops]
    return dict(study=np.mean([o["study"] for o in outs], 0), inst=np.mean([o["inst"] for o in outs], 0))


@torch.no_grad()
def _infer_one(model, series_list, center, side, size, per_series, max_series, device, chunk, amp_dtype, crop):
    types = [s["type"] for s in series_list]
    counts = [s["raw"].shape[0] for s in series_list]
    keep = select_series(types, counts, max_series)
    grid = size // 32
    feats_by_series, subsets, insts = [], [], []
    for k in keep:
        s = series_list[k]
        raw, fh, fv = prepare_series_images(s["raw"], side, s["r"][0], s["c"][0])
        n = raw.shape[0]
        idx = np.arange(n)
        coords = torch.from_numpy(series_coords(s, idx, grid, crop, fh, fv, center, side)).to(device)
        t = torch.full((n,), s["type"], dtype=torch.long, device=device)
        mems = []
        for a in range(0, n, chunk):
            x = torch.from_numpy(crop_resize(stack3(raw, idx[a:a + chunk]), crop, size)).to(device)
            with torch.autocast("cuda", dtype=amp_dtype):
                f, inst = model.encode(x)
                mems.append(model.memory(f, coords[a:a + chunk], t[a:a + chunk]).float())
            insts.append(inst.float())
        feats_by_series.append(torch.cat(mems))
        subsets.append(pass_subsets(n, per_series))
    n_pass = max(len(s) for s in subsets)
    logits = []
    for p in range(n_pass):
        mem = torch.cat([f[sub[p % len(sub)]].flatten(0, 1) for f, sub in zip(feats_by_series, subsets)])[None]
        with torch.autocast("cuda", dtype=amp_dtype):
            logits.append(model.decode(mem, None).float())
    return dict(study=torch.cat(logits).mean(0).cpu().numpy(), inst=torch.cat(insts).mean(0).cpu().numpy())
