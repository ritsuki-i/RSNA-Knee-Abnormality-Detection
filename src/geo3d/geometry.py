# -*- coding: utf-8 -*-
"""DICOM の幾何（位置・向き・画素間隔）から、特徴マップの各トークンの患者座標 [mm] を出す。

学習（PNG キャッシュ）と Kaggle 推論（DICOM 直読み）の両方から使うので、numpy だけに依存させる。

座標の約束:
  - DICOM の患者座標 (LPS): +x = 患者の左, +y = 背側, +z = 頭側。
  - 検査の中心（シリーズ中心の中央値）を原点にし、右膝は x を反転して左膝にそろえる。
    これで内側は常に x' < 0、外側は x' > 0 になる。
  - 右膝の冠状断・軸位断は、画像も x 方向に反転して左膝の見え方にそろえる（矢状断は x が法線なので反転しない）。
"""
from __future__ import annotations

import numpy as np

PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
N_TYPES = 6          # 断面 3 × 液体強調 2（Fat_Suppression は Fluid_Sensitive と全件一致）
CACHE = 384          # PNG キャッシュの一辺
COORD_SCALE = 80.0   # mm。膝の撮像範囲 (約 150mm) がおよそ [-1, 1] に入る


def parse_vec(s, n):
    """'a|b|c' や '[a, b]' 形式を float 配列にする。"""
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float64)
    s = str(s).strip().strip("[]")
    sep = "|" if "|" in s else ("\\" if "\\" in s else ",")
    v = np.array([float(x) for x in s.split(sep)], dtype=np.float64)
    if v.size != n:
        raise ValueError(f"長さ {v.size} != {n}: {s}")
    return v


def series_type(plane, fluid):
    p = PLANES.get(str(plane), None)
    if p is None:
        return None
    return p * 2 + int(fluid)


def plane_from_normal(normal):
    """Anatomical_Plane が無いときの代用: 法線の最大成分で決める。"""
    a = int(np.argmax(np.abs(normal)))
    return ["Sagittal", "Coronal", "Axial"][a]


def build_series(p0, r, c, ps, rows, cols):
    """1 シリーズ分の配列を、法線方向の位置で並べ替えて返す。

    p0: (n,3) ImagePositionPatient, r/c: (n,3) 行方向（列番号が増える向き）/ 列方向（行番号が増える向き）,
    ps: (n,2) PixelSpacing = [行間隔, 列間隔], rows/cols: (n,)
    戻り値の order は入力の並びに対する添字。
    """
    normal = np.cross(r[0], c[0])
    order = np.argsort(p0 @ normal, kind="stable")
    center = (p0 + r * (cols * ps[:, 1] / 2.0)[:, None] + c * (rows * ps[:, 0] / 2.0)[:, None]).mean(0)
    return dict(order=order, normal=normal, center=center)


def study_frame(series_centers, laterality=None):
    """検査の原点と左右（+1 = 左膝, -1 = 右膝）。タグが無ければ中心 x の符号で決める。"""
    cen = np.median(np.stack(series_centers), axis=0)
    lat = str(laterality).strip().upper() if laterality is not None else ""
    if lat == "L":
        side = 1.0
    elif lat == "R":
        side = -1.0
    else:
        side = 1.0 if cen[0] >= 0 else -1.0
    return cen, side


def flip_axes(r, c, side):
    """右膝の画像をどちら向きに反転するか。(水平, 垂直)。"""
    if side > 0:
        return False, False
    if abs(r[0]) >= abs(c[0]) and abs(r[0]) > 0.5:
        return True, False
    if abs(c[0]) > 0.5:
        return False, True
    return False, False


def phys_box(ps, rows, cols, mm):
    """元 DICOM の中央 mm 四方の範囲 (y0, x0, h, w)（画素）。視野が mm より狭い辺は視野いっぱい。
    PixelSpacing が無ければ中央の正方形。PNG キャッシュの物理切り出し版（img_*_c140_384）はこの範囲を 384 にしたもの。"""
    rows, cols = int(rows), int(cols)
    sy, sx = (float(ps[0]), float(ps[1])) if ps is not None else (0.0, 0.0)
    if not (sy > 0 and sx > 0):
        s = min(rows, cols)
        return ((rows - s) // 2, (cols - s) // 2, s, s)
    side = min(float(mm), rows * sy, cols * sx)
    h = max(1, min(rows, int(round(side / sy))))
    w = max(1, min(cols, int(round(side / sx))))
    return ((rows - h) // 2, (cols - w) // 2, h, w)


def token_coords(p0, r, c, ps, rows, cols, grid, crop, flip_h, flip_v, center, side, box=None):
    """1 スライスの特徴マップ (grid×grid) の各トークン中心の正規化座標 (grid, grid, 3)。

    crop = (y0, x0, h, w): 反転後の 384 画像から切り出した範囲（これを入力サイズに拡大して使う）。
    box = (y0, x0, h, w): 384 画像が元 DICOM のどの範囲か（画素）。None なら画像全体。
    """
    by, bx, bh, bw = box if box is not None else (0, 0, rows, cols)
    y0, x0, h, w = crop
    g = (np.arange(grid) + 0.5) / grid          # 入力画像上のトークン中心（割合）
    u = x0 + g * w - 0.5                        # 反転後の 384 画像の画素座標
    v = y0 + g * h - 0.5
    if flip_h:
        u = (CACHE - 1) - u
    if flip_v:
        v = (CACHE - 1) - v
    j = bx + (u + 0.5) * bw / CACHE - 0.5       # 元 DICOM の列番号
    i = by + (v + 0.5) * bh / CACHE - 0.5       # 元 DICOM の行番号
    pos = (p0[None, None, :]
           + (j[None, :, None] * ps[1]) * r[None, None, :]
           + (i[:, None, None] * ps[0]) * c[None, None, :])
    pos = pos - center
    pos[..., 0] *= side
    return (pos / COORD_SCALE).astype(np.float32)


def select_series(types, counts, max_series):
    """種類の重複を避けて最大 max_series 本を選ぶ。各種類でスライスの多い 1 本を優先。"""
    idx = list(range(len(types)))
    idx.sort(key=lambda k: (-counts[k], k))
    first, rest, seen = [], [], set()
    for k in idx:
        (rest if types[k] in seen else first).append(k)
        seen.add(types[k])
    return sorted((first + rest)[:max_series])


def pass_subsets(n, per_series):
    """スライス n 枚を、刻み m = ceil(n / per_series) で m 組に分ける（組 k = k, k+m, ...）。"""
    m = max(1, int(np.ceil(n / per_series)))
    return [np.arange(k, n, m) for k in range(m)]


def to_uint8(arr):
    """PNG キャッシュと同じ正規化（スライスごとに 0.5/99.5 パーセンタイル）。"""
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
