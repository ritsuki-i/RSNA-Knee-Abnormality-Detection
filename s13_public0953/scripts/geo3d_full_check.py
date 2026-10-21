# -*- coding: utf-8 -*-
"""全データ学習（--fold -1、seed 違い）の提出前の確認（2026-10-17、条件は GEO3D_EXPERIMENT.md に結果を見る前に固定）。

  Gold58 はすべて学習に入っているので、精度の評価はできない。学習が壊れていないかだけを見る:
    1. 各 seed の最後のエポック（ep15）の損失が有限で、同じ設定の 5 fold の最後の損失の平均 + 0.05 以下
    2. seed どうしで重みが違う（--seed が効いている）
  終了コード 0 = 提出してよい、1 = 止める。

  python scripts/geo3d_full_check.py s11 s11full_s0,s11full_s1,...
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
base, runs = sys.argv[1], sys.argv[2].split(",")


def last_loss(p):
    m = re.findall(r"ep15 done loss (\S+)", open(p).read())
    return float(m[-1]) if m else float("nan")


ref = [last_loss(ROOT / f"outputs/geo3d/{base}/fold{k}.log") for k in range(5)]
lim = float(np.mean(ref)) + 0.05
print(f"{base} の 5 fold の最後の損失: {np.round(ref, 4).tolist()}（平均 {np.mean(ref):.4f}、上限 {lim:.4f}）")
ok = bool(np.isfinite(ref).all())
heads = []
for r in runs:
    v = last_loss(ROOT / f"outputs/geo3d/{r}/fold-1.log")
    good = bool(np.isfinite(v) and v <= lim)
    ok &= good
    sd = torch.load(ROOT / f"outputs/geo3d/{r}/fold-1.pt", map_location="cpu", weights_only=False)["model"]
    heads.append(torch.cat([t.float().flatten() for k, t in sorted(sd.items()) if k.startswith("head")]))
    print(f"  {r}: 最後の損失 {v:.4f} → {'OK' if good else 'NG'}")
diff = [float((heads[i] - heads[j]).abs().max()) for i in range(len(heads)) for j in range(i + 1, len(heads))]
distinct = len(diff) > 0 and min(diff) > 0
ok &= distinct
print(f"  seed どうしのヘッドの重みの差の最小 {min(diff) if diff else float('nan'):.3g} → {'OK' if distinct else 'NG（seed が効いていない）'}")
print(f"判定: {'提出してよい' if ok else '止める'}")
raise SystemExit(0 if ok else 1)
