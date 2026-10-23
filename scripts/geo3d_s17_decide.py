# -*- coding: utf-8 -*-
"""s17 の採否（2026-10-22、結果を見る前に固定。GEO3D_EXPERIMENT.md「s17: 順位損失を足した追加学習」）。

  採る条件（すべて）:
    (1) Gold58 OOF の macro: s17rank > s17ctrl（追加学習だけの効果を除く）
    (2) Gold58 OOF の macro: s17rank ≥ s13（0.9190）
    (3) 明らかなラベル（Gold 以外、Codex ≥ 0.8 / ≤ 0.2）の macro: s17rank − s17ctrl ≥ −0.001
  出力: outputs/geo3d/s17_choice.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import train_geo3d as tg  # noqa: E402

T = tg.T
cod, gold, folds = tg.load_labels()


def rd(run):
    fs = sorted(glob.glob(str(ROOT / f"outputs/geo3d/{run}/oof_fold*.csv")))
    assert len(fs) == 5, (run, fs)
    return pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T]


def gold_auc(P):
    Y = (gold[T] >= 0.5).astype(int)
    a = {t: roc_auc_score(Y[t], P.loc[gold.index, t]) for t in T}
    return float(np.mean(list(a.values()))), a


def clear_auc(P):
    ng = [u for u in P.index if u in cod.index and u not in gold.index]
    a = {}
    for t in T:
        c = cod.loc[ng, t]
        sel = c[(c >= 0.8) | (c <= 0.2)]
        a[t] = roc_auc_score((sel >= 0.8).astype(int), P.loc[sel.index, t])
    return float(np.mean(list(a.values()))), a


R = {r: rd(r) for r in ["s13", "s17ctrl", "s17rank"]}
G = {r: gold_auc(P) for r, P in R.items()}
C = {r: clear_auc(P) for r, P in R.items()}
for r in R:
    print(f"{r}: Gold58 {G[r][0]:.4f}  明らかなラベル {C[r][0]:.4f}  滑膜炎 Gold {G[r][1]['Synovitis']:.3f} / 明らか {C[r][1]['Synovitis']:.3f}")
print("所見ごとの Gold58（s13 / ctrl / rank）: " + "、".join(f"{t[:8]} {G['s13'][1][t]:.3f}/{G['s17ctrl'][1][t]:.3f}/{G['s17rank'][1][t]:.3f}" for t in T))
c1 = G["s17rank"][0] > G["s17ctrl"][0]
c2 = G["s17rank"][0] >= G["s13"][0]
c3 = C["s17rank"][0] - C["s17ctrl"][0] >= -0.001
adopt = bool(c1 and c2 and c3)
print(f"判定: (1) rank > ctrl {c1}  (2) rank ≥ s13 {c2}  (3) 明らかなラベルで ctrl − 0.001 以上 {c3}  → {'採る' if adopt else '採らない'}")
json.dump({"adopt": adopt, "gold": {r: G[r][0] for r in R}, "clear": {r: C[r][0] for r in R}, "cond": [bool(c1), bool(c2), bool(c3)]},
          open(ROOT / "outputs/geo3d/s17_choice.json", "w"), indent=1)
