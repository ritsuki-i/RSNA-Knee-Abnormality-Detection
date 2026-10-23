# -*- coding: utf-8 -*-
"""geo3d の OOF（outputs/geo3d/<run>/oof*_fold*.csv）を結合し、Gold58 で所見別 AUC と macro を出す。
集計は崎さんの scripts/tools/gold_oof.py と同じ（全 fold を結合してから AUC）。
出力は 3 種類: study（検査ヘッド）, inst（スライスヘッド）, mix（半々）。

  python scripts/geo3d_gold_oof.py g2 [g3 ...]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
g = pd.read_csv(ROOT / "data/interim/labeled_studies.csv").set_index("StudyInstanceUID")[T]
for run in sys.argv[1:]:
    for head, pat in (("study", "oof_fold*.csv"), ("inst", "oof_inst_fold*.csv"), ("mix", "oof_mix_fold*.csv")):
        fs = sorted(glob.glob(str(ROOT / f"outputs/geo3d/{run}/{pat}")))
        if not fs:
            continue
        o = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        idx = o.index.intersection(g.index)
        y = g.loc[idx]
        a = {t: roc_auc_score(y[t], o.loc[idx, t]) for t in T if y[t].nunique() > 1}
        print(f"{run} {head} folds={len(fs)} n_gold={len(idx)} macro={np.mean(list(a.values())):.4f}")
        print("  " + " ".join(f"{t[:8]}={v:.3f}" for t, v in a.items()))
