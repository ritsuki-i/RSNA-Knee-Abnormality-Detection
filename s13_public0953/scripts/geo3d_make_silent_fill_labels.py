# -*- coding: utf-8 -*-
"""「所見文に記載が無いのに正解は陽性」の欄への対策（s11 用、2026-10-15）。合意ラベルの表の、記載なしの欄だけを埋め直す。

対象の決め方（scripts/geo3d_silent_positive.py、Gold58 のラベルの集計。モデルの OOF の AUC は見ていない）:
  Gold58 の記載なし（GPT v2 が UNK）の欄で、正解の陽性が 2 件以上かつ陽性率 15% 以上 → PF OA、滑膜炎、挫傷、骨折。
  ほかの 8 所見は記載なしの陽性率がほぼ 0 で、記載なし = 陰性のままでよい。
埋め方（事前に決めた 2 案「関連所見の目標 × 0.5」「0.5 × ラベル + 0.5 × 画像モデルの OOF」から、
所見文ラベルの列と Gold58 の正解の AUC が上がるものだけを採る）:
  PF OA    : 0.5 × ラベル + 0.5 × OOF（0.910 → 0.932。関連所見 = 内側・外側 OA は 0.876 に下がる）
  挫傷     : 0.5 × ラベル + 0.5 × OOF（0.848 → 0.879。関連所見 = 骨折は 0.836）
  骨折     : 大きいほう（0.5 × 挫傷の目標、0.5 × ラベル + 0.5 × OOF）（0.850 → 0.921）
  滑膜炎   : 変えない（関節液 0.801、OOF 0.790 と、どちらも今の 0.810 より下がる。別の問題として扱う）
OOF は s9 の 5 fold（各検査はその検査を学習していないモデルの予測）。ただし s9 の各モデルはほかの fold の Gold58 を
学習しているので、s11 の Gold58 OOF には二次的な漏れが入る。上の選び方にも Gold58 を使ったので、対象 3 所見の
Gold58 は甘めに出る。対象外の 9 所見はラベルが同じなので、明らかなラベルと Gold58 で巻き添えの有無を公平に見られる。

  python scripts/geo3d_make_silent_fill_labels.py outputs/geo3d/labels_silentfill.csv [--base outputs/geo3d/labels_consensus.csv] [--oof s9]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--base", default="outputs/geo3d/labels_consensus.csv")
ap.add_argument("--oof", default="s9")
a = ap.parse_args()

lab = pd.read_csv(ROOT / a.base).set_index("StudyInstanceUID")
fs = sorted(glob.glob(str(ROOT / f"outputs/geo3d/{a.oof}/oof_fold*.csv")))
assert len(fs) == 5, fs
O = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T].reindex(lab.index)
assert O.notna().all().all()
v2 = pd.read_csv(ROOT / "data/external/llm_labels/report_labels_v2.csv").set_index("StudyInstanceUID").reindex(lab.index)
out = lab.copy()
for t in ["PF OA", "Contusion", "Fracture"]:
    sil = ~v2[t + "__verdict"].isin(["YES", "NO"])
    fill = 0.5 * lab.loc[sil, t] + 0.5 * O.loc[sil, t]
    if t == "Fracture":
        fill = np.maximum(fill, 0.5 * lab.loc[sil, "Contusion"])
    out.loc[sil, t] = fill
    print(f"{t:10s} 記載なし {sil.sum():5d} 欄: 目標の平均 {lab.loc[sil, t].mean():.3f} → {out.loc[sil, t].mean():.3f}、"
          f"0.2 以上 {(lab.loc[sil, t] >= 0.2).mean():.3f} → {(out.loc[sil, t] >= 0.2).mean():.3f}")
out.reset_index().to_csv(ROOT / a.out, index=False)
d = (out[T] - lab[T]).abs().max()
print(f"{a.out}: {len(out)} 検査。変わった列: {list(d[d > 0].index)}（重みの列は変えない）")
