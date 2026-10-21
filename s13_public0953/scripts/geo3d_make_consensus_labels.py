# -*- coding: utf-8 -*-
"""複数の所見文ラベルの合意から、学習の目標値と欄ごとの重みを作る（s8 用、2026-10-13）。Gold58 は見ない。

出典: Codex（自前）と、GPT 系 3 種（GPT v2、lixin73 gpt56sol、steven v4 blend）。
GPT 系 3 種は互いに 95〜99% 一致するので、1 系統として平均してから Codex と並べる。
  - 所見文に記載あり（GPT v2 の判定が YES / NO）の欄:
      目標 = (1 - mix) * Codex + mix * GPT 系の平均
      重み = 1 - (1 - w_floor) * clip(|Codex - GPT 系の平均| / 0.5, 0, 1)
    食い違いはほぼすべて「Codex は低く、GPT 系は高い」向き（軽い変性・少量の関節液・捻挫など、陽性の線引きの違い）。
  - 記載なし（UNK）の欄と GPT v2 が無い検査: Codex のまま、重み 1（s4 と同じ）。
  - Gold58 の検査は学習側で正解に置き換わるので、ここでは作らない。
  - --claude を付けると、重みの食い違いを「Codex・GPT 系・Claude（崎さんの claude_all をコピーしたもの）の最大 − 最小」にする
    （s10 用、2026-10-14 ユーザー案「出典が一致するほど強く学習させる」）。目標は変えない。
    Claude の対応: present は definite 1.0 / probable 0.8 / possible 0.6、absent 0、silent は欠損（その欄は Codex と GPT 系だけ）。

  - --entropy を付けると、食い違いをエントロピーで測る（s10 用、2026-10-14 ユーザー指示）:
      JSD = H(系統の平均) − 系統ごとの H の平均（ビット）。全体のエントロピーのうち、系統どうしの食い違いから来る分。
      重み = 1 − (1 − w_floor) · JSD / JSD の最大（3 系統なら H(1/3)、2 系統なら 1）
    平均のエントロピー H(系統の平均) そのものは使わない。全系統が陽性で一致していても、確率が 0.7 と 1.0 のように違えば重みが下がるため。

  - --syn_from_eff F を付けると、滑膜炎が記載なし（GPT v2 が UNK）の欄の目標を F × 関節液の目標にする（2026-10-14 夜）。
    滑膜炎は所見文に書かれにくく（記載なしが 84%）、記載なし = 陰性として学ぶと偽陰性が多い（ディスカッション 735304 など、
    崎さんの docs/24 と 26 も同じ結論）。画像で滑膜炎に最も近いのは関節液。F = 0.5 は当てはめていない丸めた値で、
    明記された陽性（目標の平均 0.83）> 記載なし・関節液あり > 記載なし・関節液なし ≈ 明記された陰性 の順になる。

  python scripts/geo3d_make_consensus_labels.py outputs/geo3d/labels_consensus.csv [--mix 0.5] [--w_floor 0.5]
  python scripts/geo3d_make_consensus_labels.py outputs/geo3d/labels_consensus_syn.csv --syn_from_eff 0.5
  python scripts/geo3d_make_consensus_labels.py outputs/geo3d/labels_consensus_strong.csv --w_floor 0.2 --claude
  python scripts/geo3d_make_consensus_labels.py outputs/geo3d/labels_consensus_jsd.csv --w_floor 0 --claude --entropy
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--mix", type=float, default=0.5)
ap.add_argument("--w_floor", type=float, default=0.5)
ap.add_argument("--claude", action="store_true")
ap.add_argument("--entropy", action="store_true")
ap.add_argument("--syn_from_eff", type=float, default=0.0)
a = ap.parse_args()
CK = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa", "pf_oa",
      "effusion", "synovitis", "bakers", "contusion", "fracture"]
CV = {("present", "definite"): 1.0, ("present", "probable"): 0.8, ("present", "possible"): 0.6,
      ("absent", "definite"): 0.0, ("absent", "probable"): 0.2, ("absent", "possible"): 0.4}


def load(f):
    return pd.read_csv(ROOT / f).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")


cod = load("outputs/working/codex_report_labels/non_gold_codex_labels.csv")[T].astype(float)
v2 = load("data/external/llm_labels/report_labels_v2.csv")
gpt = [v2[T], load("data/external/llm_labels_lixin73/labels_llm_gpt56sol.csv")[T],
       load("data/external/llm_labels_steven/llm_labels_v4_blend.csv")[T]]
idx = cod.index
G = sum(g.reindex(idx).astype(float) for g in gpt) / len(gpt)
ment = pd.DataFrame(v2.reindex(idx)[[t + "__verdict" for t in T]].isin(["YES", "NO"]).values, index=idx, columns=T)
ment &= G.notna()
d = (cod - G).abs()


def H(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


if a.claude:
    rows = {}
    for line in open(ROOT / "data/external/claude_labels_saki/claude_all.jsonl"):
        r = json.loads(line)
        rows[r["uid"]] = [CV.get((r["raw"][k]["status"], r["raw"][k]["certainty"]), np.nan) for k in CK]
    cla = pd.DataFrame.from_dict(rows, orient="index", columns=T).reindex(idx)
    F = np.stack([cod.values, G.values, cla.values])
    d = pd.DataFrame(np.fmax(d.values, np.nanmax(F, 0) - np.nanmin(F, 0)), index=idx, columns=T)
else:
    F = np.stack([cod.values, G.values])
if a.entropy:
    n = (~np.isnan(F)).sum(0)
    jsd = H(np.nanmean(F, 0)) - np.nanmean(np.where(np.isnan(F), np.nan, H(np.nan_to_num(F))), 0)
    d = pd.DataFrame(0.5 * np.clip(jsd / np.where(n == 3, H(1 / 3), 1.0), 0, 1), index=idx, columns=T)  # 下の式の /0.5 と打ち消す
Y = cod.where(~ment, (1 - a.mix) * cod + a.mix * G)
W = pd.DataFrame(1.0, index=idx, columns=T).where(~ment, 1 - (1 - a.w_floor) * (d / 0.5).clip(0, 1))
if a.syn_from_eff:
    unk = ~v2.reindex(idx)["Synovitis__verdict"].isin(["YES", "NO"])
    Y.loc[unk, "Synovitis"] = a.syn_from_eff * Y.loc[unk, "Effusion"]
out = pd.concat([Y, W.add_suffix("__w")], axis=1)
out.index.name = "StudyInstanceUID"
out.reset_index().to_csv(ROOT / a.out, index=False)
print(f"{a.out}: {len(out)} 検査、mix {a.mix}、重みの下限 {a.w_floor}、Claude {'あり' if a.claude else 'なし'}、食い違い {'JSD' if a.entropy else '差'}")
print("所見               記載あり  目標が 0.1 以上動いた欄  重みの平均（記載あり）  目標の平均 Codex → 新")
for t in T:
    m = ment[t]
    print(f"  {t:18s} {m.mean():.3f}     {((Y[t] - cod[t]).abs() > 0.1).mean():.3f}              "
          f"{W.loc[m, t].mean():.3f}               {cod[t].mean():.3f} → {Y[t].mean():.3f}")
