# -*- coding: utf-8 -*-
"""施設ごとの所見文の基準のずれ（s13 用、2026-10-17、手順は GEO3D_EXPERIMENT.md に結果を見る前に固定）。

1. 施設: 所見文の書式で分ける（数字 → 0 にした文字 3〜5-gram の TF-IDF → SVD 100 次元 → KMeans K = 24、seed 0。ラベルは見ない）
   → outputs/geo3d/sites.csv（StudyInstanceUID, site）
2. 診断: 所見ごとに、Gold 以外の検査で「合意ラベルの目標」を「s11 の OOF のロジット」+「施設」でロジスティック回帰（L2 C = 1、
   柔らかい目標は重み y と 1 − y の 2 行）。施設を足したときの、検査で分けた 5 分割の log loss の改善と、施設の切片の標準偏差。
3. Gold58: 施設の切片を Gold 以外で推定し、Gold58 の Codex の列を「ロジット − 切片」で補正した macro AUC を補正前と比べる。

  python scripts/geo3d_site_bias.py [--oof s11] [--k 24]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("--oof", default="s11")
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

rep = pd.read_csv(ROOT / "data/raw/train.csv").set_index("StudyInstanceUID")["Report"].fillna("")
txt = rep.str.replace(r"\d", "0", regex=True)
X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=50000, sublinear_tf=True).fit_transform(txt)
Z = normalize(TruncatedSVD(100, random_state=0).fit_transform(X))
site = pd.Series(KMeans(a.k, n_init=10, random_state=0).fit_predict(Z), index=rep.index, name="site")
out = ROOT / "outputs/geo3d/sites.csv"
site.reset_index().to_csv(out, index=False)

gold = pd.read_csv(ROOT / "data/interim/labeled_studies.csv").set_index("StudyInstanceUID")[T]
tm = pd.read_csv(ROOT / "data/interim/train_meta.csv").set_index("StudyInstanceUID")
meta = pd.concat([pd.read_csv(f, usecols=["StudyInstanceUID", "Manufacturer", "ManufacturerModelName"]).drop_duplicates("StudyInstanceUID")
                  for f in sorted(glob.glob(str(ROOT / "data/interim/dicom_meta_train_shard*of4.csv")))]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
mm = meta.Manufacturer.fillna("?").str.split().str[0].str.upper().str[:7] + " " + meta.ManufacturerModelName.fillna("?").astype(str).str[:12]
print(f"→ {out}: {len(site)} 検査、{a.k} 施設")
rows = []
for s, g in site.groupby(site):
    u = g.index
    lang = tm.reindex(u).lang_guess.value_counts()
    sc = mm.reindex(u).value_counts()
    head = txt.loc[u].str.strip().str[:24].value_counts()
    rows.append({"施設": s, "検査": len(u), "Gold": int(u.isin(gold.index).sum()), "言語": f"{lang.index[0]} {lang.iloc[0] / len(u):.0%}",
                 "機種の最多": f"{sc.index[0]} {sc.iloc[0] / len(u):.0%}", "機種の数": int((sc >= 3).sum()), "書き出しの最多": repr(head.index[0])[:30]})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))

# 2. 診断
lab = pd.read_csv(ROOT / "outputs/geo3d/labels_silentfill.csv").set_index("StudyInstanceUID")
O = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(ROOT / f"outputs/geo3d/{a.oof}/oof_fold*.csv")))]).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T]
ng = lab.index.intersection(O.index)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
S = pd.get_dummies(site.reindex(ng)).astype(float).values
kf = KFold(5, shuffle=True, random_state=0)


def fit(Xf, y, w):
    X2, y2 = np.vstack([Xf, Xf]), np.r_[np.ones(len(y)), np.zeros(len(y))]
    w2 = np.r_[y * w, (1 - y) * w]
    return LogisticRegression(C=1.0, max_iter=2000).fit(X2, y2, sample_weight=w2)


def ll(m, Xf, y, w):
    p = np.clip(m.predict_proba(Xf)[:, 1], 1e-6, 1 - 1e-6)
    return float(np.sum(w * -(y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


res, B = [], {}
for t in T:
    y = lab.loc[ng, t].values.astype(float)
    w = lab.loc[ng, t + "__w"].values.astype(float)
    x0 = lg(O.loc[ng, t].values)[:, None]
    x1 = np.hstack([x0, S])
    l0 = l1 = 0.0
    for tr, te in kf.split(x0):
        l0 += ll(fit(x0[tr], y[tr], w[tr]), x0[te], y[te], w[te]) * len(te)
        l1 += ll(fit(x1[tr], y[tr], w[tr]), x1[te], y[te], w[te]) * len(te)
    l0, l1 = l0 / len(ng), l1 / len(ng)
    m = fit(x1, y, w)
    b = m.coef_[0][1:]
    b = b - np.average(b, weights=S.sum(0))       # 施設の検査数で重み付けた平均を 0 に
    B[t] = pd.Series(b, index=sorted(site.unique()))
    res.append({"所見": t, "log loss（画像のみ）": l0, "＋施設": l1, "改善": l0 - l1, "改善 %": 100 * (l0 - l1) / l0,
                "施設の切片 SD": float(np.sqrt(np.average(b ** 2, weights=S.sum(0)))), "最小": b.min(), "最大": b.max(),
                "画像の係数": m.coef_[0][0]})
R = pd.DataFrame(res).set_index("所見")
print(f"\n診断（Gold 以外 {len(ng)} 検査、目標 = 合意ラベル、画像 = {a.oof} の OOF、検査で分けた 5 分割）:")
print(R.round(4).to_string())
pd.DataFrame(B).to_csv(ROOT / "outputs/geo3d/site_bias_diag.csv")

# 3. Gold58: Codex の列を施設の切片で補正
gc = pd.read_csv(ROOT / "outputs/working/codex_report_labels/gold_codex_labels.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T].reindex(gold.index)
gs = site.reindex(gold.index)
C0 = lg(gc.values)
C1 = C0 - np.stack([B[t].reindex(gs.values).values for t in T], 1)      # 施設のずれ（ロジットの切片）を引く
Y = gold.values


def macro(P, ix):
    return np.nanmean([roc_auc_score(Y[ix, j], P[ix, j]) if 0 < Y[ix, j].sum() < len(ix) else np.nan for j in range(len(T))])


ix = np.arange(len(Y))
m0, m1 = macro(C0, ix), macro(C1, ix)
rng = np.random.default_rng(0)
d = [macro(C1, b) - macro(C0, b) for b in (rng.integers(0, len(Y), len(Y)) for _ in range(a.boot))]
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"\nGold58 の Codex の列: 補正前 {m0:.4f} / 施設の切片で補正 {m1:.4f}（差 {m1 - m0:+.4f} [{lo:+.4f}, {hi:+.4f}]）")
print("  所見ごと（補正前 → 後）:", {t: f"{roc_auc_score(Y[:, j], C0[:, j]):.3f}→{roc_auc_score(Y[:, j], C1[:, j]):.3f}" for j, t in enumerate(T)})
print(f"  Gold58 の施設の数 {gs.nunique()}（検査数の分布 {gs.value_counts().values.tolist()}）")
