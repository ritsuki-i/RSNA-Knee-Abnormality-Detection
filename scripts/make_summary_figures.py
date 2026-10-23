# -*- coding: utf-8 -*-
"""COMPETITION_SUMMARY.md の図を作る（figures/*.png）。

数字は COMPETITION_SUMMARY.md の表と同じ（手で写したもの）。public と private の散布図だけは、
締切後に Kaggle API で取った提出一覧（figures/data/submissions_20261024.csv）から読む。
このサーバに日本語のフォントが無いので、図の中の文字は英語にした。

  python scripts/make_summary_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

# 色（dataviz の既定パレット、明るい面。系列の色は固定: private = 1、public = 2、Gold58 OOF = 3）
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_PRIV, C_PUB, C_GOLD = "#2a78d6", "#eb6834", "#1baf7a"
C_OTHER = "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def ring(**kw):
    """点の周りに面の色の 2px の輪"""
    return dict(markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, **kw)


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_progression():
    """自前モデル単体の推移（Gold58 OOF / public / private）"""
    runs = [("g2", "MIL +\nslice aux"), ("s1", "stable\nrecipe"), ("s1+TTA", "center crop"), ("s2", "10 epochs"),
            ("s4", "6x6 tokens\n16 epochs"), ("s6", "ConvNeXt-S"), ("s7", "140 mm crop"),
            ("s9 *", "consensus\nlabels"), ("s11 *", "silent-cell\nfill"), ("s13 full *", "site bias\n+ full data x3"),
            ("s17 full *", "rank-loss\nfine-tune")]
    pub = [0.873, 0.908, 0.911, 0.926, 0.941, 0.942, 0.945, 0.948, 0.949, 0.953, 0.951]
    prv = [0.858, 0.896, 0.899, 0.915, 0.928, 0.930, 0.932, 0.934, 0.935, 0.940, 0.939]
    gold = {0: 0.834, 1: 0.886, 3: 0.897, 4: 0.904, 5: 0.903, 6: 0.903, 7: 0.914, 8: 0.919}
    gold5 = {9: 0.919, 10: 0.9226}   # 全データ版は Gold58 で測れないので、同じ設定の 5 fold 版
    x = range(len(runs))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.axvspan(6.5, 10.5, color="#f0efec", lw=0)
    ax.text(8.5, 0.962, "changes from the user's advice (*)", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.plot(x, pub, color=C_PUB, lw=2, marker="o", **ring(markerfacecolor=C_PUB), label="Public LB", solid_capstyle="round")
    ax.plot(x, prv, color=C_PRIV, lw=2, marker="o", **ring(markerfacecolor=C_PRIV), label="Private LB", solid_capstyle="round")
    gx, gy = list(gold), list(gold.values())
    ax.plot(gx, gy, color=C_GOLD, lw=2, marker="o", **ring(markerfacecolor=C_GOLD), label="Gold58 5-fold OOF")
    ax.plot([8] + list(gold5), [gold[8]] + list(gold5.values()), color=C_GOLD, lw=1.2, alpha=0.6)
    ax.plot(list(gold5), list(gold5.values()), ls="none", marker="o", markersize=8, markerfacecolor=SURFACE,
            markeredgecolor=C_GOLD, markeredgewidth=2, label="Gold58 OOF of the 5-fold version")
    for vals, dy in ((pub, 6), (prv, -14)):
        ax.annotate(f"{vals[0]:.3f}", (0, vals[0]), xytext=(-6, dy), textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax.annotate(f"{pub[9]:.3f}", (9, pub[9]), xytext=(0, 9), textcoords="offset points", ha="center", color=INK, fontsize=9, weight="bold")
    ax.annotate(f"{prv[9]:.3f}", (9, prv[9]), xytext=(0, -16), textcoords="offset points", ha="center", color=INK, fontsize=9, weight="bold")
    ax.annotate(f"{gold[0]:.3f}", (0, gold[0]), xytext=(-6, -4), textcoords="offset points", ha="right", color=INK2, fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r}\n{d}" for r, d in runs], fontsize=8.5)
    ax.set_xlim(-0.8, len(runs) - 0.4)
    ax.set_ylim(0.83, 0.97)
    ax.set_ylabel("macro AUC")
    ax.grid(axis="x", visible=False)
    ax.set_title("geo3d single model: score by run (Oct 10 - Oct 22)")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    save(fig, "fig1_geo3d_progression.png")


def fig_private_gain():
    """private の伸びの内訳（g2 → s13 全データ）"""
    steps = [("Stable recipe + center-crop TTA  (g2 -> s1+TTA)", 0.041, False),
             ("10 -> 16 epochs, 6x6 tokens  (-> s4)", 0.029, False),
             ("ConvNeXt-Small, 140 mm crop  (-> s7)", 0.004, False),
             ("Consensus labels + silent-cell fill  (-> s11)  [advice]", 0.003, True),
             ("Site bias + full data x3 seeds  (-> s13 full)  [advice]", 0.005, True)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    y = list(range(len(steps)))[::-1]
    ax.barh(y, [s[1] for s in steps], height=0.42, color=C_PRIV, edgecolor=SURFACE, linewidth=0)
    for yi, (lab, v, adv) in zip(y, steps):
        ax.text(v + 0.0006, yi, f"+{v:.3f}", va="center", color=INK, fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in steps], fontsize=9.5, color=INK2)
    ax.set_xlim(0, 0.047)
    ax.set_xlabel("private LB gain (macro AUC)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the private +0.082 came from (g2 0.858 -> s13 full 0.940)")
    save(fig, "fig2_private_gain_breakdown.png")


def fig_public_private():
    """全提出の public と private"""
    d = pd.read_csv(OUT / "data/submissions_20261024.csv").dropna(subset=["publicScore"])
    d = d[d.publicScore >= 0.85]
    own = {57036633, 57043771, 57046438, 57054962, 57059455, 57065494, 57078985, 57087771, 57114805, 57140602,
           57154760, 57178683, 57226112, 57368467, 57482007, 57526262,
           57079265, 57091712, 57118853, 57157282, 57540133}            # geo3d だけ（単体と自前どうしの平均）
    mix = {57231572, 57232019, 57296480, 57323180, 57366315, 57366376, 57381683, 57473665, 57476015, 57495948,
           57531043, 57540421, 57540735}                                 # geo3d + ほかのモデル
    d["grp"] = ["own" if r in own else "mix" if r in mix else "other" for r in d.ref]
    gap = (d.privateScore - d.publicScore).mean()
    # 小数 3 桁で同じ点になる提出は、横に少しずらして重ならないようにする（丸めの幅 0.001 より小さく）
    d = d.sort_values(["publicScore", "privateScore", "grp"]).copy()
    k = d.groupby(["publicScore", "privateScore"]).cumcount()
    n = d.groupby(["publicScore", "privateScore"]).ref.transform("size")
    d["px"] = d.publicScore + (k - (n - 1) / 2) * 0.00028
    style = {"other": (C_OTHER, "Saki's models / public models / before geo3d"),
             "own": (C_PRIV, "geo3d only (single or geo3d averages)"),
             "mix": (C_GOLD, "geo3d + other models")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, (lo, hi) in zip(axes, [(0.86, 0.962), (0.938, 0.960)]):
        ax.plot([lo, hi], [lo + gap, hi + gap], color=MUTED, lw=1)
        for g in ["other", "own", "mix"]:
            s = d[d.grp == g]
            ax.plot(s.px, s.privateScore, ls="none", marker="o", alpha=0.9, **ring(markerfacecolor=style[g][0]),
                    label=style[g][1])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo + gap - 0.006, hi + gap + 0.006)
        ax.set_xlabel("public LB")
    axes[0].set_ylabel("private LB")
    axes[0].set_title("All submissions (public >= 0.85)")
    axes[0].text(0.958, 0.858, f"gray line: private = public {gap:+.3f}\n(mean over these submissions)", color=INK2, fontsize=9, ha="right")
    axes[1].set_title("Zoom: public >= 0.938")
    lab = {57476015: "s13 full + R6", 57473665: "s13 full + o2\n+ nartaa + R6", 57368467: "s13 full",
           57366376: "s13 full + o2", 57062299: "Saki o2", 57318672: "public 0.958\nblend copy",
           57526262: "s17 full", 57226112: "s11"}
    off = {57476015: (-8, 6, "right"), 57473665: (8, 2, "left"), 57368467: (-8, 4, "right"), 57366376: (-8, 6, "right"),
           57062299: (14, -34, "left"), 57318672: (8, -14, "left"), 57526262: (8, -8, "left"), 57226112: (-8, 4, "right")}
    for r, t in lab.items():
        row = d[d.ref == r].iloc[0]
        dx, dy, ha = off[r]
        axes[1].annotate(t, (row.px, row.privateScore), xytext=(dx, dy), textcoords="offset points",
                         ha=ha, color=INK, fontsize=8.5,
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=5))
    axes[1].legend(loc="upper left", fontsize=8.5)
    save(fig, "fig3_public_vs_private.png")


def fig_ensemble_delta():
    """s13 全データ単体からの差（public と private）"""
    rows = [("+ Saki o2 (prob. mean)", 0.001, 0.002),
            ("+ o2 + nartaa (all 12, rank 1/2)", 0.002, 0.000),
            ("+ o2 + nartaa + R6 (rank 1/4)", 0.005, 0.004),
            ("+ R6 (rank 1/2)", 0.004, 0.005),
            ("Head mix tuned on Gold58", -0.001, -0.001),
            ("s17: rank-loss fine-tune", -0.002, -0.001),
            ("s17 + o2 + nartaa (rank 1/3)", 0.003, 0.001)]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    n = len(rows)
    y = list(range(n))[::-1]
    h = 0.34
    ax.barh([v + h / 2 + 0.02 for v in y], [r[1] for r in rows], height=h, color=C_PUB, label="Public LB", linewidth=0)
    ax.barh([v - h / 2 - 0.02 for v in y], [r[2] for r in rows], height=h, color=C_PRIV, label="Private LB", linewidth=0)
    for v, (_, p, q) in zip(y, rows):
        for val, yy in ((p, v + h / 2 + 0.02), (q, v - h / 2 - 0.02)):
            ax.text(val + (0.00012 if val >= 0 else -0.00012), yy, f"{val:+.3f}", va="center",
                    ha="left" if val >= 0 else "right", color=INK2, fontsize=8.5)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], color=INK2)
    ax.set_xlim(-0.0035, 0.0065)
    ax.set_xlabel("difference from s13 full-data x3 alone (public 0.953 / private 0.940)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("What each addition did on public vs private")
    save(fig, "fig4_ensemble_delta.png")


def fig_per_finding():
    """s13 の所見ごとの Gold58 AUC（5 fold OOF）"""
    t = {"ACL": 0.962, "MCL": 0.973, "Medial meniscus": 0.951, "Lateral meniscus": 0.877, "Medial OA": 0.992,
         "Lateral OA": 0.841, "PF OA": 0.884, "Effusion": 0.984, "Synovitis": 0.753, "Baker's cyst": 0.962,
         "Contusion": 0.935, "Fracture": 0.914}
    items = sorted(t.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = range(len(items))
    ax.barh(list(y), [v - 0.70 for _, v in items], left=0.70, height=0.5,
            color=[C_PRIV if k == "Synovitis" else C_OTHER for k, _ in items], linewidth=0)
    for yi, (k, v) in zip(y, items):
        ax.text(v + 0.003, yi, f"{v:.3f}", va="center", color=INK if k == "Synovitis" else INK2, fontsize=9)
    ax.axvline(0.919, color=MUTED, lw=1)
    ax.text(0.922, 0, "macro 0.919", color=INK2, fontsize=9, va="center")
    ax.set_yticks(list(y))
    ax.set_yticklabels([k for k, _ in items], color=INK2)
    ax.set_xlim(0.70, 1.01)
    ax.set_xlabel("Gold58 AUC (s13, 5-fold OOF)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Per-finding Gold58 AUC: synovitis stayed the weak spot")
    save(fig, "fig5_per_finding_gold58.png")


def fig_architecture():
    """geo3d（s13）の構成"""
    fig, ax = plt.subplots(figsize=(12, 4.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    def box(x, y, w, h, title, body, main=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#eef4fc" if main else "#f3f2ee", edgecolor=C_PRIV if main else BASE, lw=1.2))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.58, body, ha="center", va="top", fontsize=8.3, color=INK2, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=MUTED, lw=1.2))

    y0, h = 1.75, 2.2
    box(0.1, y0, 2.15, h, "Input", "up to 7 series per study\n(3 planes x fat-sat or not)\n140 mm center -> 384 px\n[prev, mid, next] = 3 ch\nright knee mirrored\n4 slices / series (train)")
    box(2.6, y0, 1.9, h, "Encoder", "ConvNeXt-Small\n(IN-12k -> 1k, 384)\nper slice:\n12x12 feature map\n-> 6x6 cells (~22 mm)")
    box(4.85, y0, 2.25, h, "Tokens", "each cell: 256-d\n+ patient coords (mm,\n   Fourier features)\n+ series-type embedding\n~1,000 tokens / study")
    box(7.45, y0, 2.25, h, "Study head", "per-finding gated\nattention MIL over\nall tokens of all series\n-> 12 logits")
    box(10.05, y0, 1.85, h, "Output", "mean logits over\nall 4-slice groups\nx 3 seeds (full data)\n-> sigmoid")
    for xa, xb in ((2.25, 2.6), (4.5, 4.85), (7.1, 7.45), (9.7, 10.05)):
        arrow(xa, y0 + h / 2, xb, y0 + h / 2)
    box(0.1, 0.1, 3.6, 1.25, "Slice aux head (train only)", "mean feature per slice -> 12 logits,\nBCE with the study label\n(without it the encoder barely learns)", main=False)
    box(4.0, 0.1, 4.3, 1.25, "Labels & loss (train only)", "soft targets: Codex + GPT consensus,\nsilent cells refilled (PF OA, contusion, fracture)\nGold58 expert labels x5, site x finding bias b", main=False)
    box(8.6, 0.1, 3.3, 1.25, "s17 only", "+ pairwise rank loss 0.3\n(4-epoch fine-tune from s13)", main=False)
    arrow(3.4, y0, 3.4, 1.35)
    ax.set_title("geo3d (s13) architecture")
    save(fig, "fig0_geo3d_architecture.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_architecture()
    fig_progression()
    fig_private_gain()
    fig_public_private()
    fig_ensemble_delta()
    fig_per_finding()
