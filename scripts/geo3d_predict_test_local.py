# -*- coding: utf-8 -*-
"""手元の PNG キャッシュでテスト 3 件を推論する（Kaggle の DICOM 直読みの出力と照合するため）。

  python scripts/geo3d_predict_test_local.py --run p0 --folds 0 [--compare path/to/submission.csv]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train_geo3d as tg  # noqa: E402
from geo3d.data import infer_study  # noqa: E402
from geo3d.model import GeoQueryNet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--folds", default="0")
ap.add_argument("--compare", default="")
ap.add_argument("--mix", type=float, default=0.0)
ap.add_argument("--crops", type=float, nargs="+", default=[1.0])
a = ap.parse_args()
index = pickle.load(open(ROOT / "data/interim/geo3d_index_test.pkl", "rb"))
dev = torch.device("cuda")
models, cfg = [], None
for f in a.folds.split(","):
    ck = torch.load(ROOT / f"outputs/geo3d/{a.run}/fold{f}.pt", map_location="cpu", weights_only=False)
    cfg = ck["args"]
    m = GeoQueryNet(backbone=cfg.get("backbone", "convnext_tiny.in12k_ft_in1k_384"), pretrained=False,
                    use_coords=not cfg.get("no_coords", False), head=cfg.get("head", "dec"),
                    gap=cfg.get("gap", False), pool=cfg.get("pool", 1), xattn=cfg.get("xattn", 0))
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    models.append(m.to(dev).eval())
tg.use_cache(index, "test", cfg.get("crop_mm", 0.0))
rows = []
for uid, st in index.items():
    ser = tg.series_list_png(st, cfg["max_series"])
    outs = [infer_study(m, ser, st["center"], st["side"], cfg["size"], cfg["per_series"], cfg["max_series"], dev,
                        crops=tuple(a.crops)) for m in models]
    lg = np.mean([(1 - a.mix) * o["study"] + a.mix * o["inst"] for o in outs], 0)
    rows.append([uid] + list(1 / (1 + np.exp(-lg))))
out = pd.DataFrame(rows, columns=["StudyInstanceUID"] + tg.T).set_index("StudyInstanceUID")
out.to_csv(ROOT / f"outputs/geo3d/{a.run}/test_local_{a.folds.replace(',', '')}.csv")
print(out.round(3))
if a.compare:
    k = pd.read_csv(a.compare).set_index("StudyInstanceUID").loc[out.index, tg.T]
    print("max |kaggle - local| =", float((k - out).abs().max().max()))
