# -*- coding: utf-8 -*-
"""Gold58（専門家のラベルが付いた 58 件）を、大会の train.csv から取り出す。

  大会の train.csv は、58 件だけ 12 所見の列に値が入っている（残りは空）。その行を StudyInstanceUID・Report・12 所見のまま書き出す。
  大会のデータそのものなので、このファイルは配らない（.gitignore 済み）。

  python scripts/make_gold58.py [--train data/raw/train.csv] [--out data/interim/labeled_studies.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("--train", default=str(ROOT / "data/raw/train.csv"))
ap.add_argument("--out", default=str(ROOT / "data/interim/labeled_studies.csv"))
a = ap.parse_args()
t = pd.read_csv(a.train)
g = t[t[T].notna().any(axis=1)][["StudyInstanceUID", "Report"] + T]
assert len(g) == 58, len(g)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
g.to_csv(a.out, index=False)
print(a.out, len(g))
