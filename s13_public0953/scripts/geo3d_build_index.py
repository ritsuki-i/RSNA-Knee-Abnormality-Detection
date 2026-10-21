# -*- coding: utf-8 -*-
"""dicom_meta_*.csv から、検査ごとの幾何の索引を作る（geo3d 用）。

出力: data/interim/geo3d_index_<split>.pkl
  {study_uid: dict(study_idx, center(3), side(±1), series=[dict(series_idx, type, n,
      slice_idx(n,) 位置順, p0(n,3), r(n,3), c(n,3), ps(n,2), rows(n,), cols(n,))])}
PNG 取得の途中でも動く（meta に載っている検査だけを索引に入れる）。
"""
from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from geo3d.geometry import build_series, parse_vec, plane_from_normal, series_type, study_frame  # noqa: E402

COLS = ["study_idx", "series_idx", "slice_idx", "StudyInstanceUID", "SeriesInstanceUID",
        "Rows", "Columns", "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient", "Laterality"]


def build(split):
    files = sorted(glob.glob(str(ROOT / f"data/interim/dicom_meta_{split}*.csv")))
    m = pd.concat([pd.read_csv(f, usecols=COLS, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str,
                                                       "Laterality": str}) for f in files])
    m = m.dropna(subset=["PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient"])
    ser = pd.read_csv(ROOT / f"data/raw/{split}_series.csv", dtype=str).set_index("SeriesInstanceUID")
    out, bad = {}, 0
    for uid, g in m.groupby("StudyInstanceUID", sort=False):
        series, centers = [], []
        for sidx, s in g.groupby("series_idx"):
            p0 = np.stack(s.ImagePositionPatient.map(lambda v: parse_vec(v, 3)).values)
            o = np.stack(s.ImageOrientationPatient.map(lambda v: parse_vec(v, 6)).values)
            ps = np.stack(s.PixelSpacing.map(lambda v: parse_vec(v, 2)).values)
            rows, cols = s.Rows.values.astype(np.float64), s.Columns.values.astype(np.float64)
            info = build_series(p0, o[:, :3], o[:, 3:], ps, rows, cols)
            suid = s.SeriesInstanceUID.iloc[0]
            if suid in ser.index:
                plane, fluid = ser.at[suid, "Anatomical_Plane"], int(ser.at[suid, "Fluid_Sensitive"])
            else:
                plane, fluid = plane_from_normal(info["normal"]), 0
            t = series_type(plane, fluid)
            if t is None:
                t = series_type(plane_from_normal(info["normal"]), fluid)
            k = info["order"]
            series.append(dict(series_idx=int(sidx), type=int(t), n=len(k), plane=plane,
                               slice_idx=s.slice_idx.values[k].astype(np.int32),
                               p0=p0[k], r=o[k, :3], c=o[k, 3:], ps=ps[k], rows=rows[k], cols=cols[k]))
            centers.append(info["center"])
        lat = g.Laterality.dropna()
        lat = lat.iloc[0] if len(lat) else None
        cen, side = study_frame(centers, lat)
        tag = str(lat).upper() if lat is not None else ""
        out[uid] = dict(study_idx=int(g.study_idx.iloc[0]), center=cen, side=side, lat_tag=tag,
                        series=series)
    dst = ROOT / f"data/interim/geo3d_index_{split}.pkl"
    with open(dst, "wb") as f:
        pickle.dump(out, f)
    # 左右の決め方の確認: タグがある検査で、中心 x の符号とタグが一致する割合
    tagged = [(v["lat_tag"], v["center"][0]) for v in out.values() if v["lat_tag"] in ("L", "R")]
    agree = np.mean([(t == "L") == (x >= 0) for t, x in tagged]) if tagged else float("nan")
    print(f"{split}: studies={len(out)} series={sum(len(v['series']) for v in out.values())} "
          f"tagged={len(tagged)} sign_agree={agree:.3f} -> {dst}")


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["train", "test"]):
        build(sp)
