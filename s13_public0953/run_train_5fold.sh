#!/bin/bash
# s13 の 5 fold 版（ローカル評価用。Gold58 OOF 0.919）。fold は崎さんの labels_fused.csv の fold 列（data/interim/folds_jackee.csv）。
# 検証 fold の検査は、疑似ラベルごと学習から外す。1 fold 約 3.3 時間、5 fold で約 17 時間。
#   bash run_train_5fold.sh            → outputs/geo3d/s13_re/fold{0..4}.pt と oof_fold*.csv
set -eu
cd "$(dirname "$0")"
source /mnt/home/ishikawa/kaggle/env.sh
NAME=${1:-s13_re}
ARGS="--head mil --inst_w 1.0 --per_series 4 --bs 4 --accum 2 --lr 5e-5 --warm_steps 300 --val_every 4 --pool 2 --epochs 16 \
  --backbone convnext_small.in12k_ft_in1k_384 --crop_mm 140 \
  --label_csv outputs/geo3d/labels_silentfill.csv --site_csv outputs/geo3d/sites.csv --site_sd 0.5"
for F in 0 1 2 3 4; do
  python scripts/train_geo3d.py --name $NAME --fold $F $ARGS > outputs/geo3d/train_${NAME}_f$F.log 2>&1
done
# Gold58 の OOF（全 fold を結合してから macro AUC）。元の s13 と並べる
python scripts/geo3d_gold_oof.py s13 $NAME
