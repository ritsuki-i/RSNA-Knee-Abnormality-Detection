#!/bin/bash
# public 0.953（57368467）の重みを作り直す: s13 の設定を全データ（--fold -1、Gold58 も含む 4,407 件）・seed 0〜2 で学習する。
# 1 本 約 3.3 時間（RTX 3090、16 エポック）、3 本で約 10 時間。
# 既に入っている 0.953 の重み（outputs/geo3d/s13full_s*）を消さないよう、名前の頭を引数で変えられる（既定 s13full_re）。
#   bash run_train_full.sh            → outputs/geo3d/s13full_re_s{0,1,2}/fold-1.pt
#   bash run_train_full.sh myname     → outputs/geo3d/myname_s{0,1,2}/fold-1.pt
set -eu
cd "$(dirname "$0")"
source /mnt/home/ishikawa/kaggle/env.sh
NAME=${1:-s13full_re}
ARGS="--head mil --inst_w 1.0 --per_series 4 --bs 4 --accum 2 --lr 5e-5 --warm_steps 300 --val_every 4 --pool 2 --epochs 16 \
  --backbone convnext_small.in12k_ft_in1k_384 --crop_mm 140 \
  --label_csv outputs/geo3d/labels_silentfill.csv --site_csv outputs/geo3d/sites.csv --site_sd 0.5"
RUNS=""
for S in 0 1 2; do
  python scripts/train_geo3d.py --name ${NAME}_s$S --fold -1 --seed $S $ARGS > outputs/geo3d/train_${NAME}_s$S.log 2>&1
  RUNS="$RUNS${RUNS:+,}${NAME}_s$S"
done
# 壊れていないかの確認（最後のエポックの損失が 5 fold 版の平均 + 0.05 以下、seed どうしで重みが違う）
python scripts/geo3d_full_check.py s13 $RUNS
