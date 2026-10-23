#!/bin/bash
# s17（2026-10-22）: s13 の重みから、検査ヘッドに所見ごとの対の順位損失（0.3）を足して 4 エポック追加で学習する。
# 元の段取りは scripts/geo3d_s17.sh（プロジェクト側）。ここでは対照（順位損失なし）を回さない。
#   bash run_train_s17.sh 5fold [名前]   → s13 の各 fold（outputs/geo3d/s13/fold{F}.pt）から。Gold58 OOF 0.9226（s17rank）の元
#   bash run_train_s17.sh full  [名前]   → s13 全データ版（outputs/geo3d/s13full_s{S}/fold-1.pt）から seed 0〜2。public 0.951 の重み（s17full）の元
# 1 本 約 50 分〜1 時間（RTX 3090 で 1 本だけ動かしたとき）。施設のずれ b も元の重みの隣の *_site_b.npy から引き継ぐ。
# 入っている重み（s17rank、s17full_s*）を消さないよう、名前の頭は既定で *_re にする。
set -eu
cd "$(dirname "$0")"
source /mnt/home/ishikawa/kaggle/env.sh
MODE=${1:?5fold か full を指定する}
BASE="--head mil --inst_w 1.0 --per_series 4 --bs 4 --accum 2 --pool 2 --backbone convnext_small.in12k_ft_in1k_384 --crop_mm 140 \
  --label_csv outputs/geo3d/labels_silentfill.csv --site_csv outputs/geo3d/sites.csv --site_sd 0.5"
FT="--epochs 4 --warm_steps 0 --lr 2e-5 --lr_head 1e-4 --val_every 99 --rank_w 0.3 --rank_margin 0.5 --rank_queue 256"
case $MODE in
  5fold)
    NAME=${2:-s17rank_re}
    for F in 0 1 2 3 4; do
      python scripts/train_geo3d.py --name $NAME --fold $F $BASE $FT --init_ckpt outputs/geo3d/s13/fold$F.pt \
        > outputs/geo3d/train_${NAME}_f$F.log 2>&1
    done
    python scripts/geo3d_gold_oof.py s13 s17rank $NAME
    ;;
  full)
    NAME=${2:-s17full_re}
    RUNS=""
    for S in 0 1 2; do
      python scripts/train_geo3d.py --name ${NAME}_s$S --fold -1 --seed $S $BASE $FT --init_ckpt outputs/geo3d/s13full_s$S/fold-1.pt \
        > outputs/geo3d/train_${NAME}_s$S.log 2>&1
      RUNS="$RUNS${RUNS:+,}${NAME}_s$S"
    done
    # 壊れていないかの確認（最後のエポックの損失が 5 fold 版 s17rank の平均 + 0.05 以下、seed どうしで重みが違う）
    python scripts/geo3d_full_check.py s17rank $RUNS
    ;;
  *) echo "5fold か full を指定する"; exit 1 ;;
esac
