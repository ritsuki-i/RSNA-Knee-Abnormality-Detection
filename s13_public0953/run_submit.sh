#!/bin/bash
# public 0.953（57368467）と同じ形で提出する: 全データ版 3 本のロジットの平均、中央 93% の切り出し、検査ヘッドのみ（mix 0）。
# 重みと src を Kaggle Models（ritsuki0720/rsna-knee-geo3d、instance は毎回新しい名前）に上げ、notebook ritsuki0720/rsna-knee-geo3d-infer
# を更新して見える test で完走を確かめてから提出する。Kaggle への提出（外部への送信）なので、実行は必要なときだけ。
#   bash run_submit.sh <instance 名> [run の頭（既定 s13full）]
set -eu
cd "$(dirname "$0")"
source /mnt/home/ishikawa/kaggle/env.sh
INST=${1:?instance 名（毎回新しい名前）を指定する}
NAME=${2:-s13full}
HOME=/mnt/home/ishikawa/kaggle python scripts/geo3d_submit.py --run ${NAME}_s0,${NAME}_s1,${NAME}_s2 --folds -1 --instance $INST \
  --mix 0 --crops 0.93 --message "geo3d s13 full-data x3 seeds (no CV, crop0.93) [$INST]"
