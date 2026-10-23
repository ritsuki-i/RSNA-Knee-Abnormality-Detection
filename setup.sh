#!/bin/bash
# 配る一式（コードと自前のラベル、重み）に、大会のデータ・画像キャッシュ・スライスの一覧をそろえる。
# 大会のデータと大きいファイル（スライスの一覧 92MB、画像キャッシュ 約 41GB）はこのフォルダに入れていない（.gitignore 済み）。
# 重み（16 本、約 1.5GB）は実ファイルで入っているが、git では無視する（.gitignore の *.pt）。
#
#   bash setup.sh <大会のデータがあるプロジェクトのパス> [--download-weights]
#     例: bash setup.sh /mnt/home/ishikawa/kaggle/RSNA_Knee_Abnormality_Detection
#
# そのプロジェクトから次をリンクする（無いものは作り方を表示して止まる）:
#   data/raw/*.csv（大会の CSV）、data/processed/img_{train,test}_c140_384（画像キャッシュ）、
#   data/interim/dicom_meta_*.csv（DICOM のヘッダ。スライスの一覧をここで作り直す）、重み（WEIGHTS_SHA256.txt にあるもので、無いものだけ）
# 全データ版の重みが無ければ --download-weights で Kaggle Models（ritsuki0720/rsna-knee-geo3d/pytorch/{s13full-3s,s17full-3s}/1、非公開）から取る。
set -eu
cd "$(dirname "$0")"
SRC=$(cd "${1:?大会のデータがあるプロジェクトのパスを指定する}" && pwd)
DL=${2:-}
source /mnt/home/ishikawa/kaggle/env.sh
mkdir -p data/raw data/interim data/processed

link() {   # link <元> <先>: 先が無ければリンクを張る。元が無ければ止まる
  if [ ! -e "$2" ]; then
    [ -e "$1" ] || { echo "見つからない: $1"; echo "  $3"; exit 1; }
    ln -s "$1" "$2"
  fi
}

# 1) 大会の CSV
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  link "$SRC/data/raw/$f" data/raw/$f "大会のページ（rsna-knee-abnormality-detection）から取る"
done

# 2) Gold58（大会の train.csv のうち、ラベルの付いた 58 行）
[ -e data/interim/labeled_studies.csv ] || python scripts/make_gold58.py

# 3) 画像キャッシュ（原画の中央 140mm を 384px に）
for s in train test; do
  link "$SRC/data/processed/img_${s}_c140_384" data/processed/img_${s}_c140_384 \
    "python scripts/fetch_dicom_cache.py --crop-mm 140 で作る（train は 4〜5 時間）"
done

# 4) スライスの一覧（DICOM のヘッダから作り直す。元のプロジェクトのものと同じになることを sha256 で確かめた: README）
for f in "$SRC"/data/interim/dicom_meta_train_shard?of4.csv "$SRC"/data/interim/dicom_meta_test.csv; do
  link "$f" data/interim/$(basename "$f") "python scripts/fetch_dicom_cache.py で画像と一緒に作られる"
done
for s in train test; do
  [ -e data/interim/geo3d_index_$s.pkl ] || python scripts/geo3d_build_index.py $s
done

# 5) 重み（s13・s17 の 5 fold 版と全データ版、計 16 本。2026-10-24 からこのフォルダに実ファイルで入っている。
#    git では *.pt を無視しているので、clone し直したときだけここで戻す）
for P in $(awk '{print $2}' WEIGHTS_SHA256.txt); do
  mkdir -p "$(dirname "$P")"
  if [ ! -e "$P" ] && [ -f "$SRC/$P" ]; then ln -s "$SRC/$P" "$P"; fi
done
# 全データ版はプロジェクトに無くても Kaggle Models（非公開）から取れる。5 fold 版は Kaggle に上げていない
for I in s13full-3s:s13full s17full-3s:s17full; do
  INST=${I%%:*}; R=${I##*:}
  [ -e outputs/geo3d/${R}_s0/fold-1.pt ] && [ -e outputs/geo3d/${R}_s1/fold-1.pt ] && [ -e outputs/geo3d/${R}_s2/fold-1.pt ] && continue
  if [ "$DL" = "--download-weights" ]; then
    D=outputs/working/_weights_dl
    mkdir -p $D
    HOME=/mnt/home/ishikawa/kaggle /mnt/home/ishikawa/kaggle/tools/bin/kg models instances versions download \
      ritsuki0720/rsna-knee-geo3d/pytorch/$INST/1 -p $D --untar
    for S in 0 1 2; do mv $D/${R}_s${S}_fold-1.pt outputs/geo3d/${R}_s$S/fold-1.pt; done
    rm -rf $D
  else
    echo "${R} の重みが無い。--download-weights を付けて Kaggle Models から取るか、作り直す（run_train_full.sh / run_train_s17.sh full）"; exit 1
  fi
done
MISS=$(awk '{print $2}' WEIGHTS_SHA256.txt | while read P; do [ -e "$P" ] || echo "$P"; done)
[ -z "$MISS" ] || { echo "無い重み（5 fold 版。run_train_5fold.sh / run_train_s17.sh 5fold で作り直せる）:"; echo "$MISS"; }
sha256sum -c --ignore-missing WEIGHTS_SHA256.txt
echo "そろった"
