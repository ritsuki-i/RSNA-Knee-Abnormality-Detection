# s13 全データ版（public LB 0.953）一式

自前モデル geo3d の s13 設定を、全データ（4,407 件、Gold58 を含む）で seed 0〜2 の 3 本学習し、3 本のロジットを平均したもの。
提出 `57368467`（2026-10-18、"geo3d s13 full-data x3 seeds (no CV, crop0.93)"）で **public 0.953**。自前モデルの単体で最高。
同じ設定の 5 fold 版は Gold58 OOF 0.919（単体では未提出。5 fold 版 + 崎さんの o2 で public 0.953）。

このフォルダの中だけで、学習・手元での推論・提出ができる（画像キャッシュはリンク、Python 環境は `/mnt/home/ishikawa/kaggle/env.sh`）。
2026-10-21 に、ここから次の 2 つを確かめた。
- 手元の画像キャッシュで見える test 3 件を推論すると、Kaggle の 0.953 の提出の出力（`kaggle_submitted/submission_visible_test.csv`）と差 4e-5 以内。
- `scripts/train_geo3d.py` が、このフォルダのファイルだけで動く（16 件・1 エポックの動作確認）。

## 中身

| 場所 | 中身 |
|---|---|
| `src/geo3d/model.py` | モデル GeoQueryNet（ConvNeXt-Small → 6×6 のマス → 患者座標とシリーズの種類の埋め込み → 所見ごとの注意 MIL） |
| `src/geo3d/data.py` | 前処理（右膝の左右反転、[前, 中, 後] の 3 チャンネル、切り出し）と推論（4 枚の組を全部作って平均） |
| `src/geo3d/geometry.py` | シリーズの選び方（6 種類から 1 本ずつ → 枚数の多い順に 7 本まで）、4 枚の組の作り方、DICOM 座標の計算 |
| `scripts/train_geo3d.py` | 学習の本体（データの読み込み・拡張・損失・施設のずれ・EMA・保存） |
| `run_train_full.sh` | 0.953 の重みを作り直す（全データ × seed 0〜2、約 10 時間） |
| `run_train_5fold.sh` | 同じ設定の 5 fold 版（ローカル評価用、約 17 時間）と Gold58 OOF の集計 |
| `run_submit.sh` | 0.953 と同じ形で提出する（Kaggle Models に上げ → notebook を更新 → 見える test で確認 → 提出） |
| `outputs/geo3d/s13full_s{0,1,2}/fold-1.pt` | **0.953 の重み 3 本**（各 96MB。提出で上げたものと sha256 一致）。`fold-1.log` は学習のログ |
| `outputs/geo3d/s13/` | 5 fold 版のログと OOF（`oof_fold*.csv`、Gold58 OOF 0.919 の元） |
| `outputs/geo3d/labels_silentfill.csv` | 学習の目標と重み（下の「ラベル」） |
| `outputs/geo3d/labels_consensus.csv` | その元（複数の所見文ラベルの合意） |
| `outputs/geo3d/sites.csv` | 施設（所見文の書式で分けた 24 クラスタ） |
| `outputs/working/codex_report_labels/` | Codex の所見文ラベル（Gold 以外 4,349 件と Gold58） |
| `data/interim/geo3d_index_{train,test}.pkl` | 検査・シリーズ・スライスの一覧と DICOM の座標（`scripts/geo3d_build_index.py` で作る） |
| `data/interim/labeled_studies.csv` | Gold58（専門家のラベル） |
| `data/interim/folds_jackee.csv` | fold（崎さんの `labels_fused.csv` の fold 列） |
| `data/external/llm_labels/report_labels_v2.csv` | GPT v2 の所見文ラベル（記載あり / なしの判定に使う） |
| `data/processed/img_{train,test}_c140_384` | 画像キャッシュ（原画の中央 140mm を 384px に）。**本体へのリンク**（約 41GB。作り直しは `scripts/fetch_dicom_cache.py --crop-mm 140`） |
| `kaggle_submitted/` | 0.953 の提出で実際に使った notebook のコード `geo3d_infer.py`、`kernel-metadata.json`、`geo3d_manifest.json`、見える test の出力 |
| `scripts/geo3d_kernel_template.py` | 提出 notebook の雛形（今の版。o2・nartaa・R6 と混ぜる機能も入っているが、指定しなければ geo3d だけ） |
| `scripts/geo3d_submit.py` | 提出スクリプト（重みを Kaggle Models に上げ、notebook を作って提出） |
| `scripts/geo3d_predict_test_local.py` | 手元の画像キャッシュで見える test を推論（Kaggle の出力との照合用） |
| `scripts/geo3d_full_check.py` | 全データ学習が壊れていないかの確認 |
| `scripts/geo3d_gold_oof.py` | Gold58 OOF の集計 |
| `scripts/geo3d_make_consensus_labels.py`、`geo3d_make_silent_fill_labels.py`、`geo3d_site_bias.py` | ラベルと施設の CSV を作ったスクリプト（参考。作り直すには、このフォルダに無い元データ（所見文、ほかのラベル、s9 の OOF）が要る） |
| `requirements-lock.txt` | 固定したライブラリの版（torch 2.14.1+cu126、timm 1.0.30 など） |

## 学習の設定

```
python scripts/train_geo3d.py --name s13full_s0 --fold -1 --seed 0 \
  --head mil --inst_w 1.0 --per_series 4 --bs 4 --accum 2 --lr 5e-5 --warm_steps 300 \
  --val_every 4 --pool 2 --epochs 16 --backbone convnext_small.in12k_ft_in1k_384 --crop_mm 140 \
  --label_csv outputs/geo3d/labels_silentfill.csv --site_csv outputs/geo3d/sites.csv --site_sd 0.5
```
（seed 1・2 も同じ。`--fold -1` = 検証に分けず全件で学習）

### 画像の使い方
- **シリーズ**: 1 検査から最大 7 本。種類（断面 3 × 脂肪抑制の有無）ごとに枚数の多い 1 本、残りの枠は枚数の多い順。学習中は 4 本以上なら 25% の確率で 1 本抜く。
- **空間の窓**: 原画の中央 140mm 四方を 384px に。学習では面積 75〜100% をランダムに切り出し、推論では中央 93%。
- **スライス方向の窓**: 1 枚 = [前, 中, 後] の 3 枚を 3 チャンネルに。
- **濃度の窓**: スライスごとに画素の 0.5%〜99.5% を 0〜255 に（DICOM の WW / WL は使わない）。学習ではシリーズ単位でガンマ 0.8〜1.25・明るさ・コントラストを揺らす。
- **向き**: 右膝は内側・外側が左膝と同じ向きになるよう左右反転（矢状断は反転しない）。
- **枚数**: 学習ではシリーズごとに 4 枚（等間隔、開始位置はランダム）→ 1 回の入力は約 28 枚。推論では 4 枚の組を全部作って（30 枚なら 8 組）組ごとの予測を平均。
- **モデル**: ConvNeXt-Small（ImageNet-12k → 1k、384）で 1 枚 12×12 の特徴 → 6×6 = 36 マス（1 マス約 22mm）→ 256 次元。各マスに患者座標（mm、Fourier 特徴）とシリーズの種類の埋め込みを足し、全マス（約 1,000 個）から所見ごとのゲート付き注意で 12 個のロジットにまとめる。学習中だけ、スライスごとの補助ヘッドにも検査のラベルで損失をかける。

### ラベル
- 所見文から作ったラベル。記載ありの欄は Codex と GPT 系 3 種の合意（目標と重み）、PF OA・挫傷・骨折の記載なしの欄は s9 の OOF などで埋め直したもの（`labels_silentfill.csv`）。
- Gold58 は専門家のラベルを重み 5 で使う。
- 施設（所見文の書式のクラスタ 24 個）× 所見のずれ b を、Gold 以外の損失に足して学習中に吸収する（事前分布の sd 0.5。推論では使わない）。
- 16 エポック固定で、最後のエポックの EMA（0.998）の重みを使う（検証の成績でエポックを選ばない）。

## 推論と提出
- notebook: Kaggle Models の重みと `src/geo3d` を読み、test の DICOM を直接読んで推論（`kaggle_submitted/geo3d_infer.py`）。3 本のロジットを平均して sigmoid。T4 1 枚で隠しテスト約 2 時間。
- 手元で見える test を推論する:
  ```
  for S in 0 1 2; do python scripts/geo3d_predict_test_local.py --run s13full_s$S --folds -1 --crops 0.93; done
  ```
  （出力 `outputs/geo3d/s13full_s*/test_local_-1.csv`。3 本のロジットの平均が Kaggle の出力と一致する）
- 提出: `bash run_submit.sh <新しい instance 名>`（Kaggle への送信なので、必要なときだけ）。
