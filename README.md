# geo3d s13 / s17 一式（s13 全データ版 public 0.953 / private 0.940）

自前モデル geo3d の最終形 s13 と、その追加学習版 s17 の重み・学習コード・提出 notebook。
**コンペ全体の経緯と知見は [`COMPETITION_SUMMARY.md`](COMPETITION_SUMMARY.md)** にまとめた。

- s13: 全データ（4,407 件、Gold58 を含む）で seed 0〜2 の 3 本を学習し、3 本のロジットを平均したもの。提出 `57368467`（2026-10-18）で **public 0.953 / private 0.940**。自前モデルの単体で最高。同じ設定の 5 fold 版は Gold58 OOF 0.919。
- s17: s13 の重みから順位損失を足して 4 エポック追加で学習したもの。全データ版の単体 `57526262` は public 0.951 / private 0.939。5 fold 版は Gold58 OOF 0.9226。詳しくは [`outputs/geo3d/S17_README.md`](outputs/geo3d/S17_README.md)。

画像データ（PNG キャッシュ）と大会の CSV は入れていない。使う前に `setup.sh` でそろえる。

## 重み（16 本、約 1.5GB。実ファイル）

| run | 場所 | 中身 | 評価 |
|---|---|---|---|
| s13（5 fold 版） | `outputs/geo3d/s13/fold{0..4}.pt` | fold F を抜いて学習（fold は崎さんの `labels_fused.csv` の fold 列） | Gold58 OOF 0.9190。5 fold 版 + o2 は public 0.953 / private 0.940 |
| s13（全データ版） | `outputs/geo3d/s13full_s{0,1,2}/fold-1.pt` | 全件・seed 0〜2 | public 0.953 / private 0.940 |
| s17（5 fold 版） | `outputs/geo3d/s17rank/fold{0..4}.pt` | `s13/fold{F}.pt` から 4 エポック追加 | Gold58 OOF 0.9226 |
| s17（全データ版） | `outputs/geo3d/s17full_s{0,1,2}/fold-1.pt` | `s13full_s{S}/fold-1.pt` から 4 エポック追加 | public 0.951 / private 0.939 |

- sha256 は `WEIGHTS_SHA256.txt`（`sha256sum -c WEIGHTS_SHA256.txt`）。2026-10-24 に、プロジェクト側の元の重みと、提出に使ったパッケージ（`outputs/working/geo3d_model_*`）の重みと一致することを確かめた。
- 各 run の隣に、学習ログ `*.log`、施設のずれ `*_site_b.npy`、チェックポイントから取り出した学習の設定 `*training_args.json`、OOF（5 fold 版のみ。`oof_fold*` = 検査ヘッド、`oof_inst_*` = スライスヘッド、`oof_mix_*` = 半々）がある。
- `.gitignore` で `*.pt` を無視しているので、git に置くときは重みは入らない。clone し直したら `setup.sh` で戻す（プロジェクトからリンク、または全データ版だけ Kaggle Models から取る）。

## 入れていないもの

| もの | 大きさ | そろえ方（`bash setup.sh <大会のデータがあるプロジェクトのパス>`） |
|---|---|---|
| スライスの一覧 `data/interim/geo3d_index_{train,test}.pkl` | 92MB | DICOM のヘッダ（`dicom_meta_*.csv`）から `scripts/geo3d_build_index.py` で作り直す |
| Gold58 `data/interim/labeled_studies.csv` | 88KB | 大会の `train.csv` のうちラベルの付いた 58 行（所見文込み）。`scripts/make_gold58.py` で取り出す |
| 大会の CSV `data/raw/` | — | 大会のページから。プロジェクトにあればリンク |
| 画像キャッシュ `data/processed/img_{train,test}_c140_384` | 約 41GB | プロジェクトにあればリンク。無ければ `scripts/fetch_dicom_cache.py --crop-mm 140` |

Python 環境は `/mnt/home/ishikawa/kaggle/env.sh`。2026-10-21 に、大きいファイルを外す前の一式で次の 2 つを確かめた。
- 手元の画像キャッシュで見える test 3 件を推論すると、Kaggle の 0.953 の提出の出力（`kaggle_submitted/s13full-3s/submission_visible_test.csv`）と差 4e-5 以内。
- `scripts/train_geo3d.py` がこのフォルダのファイルだけで動く（学習 40 件・1 エポックで、予備学習 → 本学習 → OOF 推論 → 保存まで通る）。学習の件数は 32 以上にする（予備学習の 1 バッチが 32 件で、端数を捨てるため）。
- 2026-10-24 に `scripts/train_geo3d.py` を順位損失（`--rank_w`）と `--init_ckpt` の入った版（s17 の学習に使ったもの）に置き換えた。追加の引数を指定しなければ s13 と同じ学習になる。置き換えた後は構文の確認だけで、学習は通していない。
- `setup.sh` で外したものをそろえ直す流れは、まだ通しで試していない。

## 中身

| 場所 | 中身 |
|---|---|
| `COMPETITION_SUMMARY.md` | コンペ全体のまとめ（自前モデルの進め方、助言から入れた工夫とスコアの推移、public と private） |
| `figures/`、`scripts/make_summary_figures.py` | まとめの図 6 枚と、その作り直し（散布図の元は締切後に取った提出一覧 `figures/data/submissions_20261024.csv`） |
| `src/geo3d/model.py` | モデル GeoQueryNet（ConvNeXt-Small → 6×6 のマス → 患者座標とシリーズの種類の埋め込み → 所見ごとの注意 MIL） |
| `src/geo3d/data.py` | 前処理（右膝の左右反転、[前, 中, 後] の 3 チャンネル、切り出し）と推論（4 枚の組を全部作って平均） |
| `src/geo3d/geometry.py` | シリーズの選び方（6 種類から 1 本ずつ → 枚数の多い順に 7 本まで）、4 枚の組の作り方、DICOM 座標の計算 |
| `scripts/train_geo3d.py` | 学習の本体（データの読み込み・拡張・損失・施設のずれ・EMA・順位損失・保存） |
| `run_train_5fold.sh` / `run_train_full.sh` | s13 の 5 fold 版（約 17 時間）/ 全データ版 × seed 0〜2（約 10 時間）を作り直す |
| `run_train_s17.sh` | s17 の 5 fold 版 / 全データ版を、入っている s13 の重みから作り直す（1 本 約 1 時間） |
| `run_submit.sh` | 全データ版 3 本だけの形で提出する（s13 または s17。Kaggle Models に上げ → notebook を更新 → 見える test で確認 → 提出） |
| `outputs/geo3d/s13*/`、`s17*/` | 重みと学習ログ・OOF（上の表） |
| `outputs/geo3d/S17_README.md`、`s17_*.json` | s17 の説明、設定・SHA256・推論の設定 |
| `outputs/geo3d/labels_silentfill.csv` | 学習の目標と重み（下の「ラベル」） |
| `outputs/geo3d/labels_consensus.csv` | その元（複数の所見文ラベルの合意） |
| `outputs/geo3d/sites.csv` | 施設（所見文の書式で分けた 24 クラスタ） |
| `outputs/working/codex_report_labels/` | Codex の所見文ラベル（Gold 以外 4,349 件と Gold58） |
| `data/interim/folds_jackee.csv` | fold（崎さんの `labels_fused.csv` の fold 列） |
| `data/external/llm_labels/report_labels_v2.csv` | GPT v2 の所見文ラベル（記載あり / なしの判定に使う） |
| `kaggle_submitted/<instance>/` | 実際に提出した notebook のコード（下の表） |
| `scripts/geo3d_kernel_template.py` | 提出 notebook の雛形（s17 の提出と同じ版。o2・nartaa・R6 と混ぜる機能も入っているが、指定しなければ geo3d だけ） |
| `scripts/geo3d_submit.py` | 提出スクリプト（重みを Kaggle Models に上げ、notebook を作って提出） |
| `scripts/geo3d_predict_test_local.py` | 手元の画像キャッシュで見える test を推論（Kaggle の出力との照合用） |
| `scripts/geo3d_full_check.py` | 全データ学習が壊れていないかの確認 |
| `scripts/geo3d_gold_oof.py` | Gold58 OOF の集計 |
| `scripts/geo3d_s17_decide.py` | s17 の採否の判定（対照 s17ctrl が要る。対照を止めたので実際には使っていない） |
| `scripts/geo3d_make_consensus_labels.py`、`geo3d_make_silent_fill_labels.py`、`geo3d_site_bias.py` | ラベルと施設の CSV を作ったスクリプト（参考。作り直すには、このフォルダに無い元データ（所見文、ほかのラベル、s9 の OOF）が要る） |
| `setup.sh` | 入れていないもの（上の表）をそろえる |
| `scripts/make_gold58.py` | 大会の train.csv から Gold58 を取り出す |
| `requirements-lock.txt` | 固定したライブラリの版（torch 2.14.1+cu126、timm 1.0.30 など） |

## 提出 notebook（`kaggle_submitted/`）

各フォルダに、notebook のコード `geo3d_infer.py`、`kernel-metadata.json`、Kaggle Models に上げたものの説明（`geo3d_manifest.json`、`model-instance-metadata.json`）、見える test 3 件の出力 `submission_visible_test.csv`、提出の記録 `submit_record.json` がある。混ぜた提出には、混ぜ方の記録 `blend_status.json` と、別プロセスで動かした公開 notebook のスクリプト（`raptor_run.py` = nartaa の CoAtNet、`r6_run.py` = heliosli の R6）も置いた。

| フォルダ | 提出 | 中身 | public | private |
|---|---|---|---|---|
| `s13full-3s` | 57368467 | s13 全データ × 3 | 0.953 | 0.940 |
| `s13full-o2` | 57366376 | s13 全データ × 3 + 崎さんの o2（確率の等重み） | 0.954 | 0.942 |
| `s13full-r6` | 57476015 | s13 全データ × 3 + R6（順位で 1/2 ずつ） | 0.957 | **0.945** |
| `s13full-4way` | 57473665 | s13 全データ × 3 + o2 + nartaa + R6（順位で 1/4 ずつ） | 0.958 | 0.944 |
| `s17full-3s` | 57526262 | s17 全データ × 3 | 0.951 | 0.939 |
| `s17full-eq3` | 57531043 | s17 全データ × 3 + o2 + nartaa（順位で 1/3 ずつ） | 0.956 | 0.941 |

- o2 を混ぜる notebook は、こちらの非公開データセット `ritsuki0720/rsna-knee-team-o2-latest`（崎さんの重み 10 本と src）を読む。崎さんの重みはこのフォルダに入れていない。
- nartaa・R6 は公開のデータセットとスクリプトをそのまま使う（`kernel-metadata.json` の `dataset_sources`）。外部データ（OAI など）の規則の上での扱いは、はっきりしないまま使った。

## 学習の設定（s13）

```
python scripts/train_geo3d.py --name s13full_s0 --fold -1 --seed 0 \
  --head mil --inst_w 1.0 --per_series 4 --bs 4 --accum 2 --lr 5e-5 --warm_steps 300 \
  --val_every 4 --pool 2 --epochs 16 --backbone convnext_small.in12k_ft_in1k_384 --crop_mm 140 \
  --label_csv outputs/geo3d/labels_silentfill.csv --site_csv outputs/geo3d/sites.csv --site_sd 0.5
```
（seed 1・2 も同じ。`--fold -1` = 検証に分けず全件で学習。5 fold 版は `--fold F`、`--seed` なし）

s17 はこれに `--epochs 4 --warm_steps 0 --lr 2e-5 --lr_head 1e-4 --val_every 99 --rank_w 0.3 --rank_margin 0.5 --rank_queue 256 --init_ckpt <s13 の重み>` を付けたもの（`run_train_s17.sh`）。

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
- notebook: Kaggle Models の重みと `src/geo3d` を読み、test の DICOM を直接読んで推論（`kaggle_submitted/*/geo3d_infer.py`）。3 本のロジットを平均して sigmoid。T4 1 枚で隠しテスト約 2 時間。
- 手元で見える test を推論する:
  ```
  for S in 0 1 2; do python scripts/geo3d_predict_test_local.py --run s13full_s$S --folds -1 --crops 0.93; done
  ```
  （出力 `outputs/geo3d/s13full_s*/test_local_-1.csv`。3 本のロジットの平均が Kaggle の出力と一致する。s17 は `--run s17full_s$S`）
- 提出: `bash run_submit.sh <新しい instance 名> [s13full|s17full]`（Kaggle への送信なので、必要なときだけ）。
