# s17（s13 + 順位損失の追加学習）

s13 の重みから、検査ヘッドに所見ごとの対の順位損失を足して 4 エポック追加で学習したもの（2026-10-22）。
s13 のファイルは変えていない。

| 版 | 場所 | 元の重み | 評価 |
|---|---|---|---|
| 5 fold 版 `s17rank` | `s17rank/fold{0..4}.pt` | `s13/fold{F}.pt`（同じ fold を抜いて学習したもの。OOF は汚れない） | Gold58 OOF 0.9226（s13 0.9190、差 +0.0036 [−0.0025, +0.0096]） |
| 全データ版 `s17full` | `s17full_s{0,1,2}/fold-1.pt` | `s13full_s{S}/fold-1.pt` | 単体の提出 `57526262`: public 0.951 / private 0.939（s13 全データ版は 0.953 / 0.940） |

- 各 run の `fold*.log`、`fold*_site_b.npy`（学習時の施設のずれ）、`*training_args.json`（チェックポイントから取り出した学習の設定）も置いた。`s17rank/` には OOF（`oof_fold*.csv` = 検査ヘッド、`oof_inst_*` = スライスヘッド、`oof_mix_*` = 半々）もある。
- `s17_parameters.json`: 全データ版 3 本の設定、SHA256、ラベルの順、提出の情報。
- `s17_inference_manifest.json`: 推論の設定（重みのパスはこのフォルダが基準）。
- `S17_WEIGHTS_SHA256.txt`: 全データ版 3 本の照合用（このフォルダで `sha256sum -c S17_WEIGHTS_SHA256.txt`）。16 本すべての照合はフォルダ直下の `WEIGHTS_SHA256.txt`。
- `s17_submission_visible_test.csv`: 単体の提出の、見える test 3 件の予測（`kaggle_submitted/s17full-3s/submission_visible_test.csv` と同じ）。

## 設定

- 順位損失: `--rank_w 0.3 --rank_margin 0.5 --rank_queue 256`。検査ヘッドのロジット（施設のずれ込み）で、所見ごとに目標の差が 0.5 を超える対に softplus(−(s_i − s_j))、重み w_i w_j。バッチが 4 件と小さいので、相手には直近 256 件の検査のロジット（勾配なし）も使う。
- 追加学習: 4 エポック、予備学習なし、エンコーダの学習率 2e-5、ヘッド 1e-4、コサイン、EMA 0.998。ほかは s13 と同じ（ラベル、施設、補助ヘッド 1.0）。施設のずれ b も s13 から引き継ぐ。
- 推論: 140mm 切り出し、入力 384、推論の切り出し 0.93、3 本のロジットの平均 → sigmoid、検査ヘッドのみ（mix 0）。モデルの定義は `src/geo3d/model.py` と同じ。
- 学習し直すには `bash run_train_s17.sh 5fold` / `bash run_train_s17.sh full`（フォルダ直下）。`scripts/train_geo3d.py` は順位損失の入った版に置き換えた（`--rank_w` を指定しなければ s13 と同じ学習になる）。
- 提出し直すには `bash run_submit.sh <instance 名> s17full`。

## 採否について

- 事前に決めた判定は「s17rank > 対照（同じ追加学習で順位損失なし）かつ s17rank ≥ s13」だったが、締切の前に対照の学習を fold 0〜2 で止めた（ユーザー指示）ので、判定はしていない。順位損失そのものの効果と、追加学習の効果は分けられていない。
- private では s13 全データ版（0.940）を上回らなかった（0.939）。Gold58 の +0.004 は揺れの範囲だった。
- 混ぜた提出: s17full + 崎さんの o2 + nartaa を順位で 1/3 ずつ（`57531043`）は public 0.956 / private 0.941。notebook は `kaggle_submitted/s17full-eq3/`。
