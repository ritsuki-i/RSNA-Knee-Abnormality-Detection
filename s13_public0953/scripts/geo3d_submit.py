# -*- coding: utf-8 -*-
"""geo3d の重みを Kaggle Models に上げ、推論 notebook を push し、完了後に提出する。

  python scripts/geo3d_submit.py --run g1 --folds 0 --instance g1-f0 --message "geo3d g1 fold0"
  python scripts/geo3d_submit.py ... --no-submit     # notebook の実行まで（提出しない）
  python scripts/geo3d_submit.py --run s9 ... --o2_k 30 --kernel rsna-knee-geo3d-o2-ens   # 崎さんの o2（v13）と確率の等重み平均

状態は outputs/logs/geo3d_submit_<instance>.json に書く。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KG = "/mnt/home/ishikawa/kaggle/tools/bin/kg"
OWNER = "ritsuki0720"
MODEL = "rsna-knee-geo3d"
FRAMEWORK = "pyTorch"
COMP = "rsna-knee-abnormality-detection"
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def kg(*args, check=True, timeout=1800):
    r = subprocess.run([KG, *args], capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if check and r.returncode != 0:
        raise RuntimeError(f"kg {' '.join(args)} failed: {out[-2000:]}")
    return r.returncode, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="例: s2 / s2,s4（複数なら等重みで平均）")
    ap.add_argument("--folds", required=True, help="例: 0 / 0,1,2,3,4 / full")
    ap.add_argument("--instance", required=True)
    ap.add_argument("--message", required=True)
    ap.add_argument("--kernel", default="rsna-knee-geo3d-infer")
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--mix", type=float, default=0.0, help="スライス補助ヘッドの混合比（0 = 検査ヘッドのみ）")
    ap.add_argument("--crops", type=float, nargs="+", default=[0.87],
                    help="推論時の中央切り出しの倍率（TTA）。fold 0 の検証 400 件で 1.0: 0.9159、0.87: 0.9206")
    ap.add_argument("--o2_k", type=int, default=None, help="崎さんの o2 と混ぜる（infer.py の k）。手元の test の予測 outputs/o2_oof/test_k<k>.csv で照合する")
    ap.add_argument("--o2_w", type=float, default=0.5, help="o2 の重み（事前に決めた等重み 0.5）")
    # データセットの版の指定は notebook では効かず最新版が付く（10/15 に確認）ので、v13 の 10 本を非公開の Kaggle Model に置いて版を固定する。
    # また notebook に付く Kaggle Model は 1 つだけ（2 つ指定すると最後の 1 つしか残らない。10/15 に確認）。
    # そこで v13 の 10 本を、こちらだけが更新する非公開データセットに置く（最新版 = v13 の中身。更新しない）
    ap.add_argument("--o2_dataset", default="ritsuki0720/rsna-knee-team-o2-v13")
    ap.add_argument("--o2_res", type=int, default=384, help="o2 の推論の解像度（384 未満なら中央切り出し。崎さんの最新版の infer.py が要る = --o2_dataset ritsuki0720/rsna-knee-team-o2-latest）。"
                    "照合は outputs/o2_oof/test_r<res>_k<k>.csv")
    ap.add_argument("--o2_w_json", default=None, help="所見ごとの o2 の重み {所見: 重み} の JSON（指定すると --o2_w より優先）")
    # nartaa の公開 CoAtNet（2026-10-19、ユーザー指示）。公開 notebook の推論スクリプトをそのまま使う（中身を確かめた版を sha256 で固定）
    ap.add_argument("--raptor_w", default=None, help='nartaa の CoAtNet を順位で混ぜる所見と重みの JSON（例 {"Synovitis": 0.5}）')
    ap.add_argument("--raptor_dataset", default="nartaa/rsna-knee-publication-swa-weights-20261007")
    # heliosli の公開 R6（2026-10-21、ユーザー指示）。公開 notebook のセル 6・7 を抜き出した版（sha256 で固定、scripts/build_r6_run.py）
    ap.add_argument("--r6", action="store_true", help="R6 を最後に別プロセスで動かす")
    ap.add_argument("--r6_datasets", default="heliosli/rsna-knee-r6-frozen-fe055406-w1,heliosli/rsna-orthofoundation-fold0-epoch7-20261004")
    ap.add_argument("--blend", default="prob", choices=["prob", "rank_equal"],
                    help="prob = これまでどおり（o2 は確率の平均、raptor は --raptor_w の順位）、rank_equal = 全モデルを所見ごとの順位で等しく平均")
    ap.add_argument("--budget_s", type=float, default=None, help="geo3d の推論を打ち切る時間（notebook の開始から）")
    ap.add_argument("--geo3d_ref", default=None, help="geo3d だけの予測を照合する、前の提出の出力（submission.csv）")
    a = ap.parse_args()
    RAPTOR_SRC, RAPTOR_SHA = ROOT / "outputs/working/raptor_src/raptor_run.py", "a67f4b0d1a572c958aba7232a7a915faa793c00e8888fb5c1ddd0a8bd7032e76"
    R6_SRC, R6_SHA = ROOT / "outputs/working/r6_src/r6_run.py", "3efe83cd0eca01c1087248e5c584445557bcfba7e5f2ff1095cfe481c5dffd7e"

    state_p = ROOT / f"outputs/logs/geo3d_submit_{a.instance}.json"
    state = {}

    def record(**kw):
        state.update(kw, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        state_p.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        print(json.dumps(kw, ensure_ascii=False), flush=True)

    # 1) 重みと src をまとめる
    # --run s2,s4 のように複数の run を並べると、全 run・全 fold のロジットを等しい重みで平均する
    runs = a.run.split(",")
    mdir = ROOT / f"outputs/working/geo3d_model_{a.instance}"
    shutil.rmtree(mdir, ignore_errors=True)
    (mdir / "geo3d").mkdir(parents=True)
    names = []
    for r in runs:
        for f in a.folds.split(","):
            n = f"fold{f}.pt" if len(runs) == 1 else f"{r}_fold{f}.pt"
            shutil.copy2(ROOT / "outputs/geo3d" / r / f"fold{f}.pt", mdir / n)
            names.append(n)
    for f in ["__init__.py", "geometry.py", "model.py", "data.py"]:
        shutil.copy2(ROOT / "src/geo3d" / f, mdir / "geo3d" / f)
    cod = pd.read_csv(ROOT / "outputs/working/codex_report_labels/non_gold_codex_labels.csv")
    prior = cod[T].astype(float).mean().clip(0.02, 0.98).tolist()
    man = {"weights": names, "prior": prior, "run": a.run, "folds": a.folds, "mix": a.mix, "crops": a.crops}
    if a.o2_k is not None:
        tag = "" if a.o2_res == 384 else f"r{a.o2_res}_"
        assert a.o2_res == 384 or a.o2_dataset.endswith("-latest"), "384 以外は最新版の infer.py の入ったデータセットで"
        chk = pd.read_csv(ROOT / f"outputs/o2_oof/test_{tag}k{a.o2_k}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[T]
        w = json.load(open(a.o2_w_json)) if a.o2_w_json else a.o2_w
        if isinstance(w, dict):
            assert sorted(w) == sorted(T), w
        man["o2"] = {"k": a.o2_k, "w": w, "source": a.o2_dataset, "res": a.o2_res,
                     "check": {u: [float(v) for v in r] for u, r in zip(chk.index, chk.values)}}
    if a.raptor_w:
        import hashlib
        rw = json.loads(a.raptor_w)
        assert rw and all(t in T and 0 < float(v) <= 1 for t, v in rw.items()), rw
        assert hashlib.sha256(RAPTOR_SRC.read_bytes()).hexdigest() == RAPTOR_SHA, "raptor_run.py が確かめた版と違う"
        shutil.copy2(RAPTOR_SRC, mdir / "raptor_run.py")
        man["raptor"] = {"w": rw, "timeout_s": 7200, "source": a.raptor_dataset}
    if a.r6:
        import hashlib
        assert hashlib.sha256(R6_SRC.read_bytes()).hexdigest() == R6_SHA, "r6_run.py が確かめた版と違う"
        shutil.copy2(R6_SRC, mdir / "r6_run.py")
        man["r6"] = {"deadline_h": 8.3, "kill_h": 8.65, "source": a.r6_datasets}
    man["blend"] = a.blend
    if a.blend == "rank_equal":
        assert a.o2_k is not None and a.raptor_w and a.r6, "rank_equal は geo3d・o2・raptor・r6 の 4 つで使う"
    if a.budget_s:
        man["budget_s"] = a.budget_s
    (mdir / "geo3d_manifest.json").write_text(json.dumps(man, indent=1))
    (mdir / "model-instance-metadata.json").write_text(json.dumps({
        "ownerSlug": OWNER, "modelSlug": MODEL, "instanceSlug": a.instance, "framework": FRAMEWORK,
        "overview": "geo3d GeoQueryNet weights", "usage": "", "licenseName": "Apache 2.0", "fineTunable": False,
        "trainingData": [], "modelInstanceType": "Unspecified", "baseModelInstanceId": 0,
        "externalBaseModelUrl": ""}, indent=1))
    record(stage="packaged", files=sorted(p.name for p in mdir.iterdir()))

    # 2) Kaggle Models へ（モデルが無ければ作る。バリエーションが無ければ作り、あれば新しい版）
    tmp = ROOT / "outputs/working/geo3d_kg_tmp"
    tmp.mkdir(exist_ok=True)
    rc, _ = kg("models", "get", f"{OWNER}/{MODEL}", "-p", str(tmp), check=False)
    if rc != 0:
        meta_dir = ROOT / "outputs/working/geo3d_model_meta"
        meta_dir.mkdir(exist_ok=True)
        (meta_dir / "model-metadata.json").write_text(json.dumps({
            "ownerSlug": OWNER, "title": MODEL, "slug": MODEL, "subtitle": "", "isPrivate": True,
            "description": "RSNA Knee geo3d weights", "publishTime": "", "provenanceSources": ""}, indent=1))
        kg("models", "create", "-p", str(meta_dir))
        record(stage="model_created")
    ref = f"{OWNER}/{MODEL}/{FRAMEWORK}/{a.instance}"
    rc, out = kg("models", "instances", "get", ref, "-p", str(tmp), check=False)
    if rc != 0:
        _, out = kg("models", "instances", "create", "-p", str(mdir), "-r", "zip")
    else:
        _, out = kg("models", "instances", "versions", "create", ref, "-p", str(mdir), "-r", "zip", "-n", a.message)
    record(stage="uploaded", out=out[-500:])

    # 版番号を取る（処理が終わるまで待つ）
    version = None
    for _ in range(60):
        rc, out = kg("models", "instances", "versions", "list", ref, "-v", check=False)
        nums = []
        lines = out.strip().splitlines()
        # 版が増えると先頭に「Next Page Token = …」が出て、一覧には他の instance の行も混ざる（10/15 に確認）。見出しの行を探し、自分の instance の行だけ読む
        hi = next((i for i, l in enumerate(lines) if "," in l and "version" in l.lower()), None)
        if hi is not None:
            head = [h.strip().lower() for h in lines[hi].split(",")]
            col = next((i for i, h in enumerate(head) if "version" in h and "number" in h), None)
            if col is None:
                col = next((i for i, h in enumerate(head) if h in ("version", "versionnumber", "id")), 0)
            vcol = next((i for i, h in enumerate(head) if h == "variation"), None)
            for line in lines[hi + 1:]:
                tok = line.split(",")
                if vcol is not None and (len(tok) <= vcol or tok[vcol].strip() != a.instance):
                    continue
                if len(tok) > col and tok[col].strip().isdigit():
                    nums.append(int(tok[col]))
        if nums:
            version = max(nums)
            rc2, files = kg("models", "instances", "versions", "files", f"{ref}/{version}", check=False)
            if rc2 == 0 and "geo3d_manifest.json" in files:
                break
        time.sleep(30)
    if version is None:
        raise RuntimeError(f"版番号が取れない: {out[-500:]}")
    record(stage="model_ready", model_version=version)

    # 3) notebook
    kdir = ROOT / f"outputs/working/geo3d_kernel_{a.instance}"
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/geo3d_kernel_template.py", kdir / "geo3d_infer.py")
    kid = f"{OWNER}/{a.kernel}"
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": a.kernel, "code_file": "geo3d_infer.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "competition_sources": [COMP],
        "dataset_sources": ([a.o2_dataset] if a.o2_k is not None else []) + ([a.raptor_dataset] if a.raptor_w else [])
                           + (a.r6_datasets.split(",") if a.r6 else []),
        "kernel_sources": [], "model_sources": [f"{ref.lower()}/{version}"],
        "machine_shape": "NvidiaTeslaT4"}, indent=1))
    _, out = kg("kernels", "push", "-p", str(kdir), timeout=600)
    record(stage="kernel_pushed", out=out[-300:])
    mv = re.search(r"[Vv]ersion (\d+)", out)
    kver = int(mv.group(1)) if mv else None
    # Kaggle 側で入力が外れることがある（10/15: 2 つ目の Model や、作ったばかりの版が外れた）。push 後の設定を読み直して照合する
    pdir = ROOT / f"outputs/working/geo3d_kernel_{a.instance}_pull"
    shutil.rmtree(pdir, ignore_errors=True)
    pdir.mkdir(parents=True)
    time.sleep(20)
    kg("kernels", "pull", kid, "-p", str(pdir), "-m", check=False)
    try:
        srv = json.loads((pdir / "kernel-metadata.json").read_text())
        meta = json.loads((kdir / "kernel-metadata.json").read_text())
        lost = [x for key in ("model_sources", "dataset_sources") for x in meta[key]
                if x.lower() not in [y.lower() for y in srv.get(key, [])]]
    except Exception as e:  # noqa: BLE001
        lost = None
        record(stage="sources_unchecked", err=repr(e)[:200])
    if lost:
        record(stage="sources_lost", lost=lost, server={k: srv.get(k) for k in ("model_sources", "dataset_sources")})
        sys.exit(1)
    record(stage="kernel_running", kernel_version=kver)
    t0 = time.time()
    while True:
        time.sleep(60)
        rc, st = kg("kernels", "status", kid, check=False)
        s = st.strip().splitlines()[-1] if st.strip() else ""
        if "complete" in s.lower():
            break
        if "error" in s.lower() or "cancel" in s.lower():
            odir = ROOT / f"outputs/working/geo3d_kernel_{a.instance}_output"
            odir.mkdir(exist_ok=True)
            kg("kernels", "output", kid, "-p", str(odir), check=False)
            record(stage="kernel_failed", status=s)
            sys.exit(1)
        if time.time() - t0 > 3 * 3600:
            record(stage="kernel_timeout", status=s)
            sys.exit(1)
    odir = ROOT / f"outputs/working/geo3d_kernel_{a.instance}_output"
    shutil.rmtree(odir, ignore_errors=True)
    odir.mkdir()
    kg("kernels", "output", kid, "-p", str(odir), check=False)
    sub = pd.read_csv(odir / "submission.csv")
    ok = sub.shape[1] == 13 and len(sub) >= 1 and not sub[T].isna().any().any() and \
        ((sub[T] >= 0) & (sub[T] <= 1)).all().all()
    record(stage="kernel_complete", shape=list(sub.shape), valid=bool(ok),
           preview=sub.head(3).round(3).to_dict("records"))
    if not ok:
        sys.exit(1)
    if a.blend == "rank_equal":
        # 見える test: 4 つとも完走したか、各モデルの列が取り違えられていないか（ほかの 3 つの平均との順位相関）、
        # 混ぜ方を手元で計算し直して提出と一致するか、geo3d が前の提出と同じか
        ids = sub.StudyInstanceUID.astype(str).values
        rst = json.loads((odir / "raptor_status.json").read_text()) if (odir / "raptor_status.json").exists() else {}
        st6 = json.loads((odir / "r6_status.json").read_text()) if (odir / "r6_status.json").exists() else {}
        if not rst.get("ok") or not st6.get("ok") or st6.get("scored") != len(ids):
            record(stage="rank4_failed", raptor=rst, r6=st6)
            sys.exit(1)

        def rd(f):
            return pd.read_csv(odir / f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, T]
        mem = {"geo3d": rd("member_geo3d.csv"), "o2": rd("member_o2.csv"), "raptor": rd("raptor_raw.csv"), "r6": rd("ours_sub.csv")}
        ok = {"geo3d": np.ones(len(ids), bool), "o2": np.ones(len(ids), bool),
              "raptor": ~(mem["raptor"] == 0.5).all(axis=1).values,
              "r6": np.isin(ids, json.loads((odir / "r6_scored.json").read_text()))}
        acc, cnt = np.zeros((len(ids), len(T))), np.zeros((len(ids), 1))
        for k, v in mem.items():
            rk = np.zeros_like(acc)
            for j, t in enumerate(T):
                rk[ok[k], j] = pd.Series(v[t].values[ok[k]]).rank(method="average", pct=True).values
            acc, cnt = acc + rk * ok[k][:, None], cnt + ok[k][:, None]
        exp = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.5)
        blend_diff = float(np.abs(exp - sub.set_index(sub.StudyInstanceUID.astype(str)).loc[ids, T].values).max())
        rho = {}
        for k, v in mem.items():
            others = np.mean([w.values for kk, w in mem.items() if kk != k], 0)
            rho[k] = round(float(pd.Series(v.values.ravel()).corr(pd.Series(others.ravel()), method="spearman")), 3)
        g_diff = None
        if a.geo3d_ref:
            ref = pd.read_csv(a.geo3d_ref, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, T]
            g_diff = float(np.abs(ref.values - mem["geo3d"].values).max())
        record(stage="rank4_check", blend_diff=blend_diff, spearman_vs_others=rho, geo3d_vs_ref=g_diff,
               raptor_seconds=rst.get("seconds"), r6=st6, r6_preview=mem["r6"].head(3).round(3).to_dict("records"))
        if not (blend_diff < 1e-6 and min(rho.values()) >= 0.5 and (g_diff is None or g_diff < 1e-4)):
            sys.exit(1)
    elif a.raptor_w:
        # 見える test で nartaa が完走したか、列を取り違えていないか（混ぜていない所見で、こちらの予測との順位相関）
        st = json.loads((odir / "raptor_status.json").read_text()) if (odir / "raptor_status.json").exists() else {}
        rp = odir / "raptor_raw.csv"
        if not st.get("ok") or not rp.exists():
            record(stage="raptor_failed", status=st)
            sys.exit(1)
        raw = pd.read_csv(rp, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[sub.StudyInstanceUID.astype(str)]
        rw = json.loads(a.raptor_w)
        pre_p = odir / "submission_noraptor.csv"
        if pre_p.exists():
            # 混ぜる前の予測と全所見で比べ、混ぜ方も手元で計算し直して照合する
            pre = pd.read_csv(pre_p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[sub.StudyInstanceUID.astype(str)]
            keep = list(T)
            ref = pre
            fail = (raw[T].isna() | (raw[T] == 0.5)).all(axis=1).values
            exp = pre[T].copy()
            for t, wt in rw.items():
                ours = pre[t].rank(pct=True).values
                rk = np.full(len(pre), np.nan)
                rk[~fail] = pd.Series(raw[t].values[~fail]).rank(pct=True).values
                exp[t] = np.where(fail, ours, (1 - float(wt)) * ours + float(wt) * rk)
            blend_diff = float(np.abs(exp[T].values - sub[T].values).max())
        else:
            keep = [t for t in T if t not in rw]
            ref, blend_diff = sub.set_index(sub.StudyInstanceUID.astype(str)), 0.0
        rho = float(pd.Series(raw[keep].values.ravel()).corr(pd.Series(ref[keep].values.ravel()), method="spearman"))
        record(stage="raptor_check", seconds=st.get("seconds"), spearman_vs_ours=round(rho, 3), findings=len(keep),
               blend_diff=blend_diff, raw_preview=raw.head(3).round(3).to_dict("records"))
        if not (rho >= 0.5 and blend_diff < 1e-6):
            sys.exit(1)
    if a.no_submit:
        return

    # 4) 提出（notebook の最新版）
    if kver is None:
        raise RuntimeError("notebook の版番号が取れない（push の出力を確認）")
    _, out = kg("competitions", "submit", COMP, "-k", kid, "-f", "submission.csv", "-v", str(kver),
                "-m", a.message, timeout=600)
    record(stage="submitted", out=out[-300:])
    for _ in range(12 * 9):
        time.sleep(300)
        _, out = kg("competitions", "submissions", COMP, "--csv", check=False)
        for line in out.splitlines():
            if a.message in line:
                record(stage="scoring", row=line)
                low = line.lower()
                if "complete" in low or "error" in low:
                    record(stage="done", row=line)
                    return


if __name__ == "__main__":
    main()
