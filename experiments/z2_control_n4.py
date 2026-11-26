"""Z2 control: z2_e4_n8.py with N = 4 must reproduce the committed E4_raw.csv.gz rows (N = 4). Runs 4 (dataset, split, channel) jobs
(D1/D2, split 0, dep/ad), compares every L x lambda row. Not a pre-registered gate; it checks that the N = 8 script is the E4 protocol."""
from common import *
import subprocess, glob, shutil, pandas as pd
t0 = time.time()
src = open(os.path.join(os.path.dirname(__file__), "z2_e4_n8.py")).read().replace("N = 8;", "N = 4;").replace("z2_parts", "z2_ctrl")
tmp = os.path.join(os.path.dirname(__file__), "_z2_ctrl_tmp.py"); open(tmp, "w").write(src)
ctrl = os.path.join(RES, "z2_ctrl"); shutil.rmtree(ctrl, ignore_errors=True)
for ds in ("D1", "D2"):
    for ch in ("dep", "ad"):
        subprocess.run([sys.executable, tmp, ds, "0", ch], check=True, capture_output=True)
os.remove(tmp)
e = pd.read_csv(os.path.join(RES, "E4_raw.csv.gz"))
cols = ["ece", "ece_width", "disp", "spread", "mean_h", "ece_exact_pred", "brier", "auc", "ece_const_target"]
mx = {}
for f in sorted(glob.glob(os.path.join(ctrl, "*.csv"))):
    c = pd.read_csv(f)
    for _, r in c.iterrows():
        m = e[(e.dataset == r.dataset) & (e.N == 4) & (e.split == r.split) & (e.ch == r.ch) & (e.L == r.L) & (e.lam == r.lam)]
        assert len(m) == 1
        for k in cols:
            key = f"{r['ch']}/{k}"; d = abs(float(m[k].iloc[0]) - float(r[k])); mx[key] = max(mx.get(key, 0.0), d)
shutil.rmtree(ctrl)
save("Z2_control_N4", dict(jobs="D1,D2 x split 0 x dep,ad", compared="15 (L, lambda) rows per job"), dict(max_abs_diff=mx), t0)
print(json.dumps({k: float("%.3g" % v) for k, v in sorted(mx.items())}, indent=0))
