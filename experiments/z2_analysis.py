"""Z2 aggregation: concatenates the 12 part files, applies the pre-registered gates (ROUND Z). AUC columns are not reported."""
from common import *
import glob, pandas as pd
t0 = time.time()
files = sorted(glob.glob(os.path.join(RES, "z2_parts", "E4N8_*.csv")))
expected = [(d, s, c) for d in ("D2", "D1") for s in range(3) for c in ("dep", "ad")]
got = {tuple(os.path.basename(f)[5:-4].replace("sp", "").split("_")[0:3]) for f in files}
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
missing = [e for e in expected if (e[0], str(e[1]), e[2]) not in got]
rep = {}
d = df[(df.ch == "dep") & (df.L == 256)]
cells = []
for (ds, lam), g in d.groupby(["dataset", "lam"]):
    cells.append(dict(dataset=ds, lam=lam, n_splits=int(len(g)), ece_width=float(g.ece_width.mean()), ece_exact=float(g.ece_exact_pred.mean()),
                      abs_diff_width=float((g.ece_width - g.ece_exact_pred).abs().max()), disp_max=float(g.disp.max()),
                      ece_quantile=float(g.ece.mean())))
a = df[(df.ch == "ad") & (df.L == 256)]
ad_cells = []
for (ds, lam), g in a.groupby(["dataset", "lam"]):
    ad_cells.append(dict(dataset=ds, lam=lam, n_splits=int(len(g)), ece_width=float(g.ece_width.mean()), ece_const_target=float(g.ece_const_target.mean()),
                         disp=float(g.disp.mean()), disp_max=float(g.disp.max())))
gA = bool(len(cells) == 6 and all(c["abs_diff_width"] <= 1e-8 for c in cells))
gB = bool(len(cells) == 6 and all(c["disp_max"] <= 1e-4 for c in cells))
res = dict(missing_jobs=[list(map(str, m)) for m in missing], n_part_files=len(files),
           gate_Z2a_prop6_dep=dict(passed=gA, max_abs_diff_width=max(c["abs_diff_width"] for c in cells) if cells else None, tol=1e-8, n_cells=len(cells)),
           gate_Z2b_dep_collapse=dict(passed=gB, max_disp=max(c["disp_max"] for c in cells) if cells else None, tol=1e-4),
           dep_cells=cells, ad_cells=ad_cells,
           ad_ece_width_range=[min(c["ece_width"] for c in ad_cells), max(c["ece_width"] for c in ad_cells)] if ad_cells else None,
           ad_disp_max=max(c["disp_max"] for c in ad_cells) if ad_cells else None)
curves = df.groupby(["dataset", "ch", "lam", "L"])[["ece_width", "ece_exact_pred", "disp", "spread"]].mean().reset_index()
res["curves"] = curves.to_dict("records")
save("Z2_E4_N8", dict(N=8, n=25, n_test=50, splits=[0, 1, 2], Ls=[1, 4, 16, 64, 256], gamma=0.05), res, t0)
print(json.dumps({k: res[k] for k in ("missing_jobs", "gate_Z2a_prop6_dep", "gate_Z2b_dep_collapse", "ad_ece_width_range", "ad_disp_max")}, indent=1))
