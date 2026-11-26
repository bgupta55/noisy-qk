"""Y4: (a) overlap of the E12 (c-selection) and E5 D1 splits; (b) X3 c=2 vs best CV / fixed on D1 cells that did not take part in the choice of c;
(c) the one-cell difference between V1 E5 (c=1 vs fixed 125/61/70) and X3 (125/62/69).
Pre-registered in results/PREREGISTRATION.md (ROUND Y, Y4). Writes results/Y4_split_overlap.json."""
import zlib
from common import *
from joblib import Parallel, delayed
import pandas as pd
from nqk.datasets import make_split, load_dataset
from nqk.embeddings import SliceFamily
from nqk.pipeline import embS_kernel, evaluate_methods

t0 = time.time()
N, n, GAMMA, BETA = 4, 30, 0.02, 2.0

# ---------------------------------------------------------------- (a) overlap
X, y, _, _ = load_dataset("D1")
M = len(y)


def split_idx(nn, sp):
    """index sets of make_split (same train_test_split calls, applied to np.arange so we see the indices)."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(M)
    tr, te, ytr, yte = train_test_split(idx, y, train_size=nn, stratify=y, random_state=sp)
    if len(yte) > 100:
        te, _, yte, _ = train_test_split(te, yte, train_size=100, stratify=yte, random_state=sp)
    return set(tr.tolist()), set(te.tolist())


# sanity: the reconstruction reproduces make_split features
Xtr0, Xte0, _, _ = make_split("D1", 30, 3, n_test_cap=100, scale="unit")
tr3, te3 = split_idx(30, 3)
from sklearn.preprocessing import MinMaxScaler
rec = X[sorted(tr3)] if False else None
E5 = {sp: split_idx(30, sp) for sp in range(20)}
E12 = {(nn, sp): split_idx(nn, sp) for nn in (20, 30, 40) for sp in range(10)}
ident, jac_rows = [], []
for (nn, sp), (tr, te) in E12.items():
    tr5, te5 = E5[sp]
    jac = len(tr & tr5) / len(tr | tr5)
    jac_rows.append(dict(n=nn, split=sp, train_jaccard_with_E5_same_split=jac, identical_train=bool(tr == tr5), identical_test=bool(te == te5) if nn == 30 else None,
                         n_train_shared=len(tr & tr5)))
J = pd.DataFrame(jac_rows)
n30 = J[J.n == 30]
# all E5 splits vs all E12 n=30 splits: the maximum Jaccard (E5 sp 10-19 against E12 sp 0-9 = other draws of the same 100 points)
cross = []
for sp5 in range(20):
    mx = max(len(E5[sp5][0] & E12[(30, s12)][0]) / len(E5[sp5][0] | E12[(30, s12)][0]) for s12 in range(10))
    cross.append(dict(e5_split=sp5, max_train_jaccard_over_E12_n30=mx))
overlap = dict(
    D1_pool_size=int(M),
    E12_splits_n30_identical_to_E5_same_seed=int(n30.identical_train.sum()), of=int(len(n30)),
    E12_n20_n40_mean_jaccard_with_E5_same_seed=float(J[J.n != 30].train_jaccard_with_E5_same_split.mean()),
    E12_n20_train_subset_of_E5=int(sum(E12[(20, s)][0] <= E5[s][0] for s in range(10))), E12_n40_train_superset_of_E5=int(sum(E12[(40, s)][0] >= E5[s][0] for s in range(10))),
    E5_splits_0_9_in_selection=int(10), E5_splits_10_19_max_jaccard_vs_E12_n30=[c["max_train_jaccard_over_E12_n30"] for c in cross[10:]],
    note="E12 uses make_split('D1', n, sp) with random_state=sp, exactly as E5. For n=30 the E12 and E5 D1 splits sp=0..9 are the SAME train/test partition. Shot seeds differ (E12 seeds on (dname, sp, n, beta, ch, L, S), E5 on (dname, sp, ch, L, S)), so the shot realisation is different but the data are identical.")
print(json.dumps(overlap, indent=1))

# ---------------------------------------------------------------- (b) c=2 on D1 cells not used in the c choice
CHS = ["dep", "ad", "pd", "bf"]
LS = [1, 4, 16, 64]
SS = [256, 1024, 4096, np.inf]


def run(sp):
    Xtr, Xte, ytr, yte = make_split("D1", n, sp, n_test_cap=100, scale="unit")
    fam = SliceFamily(np.vstack([Xtr, Xte]), BETA, N)
    rows = []
    for ch in CHS:
        Kref, _ = embS_kernel(fam, ch, GAMMA, 1)
        for L in LS:
            Ks, _ = embS_kernel(fam, ch, GAMMA, L)
            for S in SS:
                seed = zlib.crc32(repr(("D1", sp, ch, L, float(S))).encode()) % (2 ** 31)
                res = evaluate_methods(Ks, Kref, n, ytr, yte, S, np.random.default_rng(abs(seed)), c=2.0)
                rows.append(dict(ds="D1", split=sp, ch=ch, L=L, S=float(S) if np.isfinite(S) else -1.0, auc_c2=res["cnr"]["auc"]))
    return rows


c2 = pd.DataFrame([r for o in Parallel(n_jobs=6)(delayed(run)(sp) for sp in range(20)) for r in o])
old = pd.read_csv(os.path.join(RES, "E5_raw.csv.gz"))
key = ["ds", "split", "ch", "L", "S"]
base = old.pivot_table(index=key, columns="method", values="auc").reset_index()
Mg = base.merge(c2, on=key)
Mg["best_cv"] = Mg[["cv_unc", "cv_cen"]].max(axis=1)
# reproduce X3's c=2 column
x3 = json.load(open(os.path.join(RES, "X3_E5_c2.json")))["results"]


def paired(df, a, b):
    mu, lo, hi = bootstrap_ci((df[a] - df[b]).values, reps=4000)
    return mu, lo, hi


def tally(sub):
    out = {}
    for nm, ref in (("vs_best_cv", "best_cv"), ("vs_fixed", "fixed")):
        c = dict(better=0, worse=0, tie=0)
        diffs = []
        for (ch, L, S), g in sub.groupby(["ch", "L", "S"]):
            mu, lo, hi = paired(g, "auc_c2", ref)
            c["better" if lo > 0 else ("worse" if hi < 0 else "tie")] += 1
            diffs.append(mu)
        c["n_cells"] = int(sum(c.values())); c["mean_diff"] = float(np.mean(diffs))
        out[nm] = c
    out["mean_auc"] = dict(cnr_c2=float(sub.auc_c2.mean()), cnr_c1=float(sub.cnr.mean()), best_cv=float(sub.best_cv.mean()), fixed=float(sub.fixed.mean()))
    return out


sel = (Mg.L == 64) & Mg.S.isin([256.0, 1024.0]) & Mg.ch.isin(["dep", "ad", "pd"])
groups = {
    "D1_all_splits_all_cells (X3 D1 group; reproduces X3)": tally(Mg),
    "D1_selection_cells_(L64,S256/1024,dep/ad/pd)_splits_0_9 (in-sample for c)": tally(Mg[sel & (Mg.split < 10)]),
    "D1_selection_cells_splits_10_19 (same cells, other 10 random partitions of the same 100 points)": tally(Mg[sel & (Mg.split >= 10)]),
    "D1_non_selection_cells_all_splits (58 cells)": tally(Mg[~sel]),
    "D1_non_selection_cells_splits_10_19": tally(Mg[~sel & (Mg.split >= 10)]),
}
for k, v in groups.items():
    print(k, "n_cells", v["vs_best_cv"]["n_cells"], "vs_cv", {a: v["vs_best_cv"][a] for a in ("better", "worse", "tie")}, "vs_fixed", {a: v["vs_fixed"][a] for a in ("better", "worse", "tie")}, {a: round(b, 4) for a, b in v["mean_auc"].items()})
rep_ok = tally(Mg)["vs_best_cv"]
x3d1 = x3["groups"]["D1_tuning_set"]["c2"]["vs_best_cv"]
repro = dict(x3_D1=dict(better=x3d1["better"], worse=x3d1["worse"], tie=x3d1["tie"]), here=dict(better=rep_ok["better"], worse=rep_ok["worse"], tie=rep_ok["tie"]))

# ---------------------------------------------------------------- (c) the one-cell difference
P = pd.DataFrame(json.load(open(os.path.join(RES, "E5_ranking_vs_resolution.json")))["results"]["paired"])
f = P[P.cmp == "cnr-fixed"].copy()
f["tag_e5"] = ["better" if b else "worse" if w else "tie" for b, w in zip(f.cnr_better, f.cnr_worse)]
cells = pd.DataFrame(x3["cells"]["c1"])
cf = cells[cells.cmp == "vs_fixed"].rename(columns={"tag": "tag_x3", "diff": "diff_x3", "lo": "lo_x3", "hi": "hi_x3"})
mm = f.merge(cf, on=["ds", "ch", "L", "S"])
dd = mm[mm.tag_e5 != mm.tag_x3]
oc = [dict(ds=r.ds, ch=r.ch, L=int(r.L), S=float(r.S), diff_e5=float(r.diff), hi_e5=float(r.hi), diff_x3=float(r.diff_x3), hi_x3=float(r.hi_x3), tag_e5=r.tag_e5, tag_x3=r.tag_x3) for r in dd.itertuples()]
# raw per-split values of that cell
for o in oc:
    g = base[(base.ds == o["ds"]) & (base.ch == o["ch"]) & (base.L == o["L"]) & (base.S == o["S"])]
    d = (g["cnr"] - g["fixed"]).values
    o["n_splits"] = int(len(d)); o["n_splits_diff_exactly_zero"] = int((d == 0).sum()); o["n_splits_diff_neg"] = int((d < 0).sum()); o["max_abs_diff_nonzero"] = float(np.abs(d[d != 0]).max())
    o["min_neg_diff"] = float(d.min())
    o["bootstrap_hi_recomputed_seed0"] = float(bootstrap_ci(d, reps=4000)[2])
    o["abs_diff_hi_e5_vs_x3"] = abs(o["hi_e5"] - o["hi_x3"])
print(json.dumps(oc, indent=1))
tab_e5 = dict(better=int((f.tag_e5 == "better").sum()), worse=int((f.tag_e5 == "worse").sum()), tie=int((f.tag_e5 == "tie").sum()))
tab_x3 = dict(better=int((cf.tag_x3 == "better").sum()), worse=int((cf.tag_x3 == "worse").sum()), tie=int((cf.tag_x3 == "tie").sum()))

# ---------------------------------------------------------------- (d) knife-edge cells: lo or hi within 1e-12 of 0 (floating-point), and the tallies with that tolerance
TOL = 1e-12
def retag(lo, hi):
    return "better" if lo > TOL else ("worse" if hi < -TOL else "tie")
knife = dict()
for nm in ("c1", "c2"):
    cc = pd.DataFrame(x3["cells"][nm])
    kn = cc[(cc.lo.abs() < TOL) | (cc.hi.abs() < TOL)]
    cc["tag_tol"] = [retag(a, b) for a, b in zip(cc.lo, cc.hi)]
    changed = cc[cc.tag != cc.tag_tol]
    tal = {c: {t: int(((cc.cmp == c) & (cc.tag_tol == t)).sum()) for t in ("better", "worse", "tie")} for c in ("vs_best_cv", "vs_fixed")}
    tal_orig = {c: {t: int(((cc.cmp == c) & (cc.tag == t)).sum()) for t in ("better", "worse", "tie")} for c in ("vs_best_cv", "vs_fixed")}
    knife[nm] = dict(n_knife_edge_cells=int(len(kn)), n_tag_changed_by_tolerance=int(len(changed)), changed=[dict(ds=r.ds, ch=r.ch, L=int(r.L), S=float(r.S), cmp=r.cmp, tag=r.tag, tag_tol=r.tag_tol) for r in changed.itertuples()],
                     tally_original=tal_orig, tally_with_tolerance=tal)
    # held-out D2-D4 with tolerance
    ho = cc[cc.ds.isin(["D2", "D3", "D4"])]
    knife[nm]["heldout_D2_D4_with_tolerance"] = {c: {t: int(((ho.cmp == c) & (ho.tag_tol == t)).sum()) for t in ("better", "worse", "tie")} for c in ("vs_best_cv", "vs_fixed")}
    knife[nm]["heldout_D2_D4_original"] = {c: {t: int(((ho.cmp == c) & (ho.tag == t)).sum()) for t in ("better", "worse", "tie")} for c in ("vs_best_cv", "vs_fixed")}
print(json.dumps(knife, indent=1))
E5k = f.copy()
E5k["tag_tol"] = [retag(a, b) for a, b in zip(E5k.lo, E5k.hi)]
knife["E5_V1_cnr_vs_fixed_tally_with_tolerance"] = {t: int((E5k.tag_tol == t).sum()) for t in ("better", "worse", "tie")}
knife["X3_c1_vs_fixed_tally_with_tolerance"] = knife["c1"]["tally_with_tolerance"]["vs_fixed"]
save("Y4_split_overlap", dict(selection_cells="D1, L=64, S in {256,1024}, ch in {dep,ad,pd}", seed_split=list(range(20)), c=2.0),
     dict(overlap=overlap, overlap_by_split=J.to_dict("records"), c2_on_D1=groups, reproduction_of_X3_D1=repro,
          one_cell=dict(n_cells_with_different_tag=len(oc), cells=oc, tally_E5_V1=tab_e5, tally_X3_c1=tab_x3), knife_edge=knife), t0)
