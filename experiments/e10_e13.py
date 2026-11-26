"""E10 (N-dependence), E11 (T1/T2+CZ mixture Model B vs Aer backend noise model vs hardware), E12 (hyper-parameter fairness), E13 (kernel audit)."""
import zlib
from common import *
import pickle
from joblib import Parallel, delayed
from nqk.channels import *
from nqk.embeddings import *
from nqk.kernel import *
from nqk.krr import ridge, cv_lam, center_train, audit
from nqk.metrics import fit_rate, ece_exact, auc
from nqk.datasets import make_split
from nqk.pipeline import embS_kernel, evaluate_methods
import pandas as pd

t0 = time.time()
HW = os.path.join(RES, "hardware")
OUT = {}

# =================================================================== E10
G10 = 0.05
SNAPS = list(range(5, 81, 3)) + [200]
WIN = (20, 70)


def e10_one(N, emb, ch):
    rng = np.random.default_rng(N)
    P = 6 if N >= 7 else 8
    if emb == "S":
        X = rng.uniform(0, 1, (P, N)); Vs = np.array(emb_R_layers(X, 2.0, N, 8))
    else:
        X = rng.uniform(0, np.pi, (P, N)); Vs = np.array(zz_layers(X, N, "full"))
    Ks = noisy_kernels(Vs, ("local", kraus(ch, G10)), N, SNAPS)
    ref = Ks[200]
    Ls = [L for L in SNAPS if L < 200]
    d = np.array([np.abs(Ks[L] - ref).max() for L in Ls])
    sl, ic, m, c, se = fit_rate(Ls, d, WIN, floor=1e-12)
    D = 2 ** N
    return dict(N=N, emb=emb, ch=ch, slope=sl, se=se, intercept=ic, n_pts=m, K_ref_mean=float(ref.mean()), D=D, K_ref_minus_1overD=float(abs(ref.mean() - 1 / D)))


jobs = [(N, e, c) for N in range(2, 9) for e in ("S", "ZZ") for c in ("dep", "ad")]
jobs.sort(key=lambda j: -j[0])
e10 = Parallel(n_jobs=8)(delayed(e10_one)(*j) for j in jobs)
e10.sort(key=lambda r: (r["emb"], r["ch"], r["N"]))
n_, lam_ = 25, 1.0
ece_rows = [dict(N=N, D=2 ** N, D_lam_over_n=2 ** N * lam_ / n_, ece_inf=float(ece_exact(0.40, 0.37, n_, 2 ** N, lam_))) for N in range(2, 9)]
OUT["E10"] = dict(rows=e10, ece_vs_N=ece_rows, gamma=G10, window=WIN, note="'S' = Emb-R fixed small-angle layer exp(-iH/8) (slice-like, weight-1 dominated); ZZ = full-entanglement ZZFeatureMap layer; reference L=200")
print("E10 done", time.time() - t0, flush=True)
print(pd.DataFrame(e10)[["N", "emb", "ch", "slope", "se", "intercept"]].round(4).to_string(), flush=True)

# =================================================================== E11  (Model B on the HW2 circuits, no fitted parameter)
meta = json.load(open(os.path.join(HW, "prep_meta.json")))
raw = json.load(open(os.path.join(HW, "raw_hw2.json")))
isa = pickle.load(open(os.path.join(HW, "isa_circuits.pkl"), "rb"))
snap = meta["snapshot"]
lay = meta["layout"]
Xall = np.array(meta["Xall"]); beta = meta["beta"]
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from nqk.hw import NQ, slice_gates, get_service
tg = get_service().backend(meta["backend"]).target
t_meas_us = max(tg["measure"][(q,)].duration for q in lay) * 1e6


def slice_U(x, L, inv=False):
    q = QuantumCircuit(NQ)
    slice_gates(q, x, beta, L, inverse=inv)
    return Operator(q).data


T1s = [snap["qubit_props"][str(q)]["T1_us"] for q in lay]
T2s = [min(snap["qubit_props"][str(q)]["T2_us"], 2 * snap["qubit_props"][str(q)]["T1_us"]) for q in lay]
RO = [snap["qubit_props"][str(q)]["readout_err"] for q in lay]
CZE = [snap["cz"][k] for k in snap["cz"]]
settings = [tuple(s) for s in meta["settings"]]
pairs = [tuple(p) for p in meta["pairs"]]
NP = len(pairs)
I2c = [np.eye(2, dtype=complex)]


def thermal_kraus(t_us, T1, T2):
    g = 1 - np.exp(-t_us / T1)
    lphi = 1 - np.exp(-2 * t_us / T2 + t_us / T1)
    return kraus("ad", g), kraus("pd", max(lphi, 0.0))


def apply_thermal(rho, t_us):
    for q in range(NQ):          # circuit qubit q <-> kron position NQ-1-q
        ad, pd_ = thermal_kraus(t_us, T1s[q], T2s[q])
        for kr in (ad, pd_):
            rho = apply_noise(rho, ("local_pq", [kr if i == NQ - 1 - q else I2c for i in range(NQ)]), NQ)
    return rho


L_ = "abcdefghijklmnopqrstuvwxyz"


def apply_cz_dep(rho, a, b, e):
    """Two-qubit depolarising of strength p=16e/15 (so the average gate infidelity equals e) on circuit qubits a,b."""
    p = min(16 / 15 * e, 1.0)
    pa, pb = NQ - 1 - a, NQ - 1 - b
    R = rho[0].reshape([2] * (2 * NQ))
    ket = [L_[i] for i in range(NQ)]
    bra = [L_[i] if i in (pa, pb) else L_[i + NQ] for i in range(NQ)]
    keep = [i for i in range(NQ) if i not in (pa, pb)]
    sub_out = "".join(ket[i] for i in keep) + "".join(bra[i] for i in keep)
    tr = np.einsum("".join(ket) + "".join(bra) + "->" + sub_out, R)
    Iab = np.einsum("ij,kl->ijkl", np.eye(2), np.eye(2)) / 4
    full_out = "".join(L_[i] for i in range(NQ)) + "".join(L_[i + NQ] for i in range(NQ))
    sub_I = L_[pa] + L_[pa + NQ] + L_[pb] + L_[pb + NQ]
    out = np.einsum(sub_out + "," + sub_I + "->" + full_out, tr, Iab)
    return ((1 - p) * rho[0] + p * out.reshape(2 ** NQ, 2 ** NQ))[None]


def model_b_p0(seq, t_slice_us, bonds):
    rho = np.zeros((16, 16), complex); rho[0, 0] = 1; rho = rho[None]
    for U in seq:
        rho = U[None] @ rho @ U.conj().T[None]
        rho = apply_thermal(rho, t_slice_us)
        for (a, b), e in bonds:
            for _ in range(2):          # RZZ = 2 CZ
                rho = apply_cz_dep(rho, a, b, e)
    pr = np.real(np.diag(rho[0]))
    tot = 0.0
    s = 0
    # P(measure 0000) with independent symmetric readout error: sum_s P(s) prod_q P(read 0 | bit_q(s))
    w = np.ones(16)
    for st in range(16):
        for q in range(NQ):
            w[st] *= (RO[q] if (st >> q) & 1 else 1 - RO[q])
    return float((pr * w).sum())


dur = []
for k, (L, f) in enumerate(settings):
    d_ = float(isa["isa2"][k * NP].estimate_duration(tg, unit="s")) * 1e6 - t_meas_us
    dur.append(d_)
bonds = [((0, 1), CZE[0]), ((1, 2), CZE[1]), ((2, 3), CZE[2])]
pmb = np.zeros(len(raw["counts"]))
for k, (L, f) in enumerate(settings):
    nslice = 2 * L * (1 + 2 * f)
    ts = dur[k] / nslice
    for pi, (i, j) in enumerate(pairs):
        seq = []
        for _ in range(L):
            seq.append(slice_U(Xall[i], L))
            for _ in range(f):
                seq += [slice_U(Xall[i], L, True), slice_U(Xall[i], L)]
        for _ in range(L):
            seq.append(slice_U(Xall[j], L, True))
            for _ in range(f):
                seq += [slice_U(Xall[j], L), slice_U(Xall[j], L, True)]
        pmb[k * NP + pi] = model_b_p0(seq, ts, bonds)
p_hw = np.array([c.get("0000", 0) for c in raw["counts"]]) / raw["shots"]
p_aer = np.array(meta["aer"]["hw2_noisy_8192"]); p_id = np.array(meta["aer"]["hw2_exact"])
rows11 = []
for k, (L, f) in enumerate(settings):
    s = slice(k * NP, (k + 1) * NP)
    a_mb = np.polyfit(p_id[s], pmb[s], 1)[0]; a_aer = np.polyfit(p_id[s], p_aer[s], 1)[0]; a_hw = np.polyfit(p_id[s], p_hw[s], 1)[0]
    rows11.append(dict(L=L, fold=f, slice_us=dur[k] / (2 * L * (1 + 2 * f)), circuit_us=dur[k], s_modelB=float(a_mb), s_aer_backend=float(a_aer), s_hw=float(a_hw),
                       mae_modelB_vs_aer=float(np.abs(pmb[s] - p_aer[s]).mean()), mae_modelB_vs_hw=float(np.abs(pmb[s] - p_hw[s]).mean()),
                       mae_aer_vs_hw=float(np.abs(p_aer[s] - p_hw[s]).mean()), mae_ideal_vs_hw=float(np.abs(p_id[s] - p_hw[s]).mean())))
lay_n = np.array([2 * L * (1 + 2 * f) for L, f in settings], float)
rho_mb = float(np.exp(np.polyfit(lay_n, np.log([r["s_modelB"] for r in rows11]), 1)[0]))
OUT["E11"] = dict(rows=rows11, rho_eff_modelB=rho_mb, rho_eff_aer=float(np.exp(np.polyfit(lay_n, np.log([r["s_aer_backend"] for r in rows11]), 1)[0])),
                  rho_eff_hw=float(np.exp(np.polyfit(lay_n, np.log([r["s_hw"] for r in rows11]), 1)[0])),
                  model="per-slice thermal relaxation (calibrated T1,T2 of the 4 qubits; slice duration = transpiled circuit duration minus readout, divided by #slices) + 2-qubit depolarising per CZ (calibrated CZ error) + readout confusion; NO fitted parameter",
                  T1_us=T1s, T2_us=T2s, readout=RO, cz_err=CZE)
print("E11:", pd.DataFrame(rows11).round(3).to_string(), "rho_eff MB %.4f aer %.4f hw %.4f" % (rho_mb, OUT["E11"]["rho_eff_aer"], OUT["E11"]["rho_eff_hw"]), flush=True)
rng = np.random.default_rng(3)
Vs = np.array(emb_R_layers(rng.uniform(0, 1, (10, 4)), 2.0, 4, 8))
asy = []
for tus in (1.0, 5.0, 25.0):
    ad, pd_ = thermal_kraus(tus, 250.0, 150.0)
    kr = [Kp @ Ka for Kp in pd_ for Ka in ad]
    Kinf = dense_kernel_limit(Vs, noise_superop(("local", kr), 4), 8192)
    asy.append(dict(layer_time_us=tus, gamma_T1=float(1 - np.exp(-tus / 250)), asym_max=float(np.abs(Kinf - Kinf.T).max()), Kinf_min=float(Kinf.min()), Kinf_max=float(Kinf.max()),
                    unital=bool(np.allclose(sum(k @ k.conj().T for k in kr), np.eye(2)))))
OUT["E11"]["mixture_limit_asymmetry"] = asy
print(pd.DataFrame(asy).round(4).to_string())
print("E11 done", time.time() - t0, flush=True)

# =================================================================== E12
N12, GAM = 4, 0.02
CS = [0.25, 0.5, 1.0, 2.0, 4.0]
SPL = 10


def e12_run(dname, sp, n, beta_, ch, L, S):
    Xtr, Xte, ytr, yte = make_split(dname, n, sp, n_test_cap=100, scale="unit")
    fam = SliceFamily(np.vstack([Xtr, Xte]), beta_, N12)
    Ks, _ = embS_kernel(fam, ch, GAM, L)
    Kr, _ = embS_kernel(fam, ch, GAM, 1)
    rows = []
    seed = zlib.crc32(repr((dname, sp, n, beta_, ch, L, S)).encode()) % (2 ** 31)
    for c in CS:
        r = evaluate_methods(Ks, Kr, n, ytr, yte, S, np.random.default_rng(seed), c=c)
        for m in ("cnr", "fixed", "cv_cen", "cv_unc", "fixed_centered"):
            rows.append(dict(ds=dname, split=sp, n=n, beta=beta_, ch=ch, L=L, S=S, c=c, method=m, auc=r[m]["auc"], lam=r[m]["lam"]))
    return rows


jobs = [("D1", sp, n, 2.0, ch, 64, S) for sp in range(SPL) for n in (20, 30, 40) for ch in ("dep", "ad", "pd") for S in (256, 1024)] + \
       [("D1", sp, 30, b, ch, 64, S) for sp in range(SPL) for b in (1.0, 3.0) for ch in ("dep", "ad", "pd") for S in (256, 1024)]
rows12 = [r for o in Parallel(n_jobs=10)(delayed(e12_run)(*j) for j in jobs) for r in o]
df12 = pd.DataFrame(rows12)
df12.to_csv(os.path.join(RES, "E12_raw.csv.gz"), index=False)
agg12 = []
for key, g in df12.groupby(["n", "beta", "ch", "S", "c"]):
    piv = g.pivot(index="split", columns="method", values="auc")
    bestcv = piv[["cv_cen", "cv_unc"]].max(axis=1)
    m1 = bootstrap_ci((piv["cnr"] - bestcv).values, 4000); m2 = bootstrap_ci((piv["cnr"] - piv["fixed"]).values, 4000)
    agg12.append(dict(n=int(key[0]), beta=float(key[1]), ch=key[2], S=float(key[3]), c=float(key[4]), cnr=float(piv.cnr.mean()), cv_best=float(bestcv.mean()), cv_cen=float(piv.cv_cen.mean()), cv_unc=float(piv.cv_unc.mean()),
                      fixed=float(piv.fixed.mean()), d_vs_cv=m1[0], d_vs_cv_lo=m1[1], d_vs_cv_hi=m1[2], d_vs_fixed=m2[0], d_vs_fixed_lo=m2[1], d_vs_fixed_hi=m2[2]))
A12 = pd.DataFrame(agg12)
c1 = A12[A12.c == 1.0]
OUT["E12"] = dict(rows=agg12, cv_grid="10^-8..10^2 (11 values)", folds=5, dataset="D1", L=64, gamma=GAM, splits=SPL,
                  c1_cells=len(c1), c1_cnr_better_than_cv=int((c1.d_vs_cv_lo > 0).sum()), c1_cnr_worse_than_cv=int((c1.d_vs_cv_hi < 0).sum()),
                  c1_cnr_better_than_fixed=int((c1.d_vs_fixed_lo > 0).sum()), c1_cnr_worse_than_fixed=int((c1.d_vs_fixed_hi < 0).sum()),
                  by_c=A12.groupby("c")[["cnr", "cv_best", "fixed"]].mean().reset_index().to_dict("records"),
                  by_n=c1.groupby("n")[["cnr", "cv_best", "fixed"]].mean().reset_index().to_dict("records"),
                  by_beta=c1[c1.n == 30].groupby("beta")[["cnr", "cv_best", "fixed"]].mean().reset_index().to_dict("records"))
pd.set_option("display.width", 250)
print(c1.round(3).to_string())
print(pd.DataFrame(OUT["E12"]["by_c"]).round(3).to_string())
print("E12 done", time.time() - t0, flush=True)

# =================================================================== E13
rows13 = []
Xtr, Xte, ytr, yte = make_split("D1", 30, 0, n_test_cap=30, scale="unit")
fam = SliceFamily(np.vstack([Xtr, Xte]), 2.0, 4)
for ch in ("dep", "ad", "pd", "bf"):
    for L in (1, 4, 16, 64):
        _, Kraw = embS_kernel(fam, ch, 0.02, L)
        Ktr = Kraw[:30, :30]
        a = audit(Ktr); a.update(dict(embedding="Emb-S", ch=ch, L=L, S="inf")); rows13.append(a)
        Ksh = np.random.default_rng(L).binomial(1024, np.clip(Ktr, 0, 1)) / 1024      # every entry an independent hardware-style estimate
        b = audit(Ksh); b.update(dict(embedding="Emb-S", ch=ch, L=L, S=1024)); rows13.append(b)
X = np.random.default_rng(1).uniform(0, np.pi, (30, 4)); Vs = np.array(zz_layers(X, 4, "full"))
for ch in ("dep", "ad", "pd", "bf"):
    Ks = noisy_kernels(Vs, ("local", kraus(ch, 0.05)), 4, [1, 4, 16, 64])
    for L in (1, 4, 16, 64):
        a = audit(Ks[L]); a.update(dict(embedding="Emb-ZZ", ch=ch, L=L, S="inf")); rows13.append(a)
df13 = pd.DataFrame(rows13)
inf = df13[df13.S == "inf"]
OUT["E13"] = dict(rows=rows13, summary_Sinf=inf.groupby(["embedding", "ch"])[["asym", "mass_removed", "eff_rank"]].max().reset_index().merge(
    inf.groupby(["embedding", "ch"]).min_eig.min().reset_index()).to_dict("records"),
    summary_S1024=df13[df13.S == 1024].groupby("ch")[["asym", "mass_removed"]].max().reset_index().merge(df13[df13.S == 1024].groupby("ch").min_eig.min().reset_index()).to_dict("records"))
print(pd.DataFrame(OUT["E13"]["summary_Sinf"]).round(5).to_string())
print(pd.DataFrame(OUT["E13"]["summary_S1024"]).round(5).to_string())
save("E10_E13_N_mixtures_fairness_audit", dict(note="E10 N-scan, E11 mixtures vs Aer vs HW, E12 hyper-parameter fairness, E13 audit"), OUT, t0)
