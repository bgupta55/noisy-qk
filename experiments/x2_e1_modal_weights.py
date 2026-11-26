"""X2: E1 modal-weight diagnostic (explanation of the non-tightness of the spectral rate; NOT a theorem, no gate).
Same layers, subsets (rng 7, 5 subsets of 10 from the pool of 25), channels, gammas, window [30,80] as e1_rates.py.
Exact eigen-expansion of both arms of every point:
    A_i(L) = S_i^L a0 = sum_k c_ik lam_ik^L v_ik ,   B_j(L) = (Sp_j^T)^L a0 = sum_m d_jm mu_jm^L u_jm ,   K_ij(L) = Re sum_km w_ijkm (lam_ik mu_jm)^L
    w_ijkm = c_ik d_jm (v_ik . u_jm)   (bilinear dot, no conjugation), i.e. each eigenmode is weighted by its overlap with Pi0 (c, d) and with
    the measurement (the second Pi0 / the other arm, v.u).
Predictions of the observed window slope:
    heavy      : ln|lam mu| of the heaviest non-(unit x unit) term at L_c=55 over all (i,j) of the subset (pre-registered primary)
    heavy_pair : same, but only among the pair (i,j) that attains delta at L_c (diagnostic)
    envelope   : OLS slope over L in [30,80] of ln max_{ij,km, not unit x unit} |w| |lam mu|^L
Gate for validity of the expansion: reconstruction error of K at L=30, 80 against the dense matrix power < 1e-6 (else flagged)."""
from common import *
from joblib import Parallel, delayed
import pandas as pd
from nqk.channels import *
from nqk.embeddings import *
from nqk.kernel import *
from nqk.metrics import fit_rate

t0 = time.time()
N, D = 4, 16
POOL, SUBS = 25, 5
GAMMAS = [0.02, 0.05, 0.10]
CHS = ["gdep", "dep", "bf", "pd", "ad"]
WIN = (30, 80)
LC = 55
cfg = dict(N=N, pool=POOL, subsets=SUBS, gammas=GAMMAS, channels=CHS, window=WIN, L_c=LC, expansion_gate=1e-6)


def make_layers(emb, seed):
    rng = np.random.default_rng(seed)
    if emb == "ZZ":
        X = rng.uniform(0, np.pi, (POOL, N))
        return zz_layers(X, N, "full")
    X = rng.uniform(0, 1, (POOL, N))
    return emb_R_layers(X, 2.0, N, 8)


def noise_for(ch, g):
    return ("global", g) if ch == "gdep" else ("local", kraus(ch, g))


def eig_arm(M, a0):
    w, V = np.linalg.eig(M)
    c = np.linalg.solve(V, a0)
    return w, V, c


def one(emb, g, ch):
    Vs = make_layers(emb, 100)
    noise = noise_for(ch, g)
    Nsup = noise_superop(noise, N)
    a0 = np.zeros(D * D, complex)
    a0[0] = 1
    Kref = dense_kernel_limit(Vs, Nsup, 8192)
    arms_f, arms_b = [], []
    for V in Vs:
        S, Sp = dense_layer_maps(V, Nsup)
        arms_f.append(eig_arm(S, a0))
        arms_b.append(eig_arm(Sp.T, a0))
    # observed slope per subset, recomputed independently from the expansion (not from E1.json) -> also a consistency check
    Kexp = {}
    for L in range(WIN[0], WIN[1] + 1):
        A = np.array([(V @ (c * w ** L)) for (w, V, c) in arms_f])
        B = np.array([(U @ (d * m ** L)) for (m, U, d) in arms_b])
        Kexp[L] = np.real(A @ B.T)
    Kd30 = dense_kernel(Vs, Nsup, 30, N)
    Kd80 = dense_kernel(Vs, Nsup, 80, N)
    recon = float(max(np.abs(Kexp[30] - Kd30).max(), np.abs(Kexp[80] - Kd80).max()))
    rng = np.random.default_rng(7)
    k_unit = 2 if ch != "ad" else 1
    res = dict(obs=[], heavy=[], heavy_pair=[], env=[], cls=[], heavy_modmag=[], heavy_w=[], heavy_lam=[], heavy_mu=[])
    Ls = np.arange(WIN[0], WIN[1] + 1)
    for s in range(SUBS):
        sub = np.sort(rng.choice(POOL, 10, replace=False))
        d = np.array([np.abs(Kexp[L][np.ix_(sub, sub)] - Kref[np.ix_(sub, sub)]).max() for L in Ls])
        res["obs"].append(fit_rate(Ls, d, WIN, floor=1e-13)[0])
        # term table for the subset
        best = None
        logmax = np.full(len(Ls), -np.inf)
        cls_mag = dict(UN=0.0, NU=0.0, NN=0.0)
        pair_best = {}
        dL = np.array([np.abs(Kexp[LC][i, j] - Kref[i, j]) for i in sub for j in sub]).reshape(10, 10)
        ip, jp = np.unravel_index(np.argmax(dL), dL.shape)
        for a_, i in enumerate(sub):
            lam, Vf, c = arms_f[i]
            ul = np.abs(np.abs(lam) - 1) < 1e-7
            for b_, j in enumerate(sub):
                mu, U, dd = arms_b[j]
                um = np.abs(np.abs(mu) - 1) < 1e-7
                G = Vf.T @ U
                W = (c[:, None] * G) * dd[None, :]
                R = np.abs(lam)[:, None] * np.abs(mu)[None, :]
                skip = ul[:, None] & um[None, :]
                mag = np.abs(W) * R ** LC
                mag[skip] = 0
                # class totals (max magnitude of each class at L_c)
                cls_mag["UN"] = max(cls_mag["UN"], float(mag[ul[:, None] & ~um[None, :]].max(initial=0)))
                cls_mag["NU"] = max(cls_mag["NU"], float(mag[~ul[:, None] & um[None, :]].max(initial=0)))
                cls_mag["NN"] = max(cls_mag["NN"], float(mag[~ul[:, None] & ~um[None, :]].max(initial=0)))
                kk = np.unravel_index(np.argmax(mag), mag.shape)
                if best is None or mag[kk] > best[0]:
                    best = (float(mag[kk]), float(np.log(R[kk])), float(np.abs(W[kk])), complex(lam[kk[0]]), complex(mu[kk[1]]))
                if a_ == ip and b_ == jp:
                    pair_best = dict(slope=float(np.log(R[kk])))
                # envelope: only terms that matter (>= 1e-16 at L=30)
                m30 = np.abs(W) * R ** 30
                m30[skip] = 0
                idx = np.where(m30 > 1e-16)
                if len(idx[0]):
                    lw = np.log(np.abs(W[idx])); lr = np.log(R[idx])
                    logmax = np.maximum(logmax, np.max(lw[None, :] + Ls[:, None] * lr[None, :], axis=1))
        res["heavy"].append(best[1]); res["heavy_modmag"].append(best[0]); res["heavy_w"].append(best[2])
        res["heavy_lam"].append(abs(best[3])); res["heavy_mu"].append(abs(best[4]))
        res["heavy_pair"].append(pair_best.get("slope", np.nan))
        ok = np.isfinite(logmax)
        res["env"].append(float(np.polyfit(Ls[ok], logmax[ok], 1)[0]) if ok.sum() > 3 else np.nan)
        res["cls"].append(cls_mag)
    return dict(emb=emb, gamma=g, ch=ch, recon_err=recon, **{k: v for k, v in res.items()})


jobs = [(e, g, c) for e in ("ZZ", "R") for g in GAMMAS for c in CHS]
out = Parallel(n_jobs=6)(delayed(one)(*j) for j in jobs)
e1 = {(r["emb"], r["gamma"], r["ch"]): r for r in json.load(open(os.path.join(RES, "E1_rates.json")))["results"]["rows"]}
rows = []
for r in out:
    ref = e1[(r["emb"], r["gamma"], r["ch"])]
    obs = float(np.nanmean(r["obs"]))
    se = ref["se"]
    tol = 2 * max(se, 1e-3)
    row = dict(emb=r["emb"], gamma=r["gamma"], ch=r["ch"], recon_err=r["recon_err"], expansion_valid=bool(r["recon_err"] < 1e-6),
               obs_slope_recomputed=obs, obs_slope_E1=ref["obs_slope"], se=se,
               pred_spectral=ref["pred_spectral"], pred_weighted=ref["pred_weighted"],
               pred_heavy=float(np.nanmean(r["heavy"])), pred_heavy_pair=float(np.nanmean(r["heavy_pair"])), pred_env=float(np.nanmean(r["env"])),
               heavy_rho_lam=float(np.mean(r["heavy_lam"])), heavy_rho_mu=float(np.mean(r["heavy_mu"])),
               class_mag_UN=float(np.mean([c["UN"] for c in r["cls"]])), class_mag_NU=float(np.mean([c["NU"] for c in r["cls"]])), class_mag_NN=float(np.mean([c["NN"] for c in r["cls"]])))
    for p in ("spectral", "weighted", "heavy", "heavy_pair", "env"):
        row[f"within2se_{p}"] = bool(abs(obs - row[f"pred_{p}"]) < tol)
        row[f"abs_err_{p}"] = float(abs(obs - row[f"pred_{p}"]))
    row["rel_err_heavy"] = float((obs - row["pred_heavy"]) / abs(obs)) if obs != 0 else None
    rows.append(row)
df = pd.DataFrame(rows)
valid = df[df.expansion_valid]
summ = dict(n_rows=len(df), n_valid=int(len(valid)), recon_err_max=float(df.recon_err.max()),
            obs_recompute_vs_E1_max_abs=float((df.obs_slope_recomputed - df.obs_slope_E1).abs().max()))
for p in ("spectral", "weighted", "heavy", "heavy_pair", "env"):
    summ[p] = dict(within2se=int(valid[f"within2se_{p}"].sum()), median_abs_err=float(valid[f"abs_err_{p}"].median()), mean_abs_err=float(valid[f"abs_err_{p}"].mean()))
    summ[p]["by_channel_median_abs_err"] = {c: float(g[f"abs_err_{p}"].median()) for c, g in valid.groupby("ch")}
    summ[p]["within2se_AD"] = f"{int(valid[valid.ch == 'ad'][f'within2se_{p}'].sum())}/{int((valid.ch == 'ad').sum())}"
pd.set_option("display.width", 300)
print(df[["emb", "gamma", "ch", "recon_err", "obs_slope_recomputed", "obs_slope_E1", "pred_spectral", "pred_weighted", "pred_heavy", "pred_heavy_pair", "pred_env", "heavy_rho_lam", "heavy_rho_mu"]].round(4).to_string())
print(json.dumps(summ, indent=1))
save("X2_E1_modal_weights", cfg, dict(rows=rows, summary=summ), t0)
