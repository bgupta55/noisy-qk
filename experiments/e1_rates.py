"""E1: rate identification (spectral vs mu/u predictions). Model B, N=4, fixed repeated layers (Emb-ZZ and Emb-R).
5 point-subsets of 10 from a pool of 25; delta(L)=max_ij |K_L - K_ref|, K_ref = dense matrix power L=8192 (exact limit);
slope fitted over pre-registered window; prediction = spectral (max over points in subset), pairwise-product variant also stored."""
from common import *
from joblib import Parallel, delayed
from nqk.channels import *
from nqk.embeddings import *
from nqk.kernel import *
from nqk.metrics import fit_rate

t0 = time.time()
N, D = 4, 16
POOL, SUBS, LMAX = 25, 5, 240
Ls = list(range(1, LMAX + 1))
GAMMAS = [0.02, 0.05, 0.10]
CHS = ["gdep", "dep", "bf", "pd", "ad"]
WINDOW = {0.02: (30, 80), 0.05: (30, 80), 0.10: (30, 80)}   # pre-registered window for all gamma
WINDOW2 = (80, 160)                                          # diagnostic late window (not a gate)
cfg = dict(N=N, pool=POOL, subsets=SUBS, Lmax=LMAX, gammas=GAMMAS, channels=CHS, window=WINDOW, ref_L=8192, embeddings=["ZZ", "R"])

def make_layers(emb, seed):
    rng = np.random.default_rng(seed)
    if emb == "ZZ":
        X = rng.uniform(0, np.pi, (POOL, N))
        return zz_layers(X, N, "full")
    X = rng.uniform(0, 1, (POOL, N))
    return emb_R_layers(X, 2.0, N, 8)

def noise_for(ch, g):
    return ("global", g) if ch == "gdep" else ("local", kraus(ch, g))

def one(emb, g, ch):
    Vs = make_layers(emb, 100)
    noise = noise_for(ch, g)
    Ks = noisy_kernels(Vs, noise, N, Ls)
    Nsup = noise_superop(noise, N)
    Kref = dense_kernel_limit(Vs, Nsup, 8192)
    rf, rb, wf, wb = [], [], [], []
    for V in Vs:
        S, Sp = dense_layer_maps(V, Nsup)
        rf.append(arm_radius(S)); rb.append(arm_radius(Sp.T))
        wf.append(arm_radius_weighted(S)); wb.append(arm_radius_weighted(Sp.T))
    rf, rb, wf, wb = np.array(rf), np.array(rb), np.array(wf), np.array(wb)
    rng = np.random.default_rng(7)
    unital = ch != "ad"
    k = 2 if unital else 1
    win = WINDOW[g]
    slopes, preds_max, preds_pair, cs, ses, slopes2, preds_w = [], [], [], [], [], [], []
    for s in range(SUBS):
        sub = np.sort(rng.choice(POOL, 10, replace=False))
        d = np.array([np.abs(Ks[L][np.ix_(sub, sub)] - Kref[np.ix_(sub, sub)]).max() for L in Ls])
        sl, ic, m, c, se = fit_rate(Ls, d, win, floor=1e-13)
        sl2 = fit_rate(Ls, d, WINDOW2, floor=1e-13)[0]
        slopes2.append(sl2)
        rho = max(rf[sub].max(), rb[sub].max())
        # pairwise: slowest pair (i,j): unital -> ln rf_i + ln rb_j ; non-unital -> single-exponent max
        if unital:
            pp = np.log(rf[sub].max()) + np.log(rb[sub].max())
        else:
            pp = np.log(rho)
        rhow = max(wf[sub].max(), wb[sub].max()); preds_w.append(k * np.log(rhow))
        slopes.append(sl); preds_max.append(k * np.log(rho)); preds_pair.append(pp); cs.append(c); ses.append(se)
    slopes = np.array(slopes, float)
    ok = ~np.isnan(slopes)
    obs = float(np.mean(slopes[ok])) if ok.any() else np.nan
    se_sub = float(np.std(slopes[ok], ddof=1) / np.sqrt(ok.sum())) if ok.sum() > 1 else np.nan
    se_reg = float(np.nanmean(ses)) if ok.any() else np.nan
    se = float(np.nanmax([se_sub, se_reg]))
    g_ = g
    old = dict(gdep=2 * np.log(1 - g_), dep=2 * np.log(1 - g_), ad=0.5 * np.log(1 - g_), pd=0.5 * np.log(1 - g_), bf=np.log(1 - 2 * g_))
    u1 = unit_stats(kraus("dep" if ch == "gdep" else ch, g_))["u"]
    return dict(emb=emb, gamma=g, ch=ch, window=win, obs_slope_late=float(np.nanmean(slopes2)) if np.any(~np.isnan(np.array(slopes2,float))) else np.nan, obs_slope=obs, se=se, se_subsets=se_sub, se_reg=se_reg,
                pred_spectral=float(np.mean(preds_max)), pred_weighted=float(np.mean(preds_w)), pred_pair=float(np.mean(preds_pair)),
                pred_old_mu=old[ch], pred_u1=float(np.log(u1)), pred_uN=float(np.log(u_N_numeric(kraus("dep" if ch == "gdep" else ch, g_), N))) ,
                rho_f_max=float(rf.max()), rho_b_max=float(rb.max()), prefactor_c=float(np.nanmean(cs)),
                delta_at_win_start=float(np.abs(Ks[win[0]] - Kref).max()), n_valid=int(ok.sum()))

jobs = [(e, g, c) for e in ("ZZ", "R") for g in GAMMAS for c in CHS]
out = Parallel(n_jobs=10)(delayed(one)(*j) for j in jobs)
# acceptance
for r in out:
    o, p, se = r["obs_slope"], r["pred_spectral"], r["se"]
    if np.isnan(o):
        r["accept_i_leq_pred"] = None; r["accept_ii_AD_tight"] = None; continue
    r["z_vs_spectral"] = float((o - p) / se) if se and se > 0 else None
    r["accept_i_leq_pred"] = bool(o <= p + 2 * max(se, 1e-4))
    pw = r["pred_weighted"]
    r["accept_i_weighted"] = bool(o <= pw + 2 * max(se, 1e-4))
    r["accept_ii_AD_tight_weighted"] = bool(abs(o - pw) < 2 * max(se, 1e-3)) if r["ch"] == "ad" else None
    r["rel_gap_weighted"] = float((o - pw) / abs(pw)) if pw != 0 else None
    r["accept_ii_AD_tight"] = bool(abs(o - p) < 2 * max(se, 1e-3)) if r["ch"] == "ad" else None
    r["old_mu_rejected"] = bool(abs(o - r["pred_old_mu"]) > 2 * max(se, 1e-3))
    r["u1_rejected"] = bool(abs(o - r["pred_u1"]) > 2 * max(se, 1e-3))
save("E1_rates", cfg, dict(rows=out), t0)
import pandas as pd
df = pd.DataFrame(out)
pd.set_option("display.width", 250)
print(df[["emb", "gamma", "ch", "obs_slope", "se", "pred_spectral", "pred_weighted", "accept_i_weighted", "accept_ii_AD_tight_weighted", "obs_slope_late", "pred_old_mu", "pred_u1", "prefactor_c", "accept_i_leq_pred", "accept_ii_AD_tight"]].round(4).to_string())
