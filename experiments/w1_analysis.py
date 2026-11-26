"""W1 offline analysis (no QPU). Implements PRESPEC_marrakesh_chains.md sections 5-6. Reads results/hardware_w1/{prep_meta.json, raw_*.json}, results/hardware_x4 (earlier chain A).
Writes results/W1_marrakesh_chains.json."""
from common import *

t0 = time.time()
HW = os.path.join(RES, "hardware_w1")
meta = json.load(open(os.path.join(HW, "prep_meta.json")))
LAYS = [2, 6, 12, 20, 28, 36]
NQ = 4
B_BOOT = 2000
rng = np.random.default_rng(3101)


def load(name):
    fn = os.path.join(HW, f"raw_{name}.json")
    return json.load(open(fn)) if os.path.exists(fn) else None


raw = {n: load(n) for n in meta["job_order"]}
done = {n: r is not None for n, r in raw.items()}
print("jobs done:", done)


def grid(job, key="counts"):
    """P[L][i][j], S from raw counts of an R job"""
    M = meta["jobs"][job]["items"]
    P = {L: np.full((6, 6), np.nan) for L in LAYS}
    S = {L: np.zeros((6, 6)) for L in LAYS}
    for it, c, s in zip(M, raw[job]["counts"], raw[job]["shots"]):
        P[it["layers"]][it["i"], it["j"]] = c.get("0000", 0) / s
        S[it["layers"]][it["i"], it["j"]] = s
    return P, S


def ols_s(K, p):
    X = np.column_stack([K, np.ones_like(K)])
    b = np.linalg.lstsq(X, p, rcond=None)[0]
    return float(b[0]), float(b[1])


def misfit(K, p, S):
    X = np.column_stack([K, np.ones_like(K)])
    b = np.linalg.lstsq(X, p, rcond=None)[0]
    res = p - X @ b
    shot = np.sqrt(np.mean(p * (1 - p) / S))
    return float(np.sqrt((res ** 2).sum() / (len(p) - 2)) / shot)


def slopes(s, kind):
    L = np.array(LAYS, float)
    s = np.asarray(s)
    e = np.isin(L, [2, 6, 12, 20]); l = np.isin(L, [20, 28, 36])
    es, ls = np.polyfit(L[e], s[e], 1)[0], np.polyfit(L[l], s[l], 1)[0]
    return float(es), float(ls), float(ls - es)


Kx = {L: np.array(meta["Kexact"][str(L)]) for L in LAYS}


def s_all(P, idx=None):
    out = []
    for L in LAYS:
        K, p = Kx[L].ravel(), P[L].ravel()
        if idx is not None:
            K, p = K[idx], p[idx]
        out.append(ols_s(K, p)[0])
    return np.array(out)


def design(model, K, ii, jj):
    cols = [K[:, None], np.ones((len(K), 1))]
    if model == "M1":
        cols.append(np.eye(6)[jj])
    if model == "M2":
        cols.append(np.eye(6)[ii])
    return np.hstack(cols)


def r2(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    rss = float(((y - X @ b) ** 2).sum()); tss = float(((y - y.mean()) ** 2).sum())
    n, r = len(y), int(np.linalg.matrix_rank(X))
    return 1 - rss / tss, (1 - (rss / (n - r)) / (tss / (n - 1)) if n - r > 0 else None), b, rss, r


II, JJ = np.meshgrid(range(6), range(6), indexing="ij"); II, JJ = II.ravel(), JJ.ravel()


def decomp(P, S, L):
    K, p, s = Kx[L].ravel(), P[L].ravel(), S[L].ravel()
    out = {}
    for m in ("M0", "M1", "M2"):
        X = design(m, K, II, JJ)
        R2, aR2, b, rss, rk = r2(p, X)
        shot = np.sqrt(np.mean(p * (1 - p) / s))
        out[m] = dict(R2=R2, adjR2=aR2, misfit=float(np.sqrt(rss / (len(p) - rk)) / shot) if len(p) - rk > 0 else None, rank=rk)
        if m == "M1":
            bj = b[2:]; out["b_j"] = (bj - bj.mean()).tolist()
        if m == "M2":
            ci = b[2:]; out["c_i"] = (ci - ci.mean()).tolist()
    gb, gc = out["M1"]["R2"] - out["M0"]["R2"], out["M2"]["R2"] - out["M0"]["R2"]
    out.update(gain_b=gb, gain_c=gc, ratio=gb / max(gc, 1e-3))
    # asymmetry P_ij - P_ji, i<j
    a, se = [], []
    for i in range(6):
        for j in range(i + 1, 6):
            p1, p2 = P[L][i, j], P[L][j, i]
            a.append(p1 - p2); se.append(np.sqrt(p1 * (1 - p1) / S[L][i, j] + p2 * (1 - p2) / S[L][j, i]))
    a, se = np.array(a), np.array(se)
    out["asym"] = dict(mean_abs=float(np.abs(a).mean()), mean_z=float(np.mean(np.abs(a) / se)), n_gt3se=int((np.abs(a) > 3 * se).sum()), n=len(a),
                       exact_mean_abs=float(np.mean([abs(Kx[L][i, j] - Kx[L][j, i]) for i in range(6) for j in range(i + 1, 6)])))
    return out


res = dict(done=done)
grids = {}
for ch in ("A", "B"):
    if done[f"{ch}-R"]:
        grids[ch] = grid(f"{ch}-R")
# Aer noisy predictions (calibration at the start of the session)
for ch in ("A", "B"):
    M = meta["jobs"][f"{ch}-R"]["items"]; pa = meta["jobs"][f"{ch}-R"]["aer_noisy_2048"]
    P = {L: np.full((6, 6), np.nan) for L in LAYS}
    for it, v in zip(M, pa):
        P[it["layers"]][it["i"], it["j"]] = v
    grids["aer_" + ch] = (P, {L: np.full((6, 6), 2048.0) for L in LAYS})

# ------------------------------------------------------------------ retention, rise
boot_idx = rng.integers(0, 36, (B_BOOT, 36))
ret = {}
for ch, (P, S) in grids.items():
    s = s_all(P)
    bs = np.array([s_all(P, ix) for ix in boot_idx])
    sl = np.array([slopes(b, None) for b in bs])
    es, ls, rise = slopes(s, None)
    ret[ch] = dict(s=s.tolist(), s_se=bs.std(0).tolist(), early=es, late=ls, rise=rise, rise_ci=[float(np.percentile(sl[:, 2], 2.5)), float(np.percentile(sl[:, 2], 97.5))],
                   early_ci=[float(np.percentile(sl[:, 0], 2.5)), float(np.percentile(sl[:, 0], 97.5))], late_ci=[float(np.percentile(sl[:, 1], 2.5)), float(np.percentile(sl[:, 1], 97.5))],
                   mean_p=[float(np.nanmean(P[L])) for L in LAYS],
                   misfit=[misfit(Kx[L].ravel(), P[L].ravel(), S[L].ravel()) for L in LAYS], boot=bs)
    print(ch, "s", np.round(s, 3), "se", np.round(bs.std(0), 3), "rise %.5f" % rise, [round(x, 5) for x in ret[ch]["rise_ci"]], "misfit", np.round(ret[ch]["misfit"], 1))
res["retention"] = {ch: {k: v for k, v in d.items() if k != "boot"} for ch, d in ret.items()}

# ------------------------------------------------------------------ D2: paired bootstrap s(B)-s(A) at every layer
if "A" in ret and "B" in ret:
    d = ret["B"]["boot"] - ret["A"]["boot"]
    dd = {}
    for k, L in enumerate(LAYS):
        pt = ret["B"]["s"][k] - ret["A"]["s"][k]
        dd[L] = dict(diff=pt, ci=[float(np.percentile(d[:, k], 2.5)), float(np.percentile(d[:, k], 97.5))])
    res["D2_diff_B_minus_A"] = dd
    print("B-A", {L: (round(v["diff"], 3), [round(x, 3) for x in v["ci"]]) for L, v in dd.items()})
    D2 = bool(dd[36]["ci"][0] > 0 or dd[36]["ci"][1] < 0)
else:
    D2 = None

# ------------------------------------------------------------------ per-input decomposition (Theorem 5)
dec = {}
for ch in ("A", "B", "aer_A", "aer_B"):
    if ch in grids:
        P, S = grids[ch]
        dec[ch] = {L: decomp(P, S, L) for L in (28, 36)}
        for L in (28, 36):
            d_ = dec[ch][L]
            print(ch, L, "R2 M0 %.3f M1 %.3f M2 %.3f gain_b %.3f gain_c %.3f ratio %.2f | misfit M0 %.1f M1 %.1f M2 %.1f | asym mean|.| %.3f z %.1f (exact %.3f)" % (
                d_["M0"]["R2"], d_["M1"]["R2"], d_["M2"]["R2"], d_["gain_b"], d_["gain_c"], d_["ratio"], d_["M0"]["misfit"], d_["M1"]["misfit"] or np.nan, d_["M2"]["misfit"] or np.nan,
                d_["asym"]["mean_abs"], d_["asym"]["mean_z"], d_["asym"]["exact_mean_abs"]))
res["decomposition"] = dec
cross = {}
if "A" in dec and "B" in dec:
    for L in (28, 36):
        cross[L] = float(np.corrcoef(dec["A"][L]["b_j"], dec["B"][L]["b_j"])[0, 1])
    cross["c_i"] = {L: float(np.corrcoef(dec["A"][L]["c_i"], dec["B"][L]["c_i"])[0, 1]) for L in (28, 36)}
res["cross_chain_corr_bj"] = cross
print("cross-chain corr b_j", cross)

# ------------------------------------------------------------------ D1: stability
d1 = {}
if done["A-R"]:
    x4m = json.load(open(os.path.join(RES, "hardware_x4", "prep_meta.json")))
    raws = [json.load(open(os.path.join(RES, "hardware_x4", f"raw_x4_{k}.json"))) for k in range(3)]
    cnt = np.array([c.get("0000", 0) for g in raws for c in g["counts"]], float)
    settings = [tuple(s) for s in x4m["settings"]]; NP = len(x4m["pairs"])
    sh = np.concatenate([[x4m["shots"][f"{a},{b}"]] * NP for a, b in settings]).astype(float)
    ex = np.array(x4m["aer"]["exact"])
    lay_of = {2 * L * (1 + 2 * f): k for k, (L, f) in enumerate(settings)}
    rng2 = np.random.default_rng(3102)
    earlier = {}
    for L in LAYS:
        k = lay_of[L]; sl = slice(NP * k, NP * (k + 1))
        p, K = cnt[sl] / sh[sl], ex[sl]
        s0 = ols_s(K, p)[0]
        bs = [ols_s(K[ix], p[ix])[0] for ix in rng2.integers(0, NP, (B_BOOT, NP))]
        earlier[L] = dict(s=s0, se=float(np.std(bs)))
    res["earlier_A_x4"] = earlier
    P, S = grids["A"]
    for L in (2, 12, 36):
        t = ret["A"]["s"][LAYS.index(L)]; se = ret["A"]["s_se"][LAYS.index(L)]
        comb = np.sqrt(se ** 2 + earlier[L]["se"] ** 2)
        d1[f"earlier_vs_today_L{L}"] = dict(today=t, today_se=se, earlier=earlier[L]["s"], earlier_se=earlier[L]["se"], diff=t - earlier[L]["s"], combined_se=float(comb), z=float((t - earlier[L]["s"]) / comb))
    print("earlier A", {L: (round(v["s"], 3), round(v["se"], 3)) for L, v in earlier.items()})
    if done["A-Rrep"]:
        Mr = meta["jobs"]["A-Rrep"]["items"]
        for L in (2, 12, 36):
            ent = [(it["i"], it["j"], c.get("0000", 0) / s) for it, c, s in zip(Mr, raw["A-Rrep"]["counts"], raw["A-Rrep"]["shots"]) if it["layers"] == L]
            K = np.array([Kx[L][i, j] for i, j, _ in ent]); pr = np.array([v for _, _, v in ent]); pt = np.array([P[L][i, j] for i, j, _ in ent])
            s_rep, o_rep = ols_s(K, pr); s_tod, o_tod = ols_s(K, pt)
            bsr = rng2.integers(0, len(K), (B_BOOT, len(K)))
            se_r = float(np.std([ols_s(K[ix], pr[ix])[0] for ix in bsr])); se_t = float(np.std([ols_s(K[ix], pt[ix])[0] for ix in bsr]))
            # same-circuit comparison: also the mean difference of p over the 12 entries, shot-noise based
            dmean = float(np.mean(pr - pt)); se_m = float(np.sqrt(np.sum(pr * (1 - pr) / 1024 + pt * (1 - pt) / 1024)) / len(pr))
            comb = np.sqrt(se_r ** 2 + se_t ** 2)
            d1[f"repeat_vs_today_L{L}"] = dict(s_repeat=s_rep, s_today_same12=s_tod, se_repeat=se_r, se_today=se_t, diff=s_rep - s_tod, combined_se=float(comb), z=float((s_rep - s_tod) / comb),
                                               mean_p_diff=dmean, mean_p_diff_se_shot=se_m, mean_p_z=dmean / se_m)
            print("repeat L", L, "s_rep %.3f s_today(12) %.3f z %.2f | mean p diff %.4f +- %.4f (z %.2f)" % (s_rep, s_tod, d1[f"repeat_vs_today_L{L}"]["z"], dmean, se_m, dmean / se_m))
res["D1_detail"] = d1

# ------------------------------------------------------------------ idle asymmetry A
Y2 = json.load(open(os.path.join(RES, "hardware_y2", "prep_meta.json")))
idle = {}
for ch in ("A", "B"):
    if not done[f"{ch}-A"]:
        continue
    M = meta["jobs"][f"{ch}-A"]["items"]; cts = raw[f"{ch}-A"]["counts"]; shs = raw[f"{ch}-A"]["shots"]; aer = meta["jobs"][f"{ch}-A"]["aer_noisy_2048"]
    p = np.array([c.get("0000", 0) / s for c, s in zip(cts, shs)])
    rows = []
    for (i, j) in [tuple(q) for q in meta["pairs3"]]:
        for tau in meta["delays"]:
            k1 = next(k for k, m in enumerate(M) if m["tau"] == tau and m["i"] == i and m["j"] == j)
            k2 = next(k for k, m in enumerate(M) if m["tau"] == tau and m["i"] == j and m["j"] == i)
            se = float(np.sqrt(p[k1] * (1 - p[k1]) / shs[k1] + p[k2] * (1 - p[k2]) / shs[k2]))
            rows.append(dict(pair=[i, j], tau=tau, K_ij=float(p[k1]), K_ji=float(p[k2]), asym_hw=float(p[k1] - p[k2]), se=se, asym_pred=float(M[k1]["K_idle_cal"] - M[k2]["K_idle_cal"]),
                             asym_aer=float(aer[k1] - aer[k2]), asym_exact=float(M[k1]["K_exact"] - M[k2]["K_exact"])))
    pos = [r for r in rows if r["tau"] > 0]
    rt = [r["asym_hw"] / r["asym_pred"] for r in pos if abs(r["asym_pred"]) > 1e-3]
    hw = np.array([r["asym_hw"] for r in pos]); pr = np.array([r["asym_pred"] for r in pos])
    t200 = [r for r in rows if r["tau"] == 200]
    idle[ch] = dict(rows=rows, n_sign_match=int(sum(np.sign(r["asym_hw"]) == np.sign(r["asym_pred"]) for r in pos)), n=len(pos), mean_ratio=float(np.mean(rt)), n_ratio=len(rt),
                    regression_ratio=float((hw * pr).sum() / (pr * pr).sum()), mean_abs_asym_200=float(np.mean([abs(r["asym_hw"]) for r in t200])),
                    mean_abs_pred_200=float(np.mean([abs(r["asym_pred"]) for r in t200])), mean_z_200=float(np.mean([abs(r["asym_hw"]) / r["se"] for r in t200])),
                    max_abs_asym_tau0_over_se=float(max(abs(r["asym_hw"]) / r["se"] for r in rows if r["tau"] == 0)),
                    mean_abs_by_tau={t: float(np.mean([abs(r["asym_hw"]) for r in rows if r["tau"] == t])) for t in meta["delays"]},
                    rms_z_vs_pred=float(np.sqrt(np.mean([((r["asym_hw"] - r["asym_pred"]) / r["se"]) ** 2 for r in pos]))),
                    rms_z_vs_aer=float(np.sqrt(np.mean([((r["asym_hw"] - r["asym_aer"]) / r["se"]) ** 2 for r in pos]))))
    print("idle", ch, "signs %d/%d" % (idle[ch]["n_sign_match"], idle[ch]["n"]), "mean ratio %.2f (n=%d) regression ratio %.2f" % (idle[ch]["mean_ratio"], idle[ch]["n_ratio"], idle[ch]["regression_ratio"]),
          "|asym|@200 hw %.3f pred %.3f z %.1f" % (idle[ch]["mean_abs_asym_200"], idle[ch]["mean_abs_pred_200"], idle[ch]["mean_z_200"]), "tau0 max z %.1f" % idle[ch]["max_abs_asym_tau0_over_se"])
    for r in rows:
        print("   ", r["pair"], r["tau"], "hw %.3f+-%.3f pred %.3f aer %.3f exact %.3f" % (r["asym_hw"], r["se"], r["asym_pred"], r["asym_aer"], r["asym_exact"]))
res["idle"] = idle

# ------------------------------------------------------------------ echo T
echo = {}
for ch in ("A", "B"):
    if not done[f"{ch}-T"]:
        continue
    M = meta["jobs"][f"{ch}-T"]["items"]; cts = raw[f"{ch}-T"]["counts"]; shs = raw[f"{ch}-T"]["shots"]
    lay = meta[ch]; sn = meta["snapshot_start"][ch]["qubit_props"]
    taus = sorted(set(m["tau"] for m in M)); tt = np.array(taus, float)
    rows = []
    for q in range(NQ):
        E, Es = [], []
        for t in taus:
            kp = next(k for k, m in enumerate(M) if m["tau"] == t and m["prep"] == "+"); km = next(k for k, m in enumerate(M) if m["tau"] == t and m["prep"] == "-")
            f = lambda c, s: sum(v for key, v in c.items() if key[::-1][q] == "1") / s
            a, b = f(cts[kp], shs[kp]), f(cts[km], shs[km])
            E.append(b - a); Es.append(np.sqrt(a * (1 - a) / shs[kp] + b * (1 - b) / shs[km]))
        E, Es = np.array(E), np.array(Es)
        ok = E > 2 * Es
        T2e = sd = None
        if ok.sum() >= 3:
            fit = lambda Ev: -1.0 / np.polyfit(tt[ok], np.log(Ev[ok]), 1)[0]
            T2e = float(fit(E)); bsl = []
            for _ in range(400):
                Eb = E + rng.normal(0, Es)
                if (Eb[ok] > 0).all():
                    bsl.append(fit(Eb))
            sd = float(np.std(bsl)) if len(bsl) > 20 else None
        qpq = sn[str(lay[q])] if str(lay[q]) in sn else sn[lay[q]]
        T2c = qpq["T2_us"]
        rows.append(dict(qubit=lay[q], T2_echo=T2e, T2_echo_se=sd, T2_cal=T2c, T1_cal=qpq["T1_us"], factor=(max(T2e / T2c, T2c / T2e) if T2e else None), n_pts=int(ok.sum()), E=E.tolist(), E_se=Es.tolist()))
        print("echo", ch, lay[q], "T2e", None if T2e is None else round(T2e, 1), "cal", round(T2c, 1), "factor", None if T2e is None else round(max(T2e / T2c, T2c / T2e), 2), "npts", int(ok.sum()))
    echo[ch] = dict(delays=taus, rows=rows, max_factor=max([r["factor"] for r in rows if r["factor"]] or [None]) if any(r["factor"] for r in rows) else None)
res["echo"] = echo

# ------------------------------------------------------------------ decision rules
dec_rules = {}
if d1:
    zs = [d1[k]["z"] for k in d1 if k.startswith("earlier")]
    zr = [d1[k]["z"] for k in d1 if k.startswith("repeat")]
    dec_rules["D1"] = dict(earlier_all_le_3=bool(all(abs(z) <= 3 for z in zs)) if zs else None, repeat_all_le_3=bool(all(abs(z) <= 3 for z in zr)) if zr else None, z_earlier=zs, z_repeat=zr)
    dec_rules["D1"]["pass"] = bool(dec_rules["D1"]["earlier_all_le_3"] and dec_rules["D1"]["repeat_all_le_3"]) if zr else None
dec_rules["D2"] = dict(pass_=D2, diff_36=res.get("D2_diff_B_minus_A", {}).get(36))
if "A" in ret and "B" in ret:
    dec_rules["D3"] = dict(A=ret["A"]["rise_ci"], B=ret["B"]["rise_ci"], pass_=bool(all((ret[c]["rise_ci"][0] > 0 or ret[c]["rise_ci"][1] < 0) for c in ("A", "B"))),
                           sign=[float(np.sign(ret[c]["rise"])) for c in ("A", "B")])
    dec_rules["D4"] = dict(ratios={c: {L: dec[c][L]["ratio"] for L in (28, 36)} for c in ("A", "B")}, pass_=bool(all(dec[c][L]["ratio"] >= 2 for c in ("A", "B") for L in (28, 36))))
    dec_rules["D5"] = dict(corr=cross, pass_=bool(all(cross[L] > 0.7 for L in (28, 36))))
if "B" in idle:
    sgn = idle["B"]["n_sign_match"] == idle["B"]["n"]; rr = idle["B"]["mean_ratio"]
    dec_rules["D6"] = dict(all_signs=sgn, mean_ratio=rr, pass_=bool(sgn and 0.6 <= rr <= 1.1))
res["decisions"] = dec_rules
print(json.dumps(dec_rules, indent=1, default=float))

# job bookkeeping
res["jobs"] = {n: (dict(job_id=r["job_id"], utc=r["utc"], usage_seconds=r["usage_seconds"], n_circuits=r["n_circuits"], wall=r["wall_seconds"], account_before=r["account_usage_before"], account_after=r["account_usage_after"]) if r else None) for n, r in raw.items()}
save("W1_marrakesh_chains", dict(source="results/hardware_w1", prespec="PRESPEC_marrakesh_chains.md", bootstrap=B_BOOT, seed=3101), res, t0)
