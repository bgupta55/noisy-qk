"""W1 supplement (no QPU): (i) PRESPEC section 5 sensitivity: earlier X4 chain-A retention restricted to ordered pairs (i<j, train-train) among the six chosen inputs, compared with today's
chain A on the same unordered pairs / same orientation; (ii) earlier Y2 echo and idle numbers next to today's; (iii) QPU bookkeeping. Writes results/W1_supplement.json."""
from common import *

t0 = time.time()
HW = os.path.join(RES, "hardware_w1")
meta = json.load(open(os.path.join(HW, "prep_meta.json")))
W = json.load(open(os.path.join(RES, "W1_marrakesh_chains.json")))["results"]
x4m = json.load(open(os.path.join(RES, "hardware_x4", "prep_meta.json")))
raws = [json.load(open(os.path.join(RES, "hardware_x4", f"raw_x4_{k}.json"))) for k in range(3)]
cnt = np.array([c.get("0000", 0) for g in raws for c in g["counts"]], float)
settings = [tuple(s) for s in x4m["settings"]]; pairs = [tuple(p) for p in x4m["pairs"]]; NP = len(pairs)
sh = np.concatenate([[x4m["shots"][f"{a},{b}"]] * NP for a, b in settings]).astype(float)
ex = np.array(x4m["aer"]["exact"])
lay_of = {2 * L * (1 + 2 * f): k for k, (L, f) in enumerate(settings)}
six = meta["six_inputs"]
LAYS = [2, 6, 12, 20, 28, 36]
rng = np.random.default_rng(3103)


def ols(K, p):
    X = np.column_stack([K, np.ones_like(K)]); return float(np.linalg.lstsq(X, p, rcond=None)[0][0])


sel = [n for n, (i, j) in enumerate(pairs) if i in six and j in six]
raw_A = json.load(open(os.path.join(HW, "raw_A-R.json")))
M = meta["jobs"]["A-R"]["items"]
today = {}
for it, c, s in zip(M, raw_A["counts"], raw_A["shots"]):
    today[(it["layers"], six[it["i"]], six[it["j"]])] = c.get("0000", 0) / s
Kx = {L: np.array(meta["Kexact"][str(L)]) for L in LAYS}
out = dict(n_pairs_restricted=len(sel), pairs=[pairs[n] for n in sel], rows={})
for L in LAYS:
    k = lay_of[L]; idx = np.array([NP * k + n for n in sel])
    p_e, K_e = cnt[idx] / sh[idx], ex[idx]
    p_t = np.array([today[(L, i, j)] for (i, j) in [pairs[n] for n in sel]])
    # today, same oriented pairs
    s_e, s_t = ols(K_e, p_e), ols(K_e, p_t)
    bs = rng.integers(0, len(sel), (2000, len(sel)))
    se_e = float(np.std([ols(K_e[b], p_e[b]) for b in bs])); se_t = float(np.std([ols(K_e[b], p_t[b]) for b in bs]))
    # paired shot-noise z on mean p
    dm = float(np.mean(p_t - p_e)); sem = float(np.sqrt(np.sum(p_t * (1 - p_t) / 1024 + p_e * (1 - p_e) / sh[idx])) / len(sel))
    out["rows"][L] = dict(s_earlier=s_e, se_earlier=se_e, s_today=s_t, se_today=se_t, z_s=(s_t - s_e) / np.hypot(se_e, se_t), mean_p_earlier=float(p_e.mean()), mean_p_today=float(p_t.mean()),
                          mean_p_diff=dm, mean_p_diff_se=sem, mean_p_z=dm / sem)
    print("L", L, "n=%d" % len(sel), "s earlier %.3f today %.3f z %.2f | mean p earlier %.3f today %.3f z %.2f" % (s_e, s_t, out["rows"][L]["z_s"], p_e.mean(), p_t.mean(), dm / sem))

# earlier Y2 (chain A, 26-nov? no: 23-24 Nov) idle + echo
Y = json.load(open(os.path.join(RES, "Y2_Y7a_marrakesh.json")))["results"]
out["earlier_Y2"] = dict(mean_abs_tau100=Y["magnitude"].get("mean_abs_tau100"), mean_ratio_hw_over_pred_cal=Y["magnitude"]["mean_ratio_hw_over_pred_cal"],
                         mean_z_tau200=Y["gates"]["iii_z"]["mean_z_tau200"], mean_abs_by_tau=Y["gates"]["ii_monotone_and_sign"]["mean_abs_by_tau"],
                         echo=[dict(qubit=r["qubit"], T2e=r["T2_echo_us"], T2cal=r["T2_cal_us"]) for r in Y["echo"]["rows"]], utc=Y["utc"], job=Y["job"])
out["calibration_change_A"] = {str(q): dict(earlier=Y["calibration"]["qubit_props"][str(q)] if str(q) in Y["calibration"]["qubit_props"] else None,
                                            today=meta["snapshot_start"]["A"]["qubit_props"][str(q)] if str(q) in meta["snapshot_start"]["A"]["qubit_props"] else None) for q in meta["A"]}
try:
    end = json.load(open(os.path.join(HW, "snapshot_end.json")))
    out["snapshot_end"] = end
except Exception as e:
    out["snapshot_end"] = str(e)
out["today_mean_p"] = {ch: W["retention"][ch]["mean_p"] for ch in ("A", "B")}
out["qpu"] = {n: dict(job_id=v["job_id"], usage_seconds=v["usage_seconds"], account_delta=v["account_after"] - v["account_before"], wall=v["wall"], utc=v["utc"]) for n, v in W["jobs"].items()}
print(json.dumps(out["earlier_Y2"], indent=1, default=float))
print(json.dumps(out["calibration_change_A"], indent=1, default=float))
print(json.dumps(out["qpu"], indent=1, default=float))
save("W1_supplement", dict(source="results/hardware_w1", seed=3103), out, t0)
