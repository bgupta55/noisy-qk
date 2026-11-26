"""Y derived numbers (POST-HOC, labelled; not used for any verdict): Hahn-echo T2 from the first three delays only, because E(120 us) and E(240 us) are non-monotone on several qubits."""
from common import *

t0 = time.time()
d = json.load(open(os.path.join(RES, "Y2_Y7a_marrakesh.json")))["results"]
taus = np.array(d["echo"]["delays_us"], float)
out = []
rng = np.random.default_rng(2031)
for r in d["echo"]["rows"]:
    E, se = np.array(r["E"]), np.array(r["E_se"])
    m = taus <= 60
    f = lambda e: -1.0 / np.polyfit(taus[m], np.log(e[m]), 1)[0]
    t2 = float(f(E))
    bs = [f(E + rng.normal(0, se)) for _ in range(400)]
    out.append(dict(qubit=r["qubit"], T2_first3_us=t2, T2_first3_se=float(np.std(bs)), T2_cal_us=r["T2_cal_us"], T2_echo_prereg_us=r["T2_echo_us"],
                    nonmonotone_after_120=bool(E[4] > E[3] + 2 * np.hypot(se[3], se[4])), E_120=float(E[3]), E_240=float(E[4])))
    print(out[-1])

# ---- Y3 derived (POST-HOC): predicted dL_emp/dlnS = alpha_pred / rate for three rates; local rate of a_L at the crossing
y3 = json.load(open(os.path.join(RES, "Y3_S_exponent.json")))["results"]
y3d = {}
for ch in ("dep", "ad"):
    rws = [y3["rows"][f"{d}_{ch}"] for d in ("D1", "D2", "D3", "D4")]
    obs = float(np.mean([r["dLemp_dlnS"] for r in rws]))
    rv1, rf3 = float(np.mean([r["rate_v1"] for r in rws])), float(np.mean([r["rate_fit3"] for r in rws]))
    loc = float(np.mean([np.mean(r["local_rate_at_crossing"]) for r in rws]))
    y3d[ch] = dict(observed_dL_per_lnS=obs, rate_v1=rv1, rate_fit3=rf3, rate_local_at_crossing=loc,
                   pred_half_over_rate_v1=0.5 / rv1, pred_half_over_rate_fit3=0.5 / rf3, pred_half_over_rate_local=0.5 / loc,
                   measured_alpha=y3["pooled"][ch]["alpha_mean"], alpha_implied_by_obs_and_fit3_rate=obs * rf3, alpha_implied_by_obs_and_local_rate=obs * loc)
    print("Y3", ch, {k: round(v, 3) for k, v in y3d[ch].items()})
save("Y_derived", dict(note="POST-HOC; not used for any verdict"), dict(echo_first3=out, y3_prediction_by_rate=y3d), t0)
