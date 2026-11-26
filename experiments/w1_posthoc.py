"""W1 POST-HOC diagnostics (not pre-specified, used for no verdict): direct change of s between layers 20 and 36 (and 28), per chain, entry bootstrap; chain-wise mean p; echo mean T2. Writes results/W1_posthoc.json."""
from common import *
t0 = time.time()
HW = os.path.join(RES, "hardware_w1"); meta = json.load(open(os.path.join(HW, "prep_meta.json")))
LAYS = [2, 6, 12, 20, 28, 36]; Kx = {L: np.array(meta["Kexact"][str(L)]).ravel() for L in LAYS}
rng = np.random.default_rng(3104); bi = rng.integers(0, 36, (4000, 36))
def ols(K, p):
    X = np.column_stack([K, np.ones_like(K)]); return float(np.linalg.lstsq(X, p, rcond=None)[0][0])
out = {}
for ch in ("A", "B"):
    raw = json.load(open(os.path.join(HW, f"raw_{ch}-R.json"))); M = meta["jobs"][f"{ch}-R"]["items"]
    P = {L: np.zeros(36) for L in LAYS}
    for it, c, s in zip(M, raw["counts"], raw["shots"]): P[it["layers"]][6 * it["i"] + it["j"]] = c.get("0000", 0) / s
    d = {}
    for a, b in ((20, 36), (28, 36), (20, 28)):
        pt = ols(Kx[b], P[b]) - ols(Kx[a], P[a])
        bs = [ols(Kx[b][ix], P[b][ix]) - ols(Kx[a][ix], P[a][ix]) for ix in bi]
        d[f"s{b}-s{a}"] = dict(diff=pt, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    out[ch] = d
    # mean p on diagonal vs off-diagonal at 36
    dg = np.array([6 * i + i for i in range(6)])
    out[ch]["diag_mean_p"] = {L: float(P[L][dg].mean()) for L in LAYS}
    out[ch]["offdiag_mean_p"] = {L: float(np.delete(P[L], dg).mean()) for L in LAYS}
    print(ch, d, out[ch]["diag_mean_p"], out[ch]["offdiag_mean_p"])
save("W1_posthoc", dict(note="POST-HOC, no verdict", seed=3104), out, t0)
