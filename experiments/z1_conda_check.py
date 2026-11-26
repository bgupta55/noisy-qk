"""Z1: compares the four JSON files re-run inside the micromamba/conda-forge environment (results/z1_conda_outputs/) with the committed ones
(results/*.json; ignoring provenance, wall_time_s, date). Run with the conda python so that the imported versions are recorded. Pre-registered check (iv): max |diff| <= 1e-9."""
from common import *
import importlib.metadata as md
t0 = time.time()
pins = {}
for line in open(os.path.join(ROOT, "environment.yml")):
    s = line.strip().lstrip("- ").strip()
    for sep in ("==", "="):
        if sep in s and not s.startswith("#") and not s.startswith("name") and not s.startswith("python"):
            a, b = s.split(sep, 1); pins[a.strip()] = b.strip(); break
mod = {"numpy": "numpy", "scipy": "scipy", "scikit-learn": "scikit-learn", "pandas": "pandas", "matplotlib": "matplotlib", "joblib": "joblib",
       "pyarrow": "pyarrow", "pytest": "pytest", "qiskit": "qiskit", "qiskit-aer": "qiskit-aer", "qiskit-ibm-runtime": "qiskit-ibm-runtime"}
vers = {k: dict(pinned=pins.get(k), installed=md.version(v), equal=(pins.get(k) == md.version(v))) for k, v in mod.items()}
def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if k not in ("provenance", "wall_time_s", "date")}
    if isinstance(o, list): return [strip(v) for v in o]
    return o
def md_(a, b):
    if isinstance(a, dict):
        r = [md_(a[k], b[k]) for k in a]; return (max([x[0] for x in r] or [0.0]), sum(x[1] for x in r))
    if isinstance(a, list):
        if len(a) != len(b): return (float("inf"), 1)
        r = [md_(u, v) for u, v in zip(a, b)]; return (max([x[0] for x in r] or [0.0]), sum(x[1] for x in r))
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)): return (abs(a - b), 1)
    return (0.0 if a == b else float("inf"), 1)
cmp = {}
for f in ("Y1_arm_regression", "Y3_S_exponent", "Y4_split_overlap", "X1_E6_fitted_rho"):
    a = json.load(open(os.path.join(RES, f + ".json"))); b = json.load(open(os.path.join(RES, "z1_conda_outputs", f + ".json")))
    m, n = md_(strip(a["results"]), strip(b["results"]))
    cmp[f] = dict(max_abs_diff=m, n_compared=n, committed_python=a["provenance"]["python"], conda_python=b["provenance"]["python"], ok=bool(m <= 1e-9))
save("Z1_conda_check", dict(env="micromamba 2.10.0, environment.yml unchanged, conda-forge + pip block", tol=1e-9),
     dict(versions=vers, all_pins_equal=all(v["equal"] for v in vers.values()), python_conda=platform.python_version(), comparisons=cmp, all_ok=all(c["ok"] for c in cmp.values())), t0)
print(json.dumps(dict(all_pins_equal=all(v["equal"] for v in vers.values()), cmp=cmp), indent=1))
