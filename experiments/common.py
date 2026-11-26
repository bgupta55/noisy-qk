import os, sys, json, time, platform, hashlib, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def provenance(config):
    import scipy, sklearn
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, machine=platform.platform(),
                config_hash=hashlib.sha1(json.dumps(_clean(config), sort_keys=True).encode()).hexdigest()[:12],
                date="2026-11-23")


def save(name, config, results, t0):
    out = dict(experiment=name, config=_clean(config), provenance=provenance(config),
               wall_time_s=round(time.time() - t0, 2), results=_clean(results))
    p = os.path.join(RES, name + ".json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", p, "in %.1fs" % (time.time() - t0))


def bootstrap_ci(x, reps=10000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (reps, len(x)))
    b = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
