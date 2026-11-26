"""X7: deposit manifest. SHA-256 and size of every file that goes into the archive (code, tests, results JSON/CSV, raw hardware counts, figures,
reference material). Writes DEPOSIT_MANIFEST.json at the project root. A DOI is NOT created here: it needs an upload to Zenodo from the author's account."""
import os, json, hashlib, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INCL = ["src", "experiments", "tests", "results", "figures", "reference", "run_all.sh", "environment.yml"]
SKIP = ("__pycache__", ".pytest_cache", ".pyc", ".DS_Store")
out = []
for top in INCL:
    p = os.path.join(ROOT, top)
    files = [p] if os.path.isfile(p) else [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs]
    for f in sorted(files):
        if any(s in f for s in SKIP):
            continue
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        kind = ("raw_hardware_counts" if "/hardware" in f and os.path.basename(f).startswith("raw_") else
                "transpiled_circuits" if f.endswith("isa_circuits.pkl") else top)
        out.append(dict(path=os.path.relpath(f, ROOT), bytes=os.path.getsize(f), sha256=h, kind=kind))
man = dict(created="2026-11-23", n_files=len(out), total_bytes=sum(o["bytes"] for o in out), doi=None,
           doi_note="not minted: requires a Zenodo upload by the author", files=out)
json.dump(man, open(os.path.join(ROOT, "DEPOSIT_MANIFEST.json"), "w"), indent=1)
print(man["n_files"], "files", round(man["total_bytes"] / 1e6, 1), "MB")
