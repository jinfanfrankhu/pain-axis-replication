"""Headless runner for scripts/4.2_steering/01_steering_ladder.py.

Loads the original script unmodified, points it at our pain vectors and output
folder, and answers its interactive layer prompt with the layer the authors used
for that model (read from the layer in their shipped CSV file names), so the
runs are comparable with author-results/4.2_steering.

Run from the repo root:
    python replication/tools/run_steering.py S2 MODEL_NAME [MODEL_NAME ...]
MODEL_NAME uses the authors' names (e.g. Llama_3.1_8B_instruct). Use S1 or S2.
"""
import re
import sys
import builtins
import importlib.util
from pathlib import Path

TAG = sys.argv[1]
NAMES = sys.argv[2:]
assert TAG in ("S1", "S2") and NAMES

AUTHOR_DIR = Path("author-results/4.2_steering") / TAG
AUTHOR_LAYERS = {}
for f in AUTHOR_DIR.glob(f"*_steering_{TAG}_neutral50_L*.csv"):
    m = re.match(rf"(.+)_steering_{TAG}_neutral50_L(\d+)\.csv", f.name)
    AUTHOR_LAYERS[m.group(1)] = int(m.group(2))

spec = importlib.util.spec_from_file_location("steer", "scripts/4.2_steering/01_steering_ladder.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)

s.RESULTS_DIR = Path(__import__("os").environ.get("VEC_DIR", "replication/results/3.2_pain_vectors"))
s.OUT_DIR = Path("replication/results/4.2_steering") / TAG
s.OUT_DIR.mkdir(parents=True, exist_ok=True)
s.MIN_FREE_GB = 40  # 7-9B models only

repos = {name: repo for repo, name in s.RUN_MODELS}
token = __import__("os").environ.get("HF_TOKEN")
if token:
    s.login(token=token)

current = {}
builtins.input = lambda prompt="": (print(prompt + str(current["layer"]) + "  [authors' layer]"), str(current["layer"]))[1]

failed = []
for name in NAMES:
    current["layer"] = AUTHOR_LAYERS[name]
    print(f"\n=== {name} ({TAG}), authors' layer {current['layer']}", flush=True)
    if s.model_complete(name, TAG):
        print("already complete, skipping", flush=True)
        continue
    try:
        s.run_model(repos[name], name, f"{TAG.lower()}_pain_vector", TAG)
        print(f"SUCCESS: {name}", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {name}", flush=True)
        failed.append(name)
    finally:
        s.clear_hf_cache()
print(f"STEERING DONE. Failed: {failed or 'none'}", flush=True)
