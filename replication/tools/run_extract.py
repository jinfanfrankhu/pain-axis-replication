"""Headless runner for scripts/3.2_pain_vectors/01: loads the original script
unmodified (its hash is pinned in v2_controls), overrides MODELS and OUTPUT_DIR,
then calls main(). Run from the repo root:
    python replication/tools/run_extract.py HF_REPO:OUT_NAME [...]"""
import sys
import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py")
spec = importlib.util.spec_from_file_location("extract", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

m.MODELS = [tuple(a.split(":", 1)) for a in sys.argv[1:]]
m.OUTPUT_DIR = Path("replication/results/3.2_pain_vectors")
print("MODELS =", m.MODELS, "OUTPUT_DIR =", m.OUTPUT_DIR, flush=True)
m.main()
