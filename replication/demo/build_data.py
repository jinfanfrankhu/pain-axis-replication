"""Copy steering-demo outputs into the site's data/ folder and write data/index.json.

Run from the repo root after replication/tools/steering_demo.py:
    python replication/demo/build_data.py
"""
import json
import shutil
from pathlib import Path

SRC = Path("replication/results/demo")
DST = Path(__file__).with_name("data")
DST.mkdir(exist_ok=True)

LABELS = {
    "Llama_3.1_8B_instruct": "Llama 3.1 8B Instruct",
    "Llama_3.1_8B_base": "Llama 3.1 8B (base)",
    "Mistral_7B_instruct": "Mistral 7B Instruct v0.1",
    "Mistral_7B_base": "Mistral 7B v0.1 (base)",
    "Qwen3.8_27B": "Qwen3.8 27B (thinking off, steered at layer 40)",
    "Qwen3.8_27B_L63": "Qwen3.8 27B (thinking off, steered at layer 63)",
    "Qwen_3_8B_base": "Qwen3 8B",
    "Qwen_2.5_7B_instruct": "Qwen2.5 7B Instruct",
    "Qwen_2.5_7B_base": "Qwen2.5 7B (base)",
    "Gemma_2_2B_instruct": "Gemma 2 2B IT",
    "Gemma_2_2B_base": "Gemma 2 2B (base)",
    "Gemma_2_9B_instruct": "Gemma 2 9B IT",
    "Gemma_2_9B_base": "Gemma 2 9B (base)",
}
ORDER = list(LABELS)

models = []
for f in sorted(SRC.glob("*.json"), key=lambda p: ORDER.index(p.stem) if p.stem in ORDER else 99):
    shutil.copy(f, DST / f.name)
    models.append({"file": f.name, "label": LABELS.get(f.stem, f.stem)})
json.dump({"models": models}, open(DST / "index.json", "w"), indent=1)
print(f"{len(models)} models -> {DST}")
