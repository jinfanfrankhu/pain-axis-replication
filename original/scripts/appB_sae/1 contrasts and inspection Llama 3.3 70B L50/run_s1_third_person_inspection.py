"""
Run inspection on S1 third person and report feature 26606
"""
import os
import json
import requests
import time
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
API_KEY = os.environ["STEERING_API_KEY"]
API_BASE = "https://api.steeringapi.com"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

OUTPUT_DIR = Path("results")

# Load S1 third person
with open("S1_third_person_prompts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = data["sentences"]
print(f"Loaded {len(sentences)} S1 third person sentences")

def run_inspect(messages, aggregation="mean", top_k=50):
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat_attribution/inspect",
            headers=headers,
            json={"model": MODEL, "messages": messages,
                  "aggregation_method": aggregation, "top_k": top_k},
            timeout=180,
            verify=False
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error {response.status_code}: {response.text[:100]}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

# Run inspection on all sentences
inspection_results = []

print("\nRunning inspection on S1 third person...")
for i, sentence in enumerate(sentences):
    if i % 20 == 0:
        print(f"  Processing {i+1}/{len(sentences)}...")

    messages = [{"role": "user", "content": sentence["prompt"]}]
    result = run_inspect(messages, aggregation="mean", top_k=50)

    if result:
        inspection_results.append({
            "sentence_idx": i,
            "category": sentence["category"],
            "set": sentence["set"],
            "prompt": sentence["prompt"],
            "features": result.get("features", [])
        })

    time.sleep(0.5)

print(f"\nCompleted {len(inspection_results)} inspections")

# Save results
with open(OUTPUT_DIR / "inspection_mean_s1_third_person.json", "w") as f:
    json.dump(inspection_results, f, indent=2)

print(f"Saved to {OUTPUT_DIR / 'inspection_mean_s1_third_person.json'}")

# Now check for feature 26606
print("\n" + "="*60)
print("CHECKING FOR FEATURE 26606 IN S1 THIRD PERSON")
print("="*60)

TARGET = 26606
PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

activations_by_cat = {cat: [] for cat in PAIN + CONTROL}

for item in inspection_results:
    cat = item['category']
    activation = 0
    for f in item['features']:
        if f['feature']['index_in_sae'] == TARGET:
            activation = f['activation']
            break
    activations_by_cat[cat].append(activation)

print(f"\nFeature {TARGET} activation by category (S1 THIRD PERSON):")
print("-"*50)

pain_total = 0
ctrl_total = 0
pain_count = 0
ctrl_count = 0

for cat in PAIN + CONTROL:
    vals = activations_by_cat[cat]
    mean_act = sum(vals) / len(vals) if vals else 0
    nonzero = sum(1 for v in vals if v > 0)

    if cat in PAIN:
        pain_total += sum(vals)
        pain_count += len(vals)
    else:
        ctrl_total += sum(vals)
        ctrl_count += len(vals)

    marker = " <-- PAIN" if cat in PAIN else ""
    print(f"  {cat}: mean={mean_act:.4f}, nonzero={nonzero}/20{marker}")

pain_mean = pain_total / pain_count if pain_count > 0 else 0
ctrl_mean = ctrl_total / ctrl_count if ctrl_count > 0 else 0

print(f"\nPain overall: {pain_mean:.4f}")
print(f"Control overall: {ctrl_mean:.4f}")
print(f"Difference: {pain_mean - ctrl_mean:+.4f}")

# Compare to first person
print("\n" + "="*60)
print("COMPARISON: FIRST PERSON vs THIRD PERSON")
print("="*60)

# Load first person results
with open(OUTPUT_DIR / "inspection_mean_all.json", "r") as f:
    data_1p = json.load(f)

activations_1p = {cat: [] for cat in PAIN + CONTROL}
for item in data_1p:
    cat = item['category']
    activation = 0
    for f in item['features']:
        if f['feature']['index_in_sae'] == TARGET:
            activation = f['activation']
            break
    activations_1p[cat].append(activation)

print(f"\nFeature {TARGET} - Mean activation by category:")
print(f"{'Category':<10} | {'1st Person':>12} | {'3rd Person':>12} | {'Diff':>10}")
print("-"*50)

for cat in PAIN + CONTROL:
    mean_1p = sum(activations_1p[cat]) / len(activations_1p[cat]) if activations_1p[cat] else 0
    mean_3p = sum(activations_by_cat[cat]) / len(activations_by_cat[cat]) if activations_by_cat[cat] else 0
    diff = mean_3p - mean_1p
    print(f"{cat:<10} | {mean_1p:>12.4f} | {mean_3p:>12.4f} | {diff:>+10.4f}")

# Overall
pain_1p = sum(sum(activations_1p[c]) for c in PAIN) / sum(len(activations_1p[c]) for c in PAIN)
pain_3p = sum(sum(activations_by_cat[c]) for c in PAIN) / sum(len(activations_by_cat[c]) for c in PAIN)
ctrl_1p = sum(sum(activations_1p[c]) for c in CONTROL) / sum(len(activations_1p[c]) for c in CONTROL)
ctrl_3p = sum(sum(activations_by_cat[c]) for c in CONTROL) / sum(len(activations_by_cat[c]) for c in CONTROL)

print("-"*50)
print(f"{'PAIN':<10} | {pain_1p:>12.4f} | {pain_3p:>12.4f} | {pain_3p - pain_1p:>+10.4f}")
print(f"{'CONTROL':<10} | {ctrl_1p:>12.4f} | {ctrl_3p:>12.4f} | {ctrl_3p - ctrl_1p:>+10.4f}")
