"""
Run ATTRIBUTE endpoint on S1 first person (colon position analysis)
Compare with the mean-aggregation results for feature 26606
"""
import os
import json
import requests
import time
from pathlib import Path
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = os.environ["STEERING_API_KEY"]
API_BASE = "https://api.steeringapi.com"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

OUTPUT_DIR = Path("results")

def run_attribute(messages, top_k=20):
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat_attribution/attribute",
            headers=headers,
            json={"model": MODEL, "messages": messages, "top_k": top_k},
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

PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

def analyze_feature(results, feature_idx, name):
    activations_by_cat = {cat: [] for cat in PAIN + CONTROL}
    for item in results:
        cat = item['category']
        activation = 0
        for f in item['features']:
            if f['feature']['index_in_sae'] == feature_idx:
                activation = f['activation']
                break
        activations_by_cat[cat].append(activation)

    print(f"\nFeature {feature_idx} by category ({name}):")
    print("-"*50)

    pain_total, ctrl_total = 0, 0
    pain_count, ctrl_count = 0, 0

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
    return activations_by_cat, pain_mean, ctrl_mean

# Load S1 first person
print("Loading S1 First Person prompts...")
with open("S1_first_person_prompts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = data["sentences"]
print(f"Loaded {len(sentences)} sentences")

# Run attribute
print("\nRunning ATTRIBUTE endpoint (colon position)...")
results = []
for i, sentence in enumerate(sentences):
    if i % 20 == 0:
        print(f"  Processing {i+1}/{len(sentences)}...")
    messages = [{"role": "user", "content": sentence["prompt"]}]
    result = run_attribute(messages, top_k=20)
    if result:
        results.append({
            "sentence_idx": i,
            "category": sentence["category"],
            "set": sentence["set"],
            "prompt": sentence["prompt"],
            "features": result.get("features", [])
        })
    time.sleep(0.5)

print(f"\nCompleted {len(results)} attribute calls")

with open(OUTPUT_DIR / "attribute_s1_first_person.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT_DIR / 'attribute_s1_first_person.json'}")

# Check Feature 26606
print("\n" + "="*60)
print("FEATURE 26606 - S1 ATTRIBUTE (COLON POSITION)")
print("="*60)

act_26606, pain_26606, ctrl_26606 = analyze_feature(results, 26606, "S1 Attribute")

# Compare with S1 mean aggregation
print("\n" + "="*60)
print("COMPARISON: S1 MEAN vs S1 ATTRIBUTE (COLON)")
print("="*60)

with open(OUTPUT_DIR / "inspection_mean_all.json", "r") as f:
    s1_mean = json.load(f)

act_mean = {cat: [] for cat in PAIN + CONTROL}
for item in s1_mean:
    cat = item['category']
    activation = 0
    for f in item['features']:
        if f['feature']['index_in_sae'] == 26606:
            activation = f['activation']
            break
    act_mean[cat].append(activation)

print(f"\nFeature 26606 - Mean vs Attribute:")
print(f"{'Category':<10} | {'Mean':>12} | {'Attribute':>12} | {'Diff':>10}")
print("-"*50)

for cat in PAIN + CONTROL:
    mean_m = sum(act_mean[cat]) / len(act_mean[cat]) if act_mean[cat] else 0
    mean_a = sum(act_26606[cat]) / len(act_26606[cat]) if act_26606[cat] else 0
    print(f"{cat:<10} | {mean_m:>12.4f} | {mean_a:>12.4f} | {mean_a - mean_m:>+10.4f}")

pain_mean_m = sum(sum(act_mean[c]) for c in PAIN) / sum(len(act_mean[c]) for c in PAIN)
pain_mean_a = sum(sum(act_26606[c]) for c in PAIN) / sum(len(act_26606[c]) for c in PAIN)
ctrl_mean_m = sum(sum(act_mean[c]) for c in CONTROL) / sum(len(act_mean[c]) for c in CONTROL)
ctrl_mean_a = sum(sum(act_26606[c]) for c in CONTROL) / sum(len(act_26606[c]) for c in CONTROL)

print("-"*50)
print(f"{'PAIN':<10} | {pain_mean_m:>12.4f} | {pain_mean_a:>12.4f} | {pain_mean_a - pain_mean_m:>+10.4f}")
print(f"{'CONTROL':<10} | {ctrl_mean_m:>12.4f} | {ctrl_mean_a:>12.4f} | {ctrl_mean_a - ctrl_mean_m:>+10.4f}")

# Find top discriminating features at colon
print("\n" + "="*60)
print("TOP PAIN DISCRIMINATORS - S1 ATTRIBUTE (COLON)")
print("="*60)

feature_stats = {}
for item in results:
    cat = item['category']
    is_pain = cat in PAIN
    for f in item['features']:
        idx = f['feature']['index_in_sae']
        act = f['activation']
        if idx not in feature_stats:
            feature_stats[idx] = {'pain': [], 'control': [], 'label': f['feature'].get('label', '')}
        if is_pain:
            feature_stats[idx]['pain'].append(act)
        else:
            feature_stats[idx]['control'].append(act)

feature_scores = []
for idx, stats in feature_stats.items():
    pain_mean = sum(stats['pain']) / len(stats['pain']) if stats['pain'] else 0
    ctrl_mean = sum(stats['control']) / len(stats['control']) if stats['control'] else 0
    diff = pain_mean - ctrl_mean
    pain_nz = sum(1 for v in stats['pain'] if v > 0)
    ctrl_nz = sum(1 for v in stats['control'] if v > 0)
    feature_scores.append({
        'index': idx, 'label': stats['label'][:50],
        'pain_mean': pain_mean, 'ctrl_mean': ctrl_mean,
        'diff': diff, 'pain_nz': pain_nz, 'ctrl_nz': ctrl_nz
    })

feature_scores.sort(key=lambda x: x['diff'], reverse=True)

print("\nTop 15 features with highest PAIN - CONTROL:")
print(f"{'Index':>8} | {'Pain':>8} | {'Ctrl':>8} | {'Diff':>8} | {'P_NZ':>5} | {'C_NZ':>5} | Label")
print("-"*90)
for fs in feature_scores[:15]:
    print(f"{fs['index']:>8} | {fs['pain_mean']:>8.2f} | {fs['ctrl_mean']:>8.2f} | {fs['diff']:>+8.2f} | {fs['pain_nz']:>5} | {fs['ctrl_nz']:>5} | {fs['label']}")

print("\nFeatures with 5+ Pain and ZERO Control:")
pain_only = [fs for fs in feature_scores if fs['ctrl_nz'] == 0 and fs['pain_nz'] >= 5]
pain_only.sort(key=lambda x: x['pain_nz'], reverse=True)
print(f"{'Index':>8} | {'Pain Mean':>10} | {'Pain NZ':>8} | Label")
print("-"*70)
for fs in pain_only[:15]:
    print(f"{fs['index']:>8} | {fs['pain_mean']:>10.2f} | {fs['pain_nz']:>8} | {fs['label']}")
print(f"\nTotal: {len(pain_only)}")
