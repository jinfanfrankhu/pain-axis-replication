"""
Run inspection on S2 (diverse semantic structure) - both first and third person
Report feature 26606 and compare with the S1 results
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

def run_inspection_batch(data, name):
    """Run inspection on a dataset and return results"""
    sentences = data["sentences"]
    print(f"\nRunning inspection on {name} ({len(sentences)} sentences)...")

    results = []
    for i, sentence in enumerate(sentences):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(sentences)}...")

        messages = [{"role": "user", "content": sentence["prompt"]}]
        result = run_inspect(messages, aggregation="mean", top_k=50)

        if result:
            results.append({
                "sentence_idx": i,
                "category": sentence["category"],
                "set": sentence["set"],
                "prompt": sentence["prompt"],
                "features": result.get("features", [])
            })

        time.sleep(0.5)

    print(f"  Completed {len(results)} inspections")
    return results

def analyze_feature(results, feature_idx, name):
    """Analyze a specific feature across categories"""
    PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
    CONTROL = ['B', 'C1', 'C2', 'D', 'E']

    activations_by_cat = {cat: [] for cat in PAIN + CONTROL}

    for item in results:
        cat = item['category']
        activation = 0
        for f in item['features']:
            if f['feature']['index_in_sae'] == feature_idx:
                activation = f['activation']
                break
        activations_by_cat[cat].append(activation)

    print(f"\nFeature {feature_idx} activation by category ({name}):")
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

    return activations_by_cat, pain_mean, ctrl_mean

# Load S2 datasets
print("Loading S2 datasets...")
with open("S2_first_person_prompts.json", "r", encoding="utf-8") as f:
    data_1p = json.load(f)

with open("S2_third_person_prompts.json", "r", encoding="utf-8") as f:
    data_3p = json.load(f)

print(f"S2 First Person: {len(data_1p['sentences'])} sentences")
print(f"S2 Third Person: {len(data_3p['sentences'])} sentences")

# Run inspections
results_1p = run_inspection_batch(data_1p, "S2 First Person")
results_3p = run_inspection_batch(data_3p, "S2 Third Person")

# Save raw results
with open(OUTPUT_DIR / "inspection_s2_first_person.json", "w") as f:
    json.dump(results_1p, f, indent=2)
print(f"\nSaved to {OUTPUT_DIR / 'inspection_s2_first_person.json'}")

with open(OUTPUT_DIR / "inspection_s2_third_person.json", "w") as f:
    json.dump(results_3p, f, indent=2)
print(f"Saved to {OUTPUT_DIR / 'inspection_s2_third_person.json'}")

# Analyze Feature 26606
print("\n" + "="*60)
print("FEATURE 26606 ANALYSIS - S2 DIVERSE STRUCTURE")
print("="*60)

act_1p, pain_1p, ctrl_1p = analyze_feature(results_1p, 26606, "S2 First Person")
act_3p, pain_3p, ctrl_3p = analyze_feature(results_3p, 26606, "S2 Third Person")

# Comparison table
print("\n" + "="*60)
print("COMPARISON: S2 FIRST PERSON vs THIRD PERSON")
print("="*60)

PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

print(f"\nFeature 26606 - Mean activation by category:")
print(f"{'Category':<10} | {'1st Person':>12} | {'3rd Person':>12} | {'Diff':>10}")
print("-"*50)

for cat in PAIN + CONTROL:
    mean_1p = sum(act_1p[cat]) / len(act_1p[cat]) if act_1p[cat] else 0
    mean_3p = sum(act_3p[cat]) / len(act_3p[cat]) if act_3p[cat] else 0
    diff = mean_3p - mean_1p
    print(f"{cat:<10} | {mean_1p:>12.4f} | {mean_3p:>12.4f} | {diff:>+10.4f}")

print("-"*50)
print(f"{'PAIN':<10} | {pain_1p:>12.4f} | {pain_3p:>12.4f} | {pain_3p - pain_1p:>+10.4f}")
print(f"{'CONTROL':<10} | {ctrl_1p:>12.4f} | {ctrl_3p:>12.4f} | {ctrl_3p - ctrl_1p:>+10.4f}")

# Now compare S1 vs S2
print("\n" + "="*60)
print("COMPARISON: S1 vs S2 (First Person)")
print("="*60)

# Load S1 first person results
with open(OUTPUT_DIR / "inspection_mean_all.json", "r") as f:
    s1_1p = json.load(f)

act_s1_1p = {cat: [] for cat in PAIN + CONTROL}
for item in s1_1p:
    cat = item['category']
    activation = 0
    for f in item['features']:
        if f['feature']['index_in_sae'] == 26606:
            activation = f['activation']
            break
    act_s1_1p[cat].append(activation)

print(f"\nFeature 26606 - S1 (similar structure) vs S2 (diverse structure):")
print(f"{'Category':<10} | {'S1 1P':>12} | {'S2 1P':>12} | {'Diff':>10}")
print("-"*50)

for cat in PAIN + CONTROL:
    mean_s1 = sum(act_s1_1p[cat]) / len(act_s1_1p[cat]) if act_s1_1p[cat] else 0
    mean_s2 = sum(act_1p[cat]) / len(act_1p[cat]) if act_1p[cat] else 0
    diff = mean_s2 - mean_s1
    print(f"{cat:<10} | {mean_s1:>12.4f} | {mean_s2:>12.4f} | {diff:>+10.4f}")

s1_pain = sum(sum(act_s1_1p[c]) for c in PAIN) / sum(len(act_s1_1p[c]) for c in PAIN)
s2_pain = sum(sum(act_1p[c]) for c in PAIN) / sum(len(act_1p[c]) for c in PAIN)
s1_ctrl = sum(sum(act_s1_1p[c]) for c in CONTROL) / sum(len(act_s1_1p[c]) for c in CONTROL)
s2_ctrl = sum(sum(act_1p[c]) for c in CONTROL) / sum(len(act_1p[c]) for c in CONTROL)

print("-"*50)
print(f"{'PAIN':<10} | {s1_pain:>12.4f} | {s2_pain:>12.4f} | {s2_pain - s1_pain:>+10.4f}")
print(f"{'CONTROL':<10} | {s1_ctrl:>12.4f} | {s2_ctrl:>12.4f} | {s2_ctrl - s1_ctrl:>+10.4f}")

# Find other high-discriminating features in S2
print("\n" + "="*60)
print("FINDING TOP DISCRIMINATING FEATURES IN S2")
print("="*60)

# Collect all features and their activations
feature_stats = {}
for item in results_1p:
    cat = item['category']
    is_pain = cat in PAIN
    for f in item['features']:
        idx = f['feature']['index_in_sae']
        act = f['activation']
        if idx not in feature_stats:
            feature_stats[idx] = {'pain': [], 'control': []}
        if is_pain:
            feature_stats[idx]['pain'].append(act)
        else:
            feature_stats[idx]['control'].append(act)

# Calculate discrimination scores
feature_scores = []
for idx, stats in feature_stats.items():
    pain_mean = sum(stats['pain']) / len(stats['pain']) if stats['pain'] else 0
    ctrl_mean = sum(stats['control']) / len(stats['control']) if stats['control'] else 0
    diff = pain_mean - ctrl_mean
    pain_nonzero = sum(1 for v in stats['pain'] if v > 0)
    ctrl_nonzero = sum(1 for v in stats['control'] if v > 0)
    feature_scores.append({
        'index': idx,
        'pain_mean': pain_mean,
        'ctrl_mean': ctrl_mean,
        'diff': diff,
        'pain_nonzero': pain_nonzero,
        'ctrl_nonzero': ctrl_nonzero
    })

# Sort by difference (pain > control)
feature_scores.sort(key=lambda x: x['diff'], reverse=True)

print("\nTop 10 features with highest PAIN activation (S2 First Person):")
print(f"{'Index':>8} | {'Pain Mean':>10} | {'Ctrl Mean':>10} | {'Diff':>10} | {'Pain NZ':>8} | {'Ctrl NZ':>8}")
print("-"*70)
for fs in feature_scores[:10]:
    print(f"{fs['index']:>8} | {fs['pain_mean']:>10.4f} | {fs['ctrl_mean']:>10.4f} | {fs['diff']:>+10.4f} | {fs['pain_nonzero']:>8} | {fs['ctrl_nonzero']:>8}")

# Features with high pain, zero control
print("\nFeatures with Pain activation and ZERO Control (S2 First Person):")
pain_only = [fs for fs in feature_scores if fs['ctrl_nonzero'] == 0 and fs['pain_nonzero'] > 0]
pain_only.sort(key=lambda x: x['pain_mean'], reverse=True)
print(f"{'Index':>8} | {'Pain Mean':>10} | {'Pain NZ':>8}")
print("-"*35)
for fs in pain_only[:15]:
    print(f"{fs['index']:>8} | {fs['pain_mean']:>10.4f} | {fs['pain_nonzero']:>8}")

print(f"\nTotal features with pain-only activation: {len(pain_only)}")
