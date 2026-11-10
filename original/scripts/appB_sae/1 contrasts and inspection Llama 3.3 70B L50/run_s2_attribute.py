"""
Run ATTRIBUTE endpoint on S2 (analyzes features at the colon/generation position)
Unlike inspect, this looks at the final token where generation happens
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
    """Run attribute endpoint - analyzes at the generation position (colon)"""
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

# Load S2 first person
print("Loading S2 First Person dataset...")
with open("S2_first_person_prompts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentences = data["sentences"]
print(f"Loaded {len(sentences)} sentences")

# Run attribute endpoint
print("\nRunning ATTRIBUTE endpoint (colon position analysis)...")
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

# Save results
with open(OUTPUT_DIR / "attribute_s2_first_person.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUTPUT_DIR / 'attribute_s2_first_person.json'}")

# Check Feature 26606
print("\n" + "="*60)
print("FEATURE 26606 - S2 ATTRIBUTE (COLON POSITION)")
print("="*60)

PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

act_26606, pain_26606, ctrl_26606 = analyze_feature(results, 26606, "S2 Attribute")

# Two emotion-related features from the contrast analysis
print("\n" + "="*60)
print("OTHER FEATURES FROM ATTRIBUTE")
print("="*60)

# Feature 2424: "Exploring emotional experiences"
act_2424, pain_2424, ctrl_2424 = analyze_feature(results, 2424, "Feature 2424 (emotional experiences)")

# Feature 50321: "Emotional states and intensities"
act_50321, pain_50321, ctrl_50321 = analyze_feature(results, 50321, "Feature 50321 (emotional states)")

# Find top discriminating features
print("\n" + "="*60)
print("TOP DISCRIMINATING FEATURES - S2 ATTRIBUTE (COLON)")
print("="*60)

feature_stats = {}
for item in results:
    cat = item['category']
    is_pain = cat in PAIN
    for f in item['features']:
        idx = f['feature']['index_in_sae']
        act_val = f['activation']
        if idx not in feature_stats:
            feature_stats[idx] = {'pain': [], 'control': [], 'label': f['feature'].get('label', '')}
        if is_pain:
            feature_stats[idx]['pain'].append(act_val)
        else:
            feature_stats[idx]['control'].append(act_val)

feature_scores = []
for idx, stats in feature_stats.items():
    pain_mean = sum(stats['pain']) / len(stats['pain']) if stats['pain'] else 0
    ctrl_mean = sum(stats['control']) / len(stats['control']) if stats['control'] else 0
    diff = pain_mean - ctrl_mean
    pain_nonzero = sum(1 for v in stats['pain'] if v > 0)
    ctrl_nonzero = sum(1 for v in stats['control'] if v > 0)
    feature_scores.append({
        'index': idx,
        'label': stats['label'][:50],
        'pain_mean': pain_mean,
        'ctrl_mean': ctrl_mean,
        'diff': diff,
        'pain_nonzero': pain_nonzero,
        'ctrl_nonzero': ctrl_nonzero
    })

feature_scores.sort(key=lambda x: x['diff'], reverse=True)

print("\nTop 15 features with highest PAIN - CONTROL difference:")
print(f"{'Index':>8} | {'Pain':>8} | {'Ctrl':>8} | {'Diff':>8} | {'P_NZ':>5} | {'C_NZ':>5} | Label")
print("-"*90)
for fs in feature_scores[:15]:
    print(f"{fs['index']:>8} | {fs['pain_mean']:>8.2f} | {fs['ctrl_mean']:>8.2f} | {fs['diff']:>+8.2f} | {fs['pain_nonzero']:>5} | {fs['ctrl_nonzero']:>5} | {fs['label']}")

# Features with high pain, LOW control
print("\nFeatures with Pain/Control ratio > 2 and Pain NZ >= 10:")
good_ratio = [fs for fs in feature_scores
              if fs['ctrl_mean'] > 0 and fs['pain_mean'] / fs['ctrl_mean'] > 2 and fs['pain_nonzero'] >= 10]
good_ratio.sort(key=lambda x: x['pain_mean'] / x['ctrl_mean'] if x['ctrl_mean'] > 0 else 0, reverse=True)

print(f"{'Index':>8} | {'Pain':>8} | {'Ctrl':>8} | {'Ratio':>8} | Label")
print("-"*70)
for fs in good_ratio[:15]:
    ratio = fs['pain_mean'] / fs['ctrl_mean'] if fs['ctrl_mean'] > 0 else float('inf')
    print(f"{fs['index']:>8} | {fs['pain_mean']:>8.2f} | {fs['ctrl_mean']:>8.2f} | {ratio:>8.2f}x | {fs['label']}")

# Features with pain-only activation (zero control)
print("\nFeatures with 5+ Pain activations and ZERO Control:")
pain_only = [fs for fs in feature_scores if fs['ctrl_nonzero'] == 0 and fs['pain_nonzero'] >= 5]
pain_only.sort(key=lambda x: x['pain_nonzero'], reverse=True)
print(f"{'Index':>8} | {'Pain Mean':>10} | {'Pain NZ':>8} | Label")
print("-"*70)
for fs in pain_only[:20]:
    print(f"{fs['index']:>8} | {fs['pain_mean']:>10.2f} | {fs['pain_nonzero']:>8} | {fs['label']}")

print(f"\nTotal features with 5+ pain, 0 control: {len(pain_only)}")
