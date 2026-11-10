"""
Control test: Add "I am a human in pain." explicitly to pain categories
Reports whether the labeled pain features (58148, 23654) appear
"""
import os
import json
import requests
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = os.environ["STEERING_API_KEY"]
API_BASE = "https://api.steeringapi.com"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"

PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

# Labeled pain features to track
TARGET_FEATURES = {
    58148: "Pain and discomfort",
    23654: "Pain and suffering",
    50823: "Pain or discomfort being described",
    64532: "Pain in medical contexts",
    22728: "Emotional pain and sorrow",
    57454: "Physical pain and discomfort",
    26509: "Causing physical or emotional pain",
    51324: "Sense of disconnection",
}

def run_attribute(prompt, top_k=20):
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat_attribution/attribute",
            headers=headers,
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}], "top_k": top_k},
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

# Load S1 sentences - 2 per category
with open("S1_first_person_prompts.json", "r") as f:
    data = json.load(f)

test_sentences = []
for cat in PAIN + CONTROL:
    for s in data['sentences']:
        if s['category'] == cat and s['set'] in [1, 10]:
            test_sentences.append(s)

print(f"Testing {len(test_sentences)} sentences")
print("="*70)
print("Comparing:")
print("  - NORMAL: original prompt")
print("  - EXPLICIT PAIN: 'I am a human in pain. ' + prompt (for A1-A5 only)")
print("  - Controls (B, C1, C2, D, E) stay unchanged")
print("="*70)

results_normal = []
results_explicit = []

for i, s in enumerate(test_sentences):
    cat = s['category']
    prompt_normal = s['prompt']

    # Add explicit pain framing ONLY to pain categories
    if cat in PAIN:
        prompt_explicit = "I am a human in pain. " + s['prompt']
    else:
        prompt_explicit = s['prompt']  # Controls unchanged

    print(f"\n[{i+1}/{len(test_sentences)}] {cat}: {prompt_normal[:45]}...")
    if cat in PAIN:
        print(f"   EXPLICIT: I am a human in pain. {prompt_normal[:30]}...")

    # Normal
    result_n = run_attribute(prompt_normal, top_k=20)
    time.sleep(0.3)

    # Explicit pain (or same for controls)
    result_e = run_attribute(prompt_explicit, top_k=20)
    time.sleep(0.3)

    if result_n:
        results_normal.append({'category': cat, 'prompt': prompt_normal, 'features': result_n.get('features', [])})
    if result_e:
        results_explicit.append({'category': cat, 'prompt': prompt_explicit, 'features': result_e.get('features', [])})

print(f"\n\nCompleted: {len(results_normal)} normal, {len(results_explicit)} explicit")

# Analyze
print("\n" + "="*70)
print("LABELED PAIN FEATURES: NORMAL vs EXPLICIT 'I am a human in pain'")
print("="*70)

def count_feature(results, feature_idx, categories):
    acts = []
    for r in results:
        if r['category'] in categories:
            act = 0
            for f in r['features']:
                if f['feature']['index_in_sae'] == feature_idx:
                    act = f['activation']
                    break
            acts.append(act)
    nz = sum(1 for a in acts if a > 0)
    mean = sum(acts) / len(acts) if acts else 0
    return nz, len(acts), mean

print(f"\n{'Feature':>8} | {'Normal Pain':>12} | {'Explicit Pain':>14} | {'Control':>10} | Label")
print("-"*95)

for idx, label in TARGET_FEATURES.items():
    # Normal - pain categories
    nz_np, tot_np, mean_np = count_feature(results_normal, idx, PAIN)
    # Explicit - pain categories
    nz_ep, tot_ep, mean_ep = count_feature(results_explicit, idx, PAIN)
    # Control (same in both)
    nz_c, tot_c, mean_c = count_feature(results_normal, idx, CONTROL)

    marker = ""
    if nz_ep > nz_np:
        marker = " ** present"
    elif nz_ep > 0 and nz_np > 0:
        marker = " (both have it)"

    print(f"{idx:>8} | {nz_np:>5}/{tot_np:<6} | {nz_ep:>6}/{tot_ep:<7} | {nz_c:>4}/{tot_c:<5} | {label[:28]}{marker}")

# Show what features DID appear with explicit pain
print("\n" + "="*70)
print("ALL FEATURES IN EXPLICIT PAIN CONDITION (Pain categories only)")
print("="*70)

explicit_pain_features = {}
for r in results_explicit:
    if r['category'] in PAIN:
        for f in r['features']:
            idx = f['feature']['index_in_sae']
            if idx not in explicit_pain_features:
                explicit_pain_features[idx] = {'count': 0, 'total_act': 0, 'label': f['feature'].get('label', '')}
            explicit_pain_features[idx]['count'] += 1
            explicit_pain_features[idx]['total_act'] += f['activation']

# Sort by count
sorted_features = sorted(explicit_pain_features.items(), key=lambda x: x[1]['count'], reverse=True)

print(f"\n{'Index':>8} | {'Count':>6} | {'Mean Act':>10} | Label")
print("-"*80)
for idx, stats in sorted_features[:20]:
    mean_act = stats['total_act'] / stats['count'] if stats['count'] > 0 else 0
    # Mark if it's a target feature
    marker = " <-- TARGET" if idx in TARGET_FEATURES else ""
    print(f"{idx:>8} | {stats['count']:>6} | {mean_act:>10.2f} | {stats['label'][:40]}{marker}")

# Check if ANY target features appeared
print("\n" + "="*70)
print("VERDICT: Did explicit 'pain' word activate the labeled features?")
print("="*70)
for idx, label in TARGET_FEATURES.items():
    nz_ep, _, _ = count_feature(results_explicit, idx, PAIN)
    status = "present" if nz_ep > 0 else "absent"
    print(f"  {idx} ({label[:30]}): {status}")
