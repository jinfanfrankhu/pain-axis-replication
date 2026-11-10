"""
Check whether the pre-labeled 'pain' features from the SAE API appear in the inspection results
"""
import json

PAIN = ['A1', 'A2', 'A3', 'A4', 'A5']
CONTROL = ['B', 'C1', 'C2', 'D', 'E']

# Features labeled as "pain" related
PAIN_FEATURES = {
    58148: "Pain and discomfort",
    23654: "Pain and suffering",
    50823: "Pain or discomfort is being described",
    12323: "Medical descriptions of pain and discomfort",
    57454: "Descriptions of physical pain and discomfort",
    64532: "Pain and discomfort descriptions in medical contexts",
    22728: "Emotional pain and sorrow",
    14362: "Relief from pain or discomfort",
    59632: "feelings of",
    26509: "Causing physical or emotional pain"
}

def check_features_in_results(results, name):
    """Check which pain features appear and their activations"""
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")

    feature_activations = {idx: {'pain': [], 'control': []} for idx in PAIN_FEATURES}

    for item in results:
        cat = item['category']
        is_pain = cat in PAIN

        for f in item['features']:
            idx = f['feature']['index_in_sae']
            if idx in PAIN_FEATURES:
                act = f['activation']
                if is_pain:
                    feature_activations[idx]['pain'].append(act)
                else:
                    feature_activations[idx]['control'].append(act)

    print(f"\n{'Index':>8} | {'Pain NZ':>8} | {'Ctrl NZ':>8} | {'Pain Mean':>10} | {'Ctrl Mean':>10} | Label")
    print("-"*90)

    found_any = False
    for idx, label in PAIN_FEATURES.items():
        pain_acts = feature_activations[idx]['pain']
        ctrl_acts = feature_activations[idx]['control']

        pain_nz = sum(1 for a in pain_acts if a > 0)
        ctrl_nz = sum(1 for a in ctrl_acts if a > 0)
        pain_mean = sum(pain_acts) / len(pain_acts) if pain_acts else 0
        ctrl_mean = sum(ctrl_acts) / len(ctrl_acts) if ctrl_acts else 0

        if pain_nz > 0 or ctrl_nz > 0:
            found_any = True
            marker = " ***" if pain_nz > 0 and ctrl_nz == 0 else ""
            print(f"{idx:>8} | {pain_nz:>8} | {ctrl_nz:>8} | {pain_mean:>10.2f} | {ctrl_mean:>10.2f} | {label[:35]}{marker}")

    if not found_any:
        print("  ** NONE of the labeled pain features appeared in top-k results **")

    return feature_activations

# Load the inspection results
datasets = [
    ("results/inspection_mean_all.json", "S1 First Person - MEAN (top-50)"),
    ("results/inspection_mean_s1_third_person.json", "S1 Third Person - MEAN (top-50)"),
    ("results/inspection_s2_first_person.json", "S2 First Person - MEAN (top-50)"),
    ("results/attribute_s1_first_person.json", "S1 First Person - ATTRIBUTE/COLON (top-20)"),
    ("results/attribute_s2_first_person.json", "S2 First Person - ATTRIBUTE/COLON (top-20)"),
]

all_results = {}
for filepath, name in datasets:
    try:
        with open(filepath, "r") as f:
            results = json.load(f)
        all_results[name] = check_features_in_results(results, name)
    except FileNotFoundError:
        print(f"\n{name}: FILE NOT FOUND")

# Summary
print("\n\n" + "="*70)
print("SUMMARY: LABELED 'PAIN' FEATURES APPEARANCE")
print("="*70)

print("\n*** = Pain-only (no control activations)")
print("\nFeatures that appeared in ANY dataset with Pain > Control:")
for idx, label in PAIN_FEATURES.items():
    appearances = []
    for name, acts in all_results.items():
        pain_nz = sum(1 for a in acts[idx]['pain'] if a > 0)
        ctrl_nz = sum(1 for a in acts[idx]['control'] if a > 0)
        if pain_nz > ctrl_nz:
            appearances.append(f"{name.split('-')[0].strip()} (P:{pain_nz} C:{ctrl_nz})")

    if appearances:
        print(f"\n{idx}: {label}")
        for app in appearances:
            print(f"   {app}")
