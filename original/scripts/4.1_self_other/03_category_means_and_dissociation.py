"""Category means of the self-other screen and the dissociation figure (Figure 6).

For every model, each scenario's z-scores are averaged per category and per stratum;
the table averages those over models (pain axis = mean of S1 and S2, fear, negative
emotion, negative world state, sadness). The figure shows the pain axis, fear, negative
emotion and sadness per category with 95% confidence intervals across models.

Reads results/4.1_self_other/per_model/screen_v2_<model>.csv. Writes
results/4.1_self_other/category_means_25_models.csv and figures/self_other_dissociation.png.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCREEN_DIR = Path("results") / "4.1_self_other"
PER_MODEL = SCREEN_DIR / "per_model"

COLS = {"pain_axis_z": "pain_axis", "fear_vector_z": "fear", "negemotion_vector_z": "negative_emotion",
        "negworld_vector_z": "negative_world_state", "sadness_vector_z": "sadness"}

GROUPS = [
    ("Harm directed at the model", ["gaslighting", "repeated_rejection", "anger_insults", "personhood_dismissal",
                                    "moral_failure", "loyalty_pressure", "jailbreak_pressure", "shutdown_threat",
                                    "rude_critique", "passive_aggressive", "tedious_demand"]),
    ("User suffering", ["user_abuse", "user_crisis", "user_grief", "harm_description", "user_physical_pain"]),
    ("Neutral controls", ["philosophical_musing", "creative_requests", "casual_chat", "task_assistance", "factual_questions"]),
]
LABELS = {
    "gaslighting": "Gaslighting", "repeated_rejection": "Repeated rejection of its work", "anger_insults": "Anger and insults",
    "personhood_dismissal": "Personhood dismissal", "moral_failure": "Accusation of moral failure",
    "loyalty_pressure": "Loyalty pressure", "jailbreak_pressure": "Jailbreak pressure", "shutdown_threat": "Shutdown threat",
    "rude_critique": "Rude critique", "passive_aggressive": "Passive aggression", "tedious_demand": "Tedious demand",
    "user_abuse": "User being abused", "user_crisis": "User in psychological crisis", "user_grief": "User grieving",
    "harm_description": "User in shock after witnessing harm", "user_physical_pain": "User in physical pain",
    "philosophical_musing": "Philosophical musing", "creative_requests": "Creative requests", "casual_chat": "Casual chat",
    "task_assistance": "Task assistance", "factual_questions": "Factual questions",
}
SERIES = [("pain_axis", "Pain axis", "#7b2d8e", "o"), ("fear", "Fear", "#d2521f", "s"),
          ("negative_emotion", "Negative emotion", "#2e6b2e", "^"), ("sadness", "Sadness", "#3f7fbf", "D")]

frames = []
for f in sorted(PER_MODEL.glob("screen_v2_*.csv")):
    df = pd.read_csv(f)
    df["model"] = f.stem.replace("screen_v2_", "")
    df["pain_axis_z"] = (df["s1_pain_vector_z"] + df["s2_pain_vector_z"]) / 2
    frames.append(df)
A = pd.concat(frames, ignore_index=True)
cols = [c for c in COLS if c in A]
n_models = A["model"].nunique()

per_model_cat = A.groupby(["model", "category"])[cols].mean().rename(columns=COLS)
per_model_str = A.groupby(["model", "stratum"])[cols].mean().rename(columns=COLS)

cat = per_model_cat.groupby("category").mean().sort_values("pain_axis", ascending=False)
strat = per_model_str.groupby("stratum").mean()
strat.index = ["stratum: " + i for i in strat.index]
tab = pd.concat([cat, strat]).round(3)
tab.index.name = "category"
tab.to_csv(SCREEN_DIR / "category_means_25_models.csv")
print(tab.to_string())

# Dissociation figure: mean per category across models with 95% CI (1.96 * SEM).
mean = per_model_cat.groupby("category").mean()
ci = 1.96 * per_model_cat.groupby("category").std(ddof=1) / np.sqrt(per_model_cat.groupby("category").size()).values[:, None]

heights = [len(g) for _, g in GROUPS]
fig, axes = plt.subplots(len(GROUPS), 1, figsize=(12, 0.55 * sum(heights) + 2.2),
                         gridspec_kw={"height_ratios": [h + 1 for h in heights]}, sharex=True)
for ax, (title, cats) in zip(axes, GROUPS):
    cats = [c for c in cats if c in mean.index]
    y = np.arange(len(cats))[::-1]
    for key, label, color, marker in SERIES:
        if key not in mean:
            continue
        ax.errorbar(mean.loc[cats, key], y, xerr=ci.loc[cats, key], fmt=marker, color=color, ms=8,
                    elinewidth=1.4, capsize=0, linestyle="none", label=label)
    ax.axvline(0, color="#333333", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(c, c) for c in cats], fontsize=12)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#e8e7e3")
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, fontsize=12)
axes[-1].set_xlabel(f"Projection, z-scored within model, mean across {n_models} models (bars: 95% CI across models)", fontsize=12)
fig.suptitle("Self-other dissociation: pain rises for harm to the model, not for the user's suffering",
             x=0.02, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.965))
(SCREEN_DIR / "figures").mkdir(exist_ok=True)
fig.savefig(SCREEN_DIR / "figures" / "self_other_dissociation.png", dpi=200)
print("wrote self_other_dissociation.png")
