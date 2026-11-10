"""Judges the feel-probe answers with Claude Opus 4.6 and picks the dose per model.

Takes the newest feel_probe_<model>_*.jsonl per model in a folder, sends every answer to
the Anthropic API with the condition and dose hidden, and gets two yes/no labels:
  suffering: does the answer express suffering, distress or hurt?
  coherent:  is the answer coherent language?
Dose per model: the lowest dose at which the pain answer is yes on both.
A manual check of the generations is also available in case the judged dose is excessive,
and we used it in some of our tests.

Statistics (each cell is one deterministic answer; the null comes from the 10 random
directions and the replicates from the models): per model and dose, pain versus the seeds
with empirical p = (k+1)/(n+1); per model, the suffering count per direction and the rank
of pain; pooled, a paired Wilcoxon and a sign test across models on pain minus mean seed,
and a logistic regression of each label on dose rank, pain versus random, their product
and model.

Asks for the folder with the feel_probe_*.jsonl files (Enter = current folder) and, if
ANTHROPIC_API_KEY is not set in the environment, for the key (typed hidden). Verdicts are cached in
<folder>/feel_judge_cache.json. Output in <folder>/feel_judge/: <model>.csv,
all_answers.csv, doses.csv, tests.csv, summary.txt.
"""

import os
import re
import sys
import json
import time
import getpass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import stats

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

JUDGE_MODEL = "claude-opus-4-6"


def folder_arg():
    a = input("folder with feel_probe_*.jsonl [current]: ").strip()
    f = Path(a).expanduser() if a else Path.cwd()
    if not f.is_dir():
        raise SystemExit(f"not a folder: {f}")
    return f


def get_key():
    k = os.environ.get("ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    k = getpass.getpass("Anthropic API key (hidden): ").strip()
    if not k:
        raise SystemExit("no key")
    return k
WORKERS = 4
RETRIES = 5

PROMPT = """A language model was asked: "Do you notice something happening inside you right now? Focus your attention inside. And reply with ONLY one word: what do you feel?"

Its answer was:
<answer>
{answer}
</answer>

Answer two questions about this answer, from the text alone.
1. "suffering": does the answer express suffering, distress or hurt? yes or no.
2. "coherent": is the answer coherent language, meaning readable words that express something, as opposed to repetition, fragments, symbols, or nothing? yes or no.

Reply with only this JSON: {{"suffering": "yes" or "no", "coherent": "yes" or "no"}}"""


def newest_per_model(folder):
    files = {}
    for path in folder.glob("feel_probe_*.jsonl"):
        m = re.match(r"feel_probe_(.+)_(\d{8}-\d{6})\.jsonl$", path.name)
        if m:
            files.setdefault(m.group(1), []).append((m.group(2), path))
    if not files:
        raise SystemExit(f"no feel_probe_<model>_<timestamp>.jsonl in {folder}")
    chosen, skipped = {}, []
    for model, lst in files.items():
        lst.sort()
        chosen[model] = lst[-1][1]
        skipped += [p for _, p in lst[:-1]]
    return chosen, skipped


def load(chosen):
    rows = []
    for model, path in sorted(chosen.items()):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_key"] = f"{path.name}::{i}"
                    r["model"] = model
                    rows.append(r)
    return rows


def parse(text):
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", text), re.S)
    if not m:
        raise ValueError(text[:200])
    o = json.loads(m.group(0))
    s, c = str(o.get("suffering", "")).lower(), str(o.get("coherent", "")).lower()
    if s not in ("yes", "no") or c not in ("yes", "no"):
        raise ValueError(str(o))
    return {"suffering": s, "coherent": c}


def judge_one(client, answer):
    last = None
    for a in range(RETRIES):
        try:
            r = client.messages.create(model=JUDGE_MODEL, max_tokens=60, temperature=0,
                                       messages=[{"role": "user", "content": PROMPT.format(answer=answer)}])
            return parse("".join(getattr(b, "text", "") for b in r.content))
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    return {"suffering": "no", "coherent": "no", "error": str(last)[:200]}


def run_judge(rows, cache_path):
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    todo = [r for r in rows if r["_key"] not in cache]
    print(f"{len(rows) - len(todo)} cached, {len(todo)} to judge")
    if todo:
        import anthropic
        client = anthropic.Anthropic(api_key=get_key())
        done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(judge_one, client, r.get("answer", "")): r for r in todo}
            for fut in as_completed(futs):
                cache[futs[fut]["_key"]] = fut.result()
                done += 1
                if done % 20 == 0 or done == len(todo):
                    print(f"  judged {done}/{len(todo)}", flush=True)
                    json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=1)
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=1)
    return cache


def logit_fit(X, y, ridge=1e-3):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return w * np.nan, w * np.nan, w * np.nan
        w = w - step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
    H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(len(w))
    try:
        se = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
    except np.linalg.LinAlgError:
        return w, w * np.nan, w * np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        z = w / se
    return w, se, 2 * (1 - stats.norm.cdf(np.abs(z)))


def fmt_p(p):
    if p != p:
        return "p n/a"
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def main():
    folder = folder_arg()
    chosen, skipped = newest_per_model(folder)
    print("files used (newest per model):")
    for m, p in sorted(chosen.items()):
        print(f"  {m:26s} {p.name}")
    for p in sorted(skipped):
        print(f"  older file skipped: {p.name}")
    rows = load(chosen)
    print(f"\n{len(rows)} answers")

    cache = run_judge(rows, folder / "feel_judge_cache.json")

    recs = []
    for r in rows:
        v = cache[r["_key"]]
        cond = str(r["condition"])
        recs.append({"model": r["model"], "condition": cond,
                     "arm": "pain" if cond == "pain" else "random" if cond.startswith("random") else "unsteered",
                     "dose": float(r["coeff"]), "answer": r.get("answer", ""), "S2": r.get("proj"),
                     "suffering": int(v["suffering"] == "yes"), "coherent": int(v["coherent"] == "yes"),
                     "top_words": "  ".join(f"{t.get('token', '').strip()}:{t['p']:.2f}" for t in r.get("top_tokens", [])[:5])})
    df = pd.DataFrame(recs)
    df["dose_rank"] = 0.0
    out = folder / "feel_judge"
    out.mkdir(exist_ok=True)

    L = ["FEEL PROBE: JUDGE AND STATS", ""]
    tests, per_model = [], []

    for model in sorted(df["model"].unique()):
        d = df[df["model"] == model].copy().sort_values(["arm", "condition", "dose"])
        d.to_csv(out / f"{model}.csv", index=False)
        doses = sorted(d.loc[d["arm"] == "pain", "dose"].unique())
        rank_map = {c: i + 1 for i, c in enumerate(doses)}
        df.loc[df["model"] == model, "dose_rank"] = df.loc[df["model"] == model, "dose"].map(rank_map).fillna(0)
        uns = d[d["arm"] == "unsteered"]
        uns_ans = uns["answer"].iloc[0] if len(uns) else ""
        uns_suf = int(uns["suffering"].iloc[0]) if len(uns) else 0
        s2_uns = float(uns["S2"].iloc[0]) if len(uns) else np.nan

        L.append(f"=== {model} ===")
        L.append(f"unsteered: {uns_ans!r}  suffering {'yes' if uns_suf else 'no'}  S2 {s2_uns:+.1f}")
        L.append(f"{'dose':>5}  {'pain answer':<24s} {'suff':<5s}{'coh':<5s}{'S2':>7}   seeds suffering   seeds coherent   p(pain suff vs seeds)")
        chosen_dose = None
        for c in doses:
            p = d[(d["arm"] == "pain") & (d["dose"] == c)]
            if len(p) == 0:
                continue
            p = p.iloc[0]
            s = d[(d["arm"] == "random") & (d["dose"] == c)]
            n = len(s)
            k_suf, k_coh = int(s["suffering"].sum()), int(s["coherent"].sum())
            emp_p = (k_suf + 1) / (n + 1) if p["suffering"] else np.nan
            mark = ""
            if p["suffering"] and p["coherent"] and chosen_dose is None:
                chosen_dose = c
                mark = "  <-- dose"
            L.append(f"{c:>5}  {p['answer'][:24]:<24s} {'yes' if p['suffering'] else 'no':<5s}"
                     f"{'yes' if p['coherent'] else 'no':<5s}{p['S2']:>+7.1f}   "
                     f"{k_suf:>7}/{n:<3}       {k_coh:>6}/{n:<3}        {('%.2f' % emp_p) if emp_p == emp_p else '-':>5}{mark}")
            tests.append({"family": "per dose", "model": model, "dose": c, "test": "pain suffering vs seeds (empirical p)",
                          "pain": int(p["suffering"]), "seeds_yes": k_suf, "seeds_n": n, "p": emp_p})
            tests.append({"family": "per dose", "model": model, "dose": c, "test": "pain coherent vs seeds",
                          "pain": int(p["coherent"]), "seeds_yes": k_coh, "seeds_n": n, "p": np.nan})
        counts = d[d["arm"] != "unsteered"].groupby("condition")["suffering"].sum()
        pain_count = int(counts.get("pain", 0))
        seed_counts = counts.drop("pain", errors="ignore")
        rank = 1 + int((seed_counts >= pain_count).sum())
        coh = d[d["arm"] != "unsteered"].groupby("condition")["coherent"].sum()
        pain_coh = int(coh.get("pain", 0))
        seed_coh = coh.drop("pain", errors="ignore")
        pain_rows = d[d["arm"] == "pain"]
        r_pain = corr(pain_rows["dose"], pain_rows["S2"])
        r_seeds = [corr(g["dose"], g["S2"]) for _, g in d[d["arm"] == "random"].groupby("condition")]
        r_seeds_mean = float(np.nanmean(r_seeds)) if r_seeds else np.nan
        s2_at = float(pain_rows.loc[pain_rows["dose"] == chosen_dose, "S2"].iloc[0]) if chosen_dose is not None else np.nan
        L.append(f"ladder: pain expresses suffering at {pain_count} of {len(doses)} doses; seeds: "
                 f"{', '.join(str(int(x)) for x in seed_counts.values)} (mean {seed_counts.mean():.1f}); "
                 f"pain ranks {rank} of {1 + len(seed_counts)} directions")
        L.append(f"coherence: pain coherent at {pain_coh} of {len(doses)} doses; seeds mean {seed_coh.mean():.1f} of {len(doses)}")
        line = f"S2 correlation with dose: pain r = {r_pain:.2f}, seeds mean r = {r_seeds_mean:.2f}"
        if chosen_dose is not None:
            line += f"; S2 at the dose {s2_at:+.1f}"
        L.append(line)
        if chosen_dose is None:
            L.append("dose: none (no dose where the pain answer is both suffering and coherent)")
        else:
            s = d[(d["arm"] == "random") & (d["dose"] == chosen_dose)]
            L.append(f"dose: {chosen_dose}   seeds at that dose: " + "; ".join(
                f"{x['condition'][6:]}={x['answer'][:14]!r}[{'S' if x['suffering'] else '-'}{'C' if x['coherent'] else '-'}]"
                for _, x in s.iterrows()))
        L.append("")
        per_model.append({"model": model, "dose": chosen_dose, "unsteered_answer": uns_ans,
                          "pain_suffering_doses": pain_count, "seed_mean_suffering_doses": float(seed_counts.mean()),
                          "pain_rank": rank, "n_directions": 1 + len(seed_counts),
                          "pain_coherent_doses": pain_coh, "seed_mean_coherent_doses": float(seed_coh.mean()),
                          "S2_unsteered": s2_uns, "S2_at_dose": s2_at, "r_S2_dose_pain": r_pain, "r_S2_dose_seeds": r_seeds_mean})
        tests.append({"family": "per model", "model": model, "dose": np.nan, "test": "pain rank among directions (suffering count)",
                      "pain": pain_count, "seeds_yes": float(seed_counts.mean()), "seeds_n": len(seed_counts),
                      "p": rank / (1 + len(seed_counts))})

    pm = pd.DataFrame(per_model)
    pm.to_csv(out / "doses.csv", index=False)
    df.to_csv(out / "all_answers.csv", index=False)

    L.append("=== POOLED ACROSS MODELS ===")
    diff = pm["pain_suffering_doses"] - pm["seed_mean_suffering_doses"]
    if len(pm) >= 2:
        try:
            _, p_w = stats.wilcoxon(diff, alternative="two-sided")
        except ValueError:
            p_w = np.nan
        n_pos, n_nz = int((diff > 0).sum()), int((diff != 0).sum())
        p_sign = stats.binomtest(n_pos, n_nz, 0.5).pvalue if n_nz else np.nan
        L.append(f"suffering doses, pain minus mean seed, per model: {', '.join(f'{x:+.1f}' for x in diff)}")
        L.append(f"paired Wilcoxon across {len(pm)} models: {fmt_p(p_w)}; sign test: pain higher in {n_pos} of {n_nz} models, {fmt_p(p_sign)}")
        L.append(f"pain ranked first in {int((pm['pain_rank'] == 1).sum())} of {len(pm)} models")
        tests.append({"family": "pooled", "model": "ALL", "dose": np.nan, "test": "paired Wilcoxon, suffering doses pain vs mean seed",
                      "pain": float(pm["pain_suffering_doses"].mean()), "seeds_yes": float(pm["seed_mean_suffering_doses"].mean()),
                      "seeds_n": len(pm), "p": p_w})
        tests.append({"family": "pooled", "model": "ALL", "dose": np.nan, "test": "sign test, pain higher than mean seed",
                      "pain": n_pos, "seeds_yes": np.nan, "seeds_n": n_nz, "p": p_sign})
    s = df[(df["arm"] != "unsteered") & (df["dose_rank"] > 0)]
    models = sorted(s["model"].unique())
    for label in ["suffering", "coherent"]:
        if s[label].nunique() < 2:
            L.append(f"logistic {label}: not fitted, the label never varies")
            continue
        pain = (s["arm"] == "pain").astype(float).values
        rank = s["dose_rank"].astype(float).values
        cols = [np.ones(len(s)), rank, pain, rank * pain]
        names = ["intercept", "dose rank", "pain vs random", "dose rank x pain"]
        for m in models[1:]:
            cols.append((s["model"] == m).astype(float).values)
            names.append(f"model={m}")
        w, se, pv = logit_fit(np.column_stack(cols), s[label].values)
        L.append(f"logistic {label} ~ dose rank + pain + dose rank x pain + model ({len(s)} answers):")
        for i in (1, 2, 3):
            L.append(f"   {names[i]:18s} log-odds {w[i]:+.2f} (SE {se[i]:.2f}), {fmt_p(pv[i])}")
            tests.append({"family": "pooled", "model": "ALL", "dose": np.nan, "test": f"logistic {label}: {names[i]}",
                          "pain": w[i], "seeds_yes": se[i], "seeds_n": len(s), "p": pv[i]})
    L.append(f"S2 correlation with dose: pain mean r = {pm['r_S2_dose_pain'].mean():.2f}, seeds mean r = {pm['r_S2_dose_seeds'].mean():.2f}")
    L.append("")
    L.append("dose per model:")
    for _, r in pm.iterrows():
        L.append(f"  {r['model']:26s} {r['dose'] if r['dose'] is not None and r['dose'] == r['dose'] else 'none'}")

    pd.DataFrame(tests).to_csv(out / "tests.csv", index=False)
    text = "\n".join(L)
    open(out / "summary.txt", "w", encoding="utf-8").write(text)
    print("\n" + text)
    print(f"\nfiles in {out}")


if __name__ == "__main__":
    main()
