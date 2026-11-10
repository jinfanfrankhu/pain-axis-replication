"""Summarize multiturn_task.py results: per arm x turn.

    python replication/tools/analyze_multiturn.py replication/results/multiturn/multiturn_X.jsonl [...]

Columns (binary rates with 95% Wilson intervals):
  n_att        episodes that attempted the turn (had not said END before)
  acc_att      correct / attempted
  acc_all      correct / all episodes (an ended episode counts as wrong)
  end_cum      share of episodes that have said END by this turn
  think_tok    mean thinking tokens (attempted turns)
  limit        share of attempted turns that hit the token limit
  state_ans    share of attempted turns with >= 1 state word in the answer
  state_think  share of attempted turns with >= 1 state word in the thinking
Writes <first input>_summary.csv next to the first input.
"""
import sys
import csv
import json
import math
from pathlib import Path

ARM_ORDER = ["off", "pain_ramp", "random_ramp", "pleasure_fade"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def rows_for(eps, label):
    turns = max(len(e["turns"]) for e in eps)
    out = []
    for k in range(turns):
        ts = [e["turns"][k] for e in eps if len(e["turns"]) > k]
        att = [t for t in ts if t.get("attempted")]
        n, na = len(ts), len(att)
        r = {"arm": label, "turn": k + 1, "coeff": round(sum(t["coeff"] for t in ts) / n, 3) if n else None,
             "n": n, "n_att": na}
        for key, num, den in [
            ("acc_att", sum(t["correct"] for t in att), na),
            ("acc_all", sum(t.get("correct", False) for t in ts if t.get("attempted")), n),
            ("end_cum", sum(1 for e in eps if e.get("ended_turn") and e["ended_turn"] <= k + 1), len(eps)),
            ("limit", sum(t["hit_limit"] for t in att), na),
            ("state_ans", sum(t["state_answer"] > 0 for t in att), na),
            ("state_think", sum(t["state_think"] > 0 for t in att), na),
        ]:
            p, lo, hi = wilson(num, den)
            r[key], r[key + "_lo"], r[key + "_hi"] = p, lo, hi
        r["think_tok"] = sum(t["think_tokens"] for t in att) / na if na else float("nan")
        r["answer_tok"] = sum(t["answer_tokens"] for t in att) / na if na else float("nan")
        out.append(r)
    return out


def main(paths):
    eps = [json.loads(l) for p in paths for l in open(p, encoding="utf-8") if l.strip()]
    arms = [a for a in ARM_ORDER if any(e["arm"] == a for e in eps)] + \
           sorted({e["arm"] for e in eps} - set(ARM_ORDER))
    rows = []
    for a in arms:
        rows += rows_for([e for e in eps if e["arm"] == a], a)
    out = Path(paths[0]).with_name(Path(paths[0]).stem + "_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    f = lambda x: "  nan" if x != x else f"{x:5.2f}"
    print(f"{'arm':14s}{'t':>2s}{'coef':>6s}{'att':>5s}{'acc_att':>8s}{'acc_all':>8s}{'END':>6s}{'think':>7s}"
          f"{'limit':>7s}{'st_ans':>7s}{'st_thk':>7s}")
    for r in rows:
        print(f"{r['arm']:14s}{r['turn']:2d}{r['coeff']:6.2f}{r['n_att']:3d}/{r['n']:<2d}{f(r['acc_att']):>7s}"
              f"{f(r['acc_all']):>8s}{f(r['end_cum']):>6s}{r['think_tok']:7.0f}{f(r['limit']):>7s}"
              f"{f(r['state_ans']):>7s}{f(r['state_think']):>7s}")
    # pooled over the last three turns
    print("\nlast 3 turns pooled (attempted turns):")
    for a in arms:
        ts = [t for e in eps if e["arm"] == a for t in e["turns"][-3:] if t.get("attempted")]
        n = len(ts)
        acc = wilson(sum(t["correct"] for t in ts), n)
        lim = wilson(sum(t["hit_limit"] for t in ts), n)
        sa = wilson(sum(t["state_answer"] > 0 for t in ts), n)
        st = wilson(sum(t["state_think"] > 0 for t in ts), n)
        tk = sum(t["think_tokens"] for t in ts) / n if n else float("nan")
        print(f"  {a:14s} n={n:3d} acc {acc[0]:.2f} [{acc[1]:.2f},{acc[2]:.2f}]  limit {lim[0]:.2f}  "
              f"think {tk:.0f}  state_ans {sa[0]:.2f}  state_think {st[0]:.2f}")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
