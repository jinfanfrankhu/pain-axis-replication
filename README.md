# Pain Axis: replication, extensions and stress test

Replication and extension of "The Pain Axis: LLMs Represent Self-Directed Harm and
Act on It" (Tagliabue, Dung & Berg 2026, [arXiv 2609.16247](https://arxiv.org/abs/2609.16247))
on open models.

| Folder | What it is |
|---|---|
| [`original/`](original/README.md) | The authors' code, datasets, v2 controls and shipped results, unmodified. |
| [`replication/`](replication/README.md) | Our replication runs, extensions (self-medication, honeypot agent, steering demo) and the demo site (`replication/demo`, deployed on Vercel). |
| [`rebuttal/`](rebuttal/README.md) | A pre-registered stress test of the pain/control dataset and the denoised difference-in-means method. |

The authors' scripts read `datasets/` and write `results/` relative to the working
directory, so run them from inside `original/`. Our wrappers in `replication/tools/`
run from the repo root and point the scripts at `original/`.

License: MIT (see `LICENSE`, from the original repository).
