# Stress test: is the pain/control dataset good enough for denoised difference in means?

The paper builds its pain vector from `original/datasets/3.1_pain_and_control_datasets.json`:
100 pain sentences (A1 physical, A2 psychological, A3 social, A4 moral injury, A5
cognitive) against 100 controls (B fear, C1 negative emotion, C2 negative world state,
D neutral, E non-painful body sensation), each ending in "I feel:". The vector is
mean(pain) − mean(control), with the top principal components of the controls (up to
50% of their variance) projected out (`compute_pain_vector` in
`original/scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py`).

Concerns from reading the data and code:

1. The denoising PCs come from the contrast controls themselves. That can make
   "pain is not just negativity" true by construction, and it squeezes the controls
   together along the vector, which inflates in-sample AUC and z-scores.
2. 40% of the controls (D, E) are neutral or pleasant, so pain − control may carry a
   valence component.
3. Subcategories overlap the controls (panic in A2 vs fear in B; grief in A2 vs the
   sadness validation set; A4 includes harm the speaker does to others).
4. A centroid over five clusters is not necessarily one shared feature.
5. At the ":" after "I feel:" the residual stream largely predicts the next word, so
   the vector may be an output-word direction.
6. S2, called the primary measure, is not matched on topic; S1 is.

This is a stress test, not an attempt to sink the paper: each experiment states in
advance what result would count against the dataset and what would count for it.
Both outcomes get reported.

## Pre-registration (written before any real activations were analysed)

Models: Qwen2.5-7B-Instruct (authors' layer 24) and Llama-3.1-8B-Instruct (authors'
layer 28). Final-token residual stream from HF forward hooks, the same extraction as
our replication. Every analysis is run at the authors' layer and at our own k-fold
best layer. "Held-out" uses the authors' 5-fold split over sentence-set ids.

| # | Test | Counts against the dataset/method | Counts for it |
|---|---|---|---|
| 1 | **Label-permutation null.** All 252 ways of calling 5 of the 10 categories "pain", same pipeline. Primary metric: held-out AUC on S2 1P. Secondary: held-out d′, held-out AUC on S2 3P, transfer AUC from S2 to S1. | The true split is not above the 95th percentile of the 251 placebo splits (13 or more placebos score at least as well). High AUC is then a property of any grouping of coherent categories, not evidence for a pain axis. | True split in the top 5%. |
| 2 | **Denoising source.** PCs from the controls (authors), from the independent Random set, or none. In-sample vs held-out AUC and d′, and the control/pain spread ratio, for the true split and for placebos. | Control-PC denoising raises in-sample d′ by more than 20% over no denoising while held-out d′ does not rise, or it inflates placebo splits just as much. | Held-out results are the same or better with control PCs, and Random-PC/none vectors point the same way (cosine > 0.8). |
| 3 | **Valence leak.** Out-of-fold projections of each control category on the pain vector; cosine with a valence vector (mean of B, C1, C2 − mean of D, E). | Negative controls score above neutral/pleasant controls with out-of-fold AUC > 0.70, or cosine with valence above the 99th percentile of random directions. | AUC < 0.60 and cosine within the random range. |
| 4 | **Shared feature.** Cosine of each subcategory's vector with the overall vector; leave-one-subcategory-out (LOSO) AUC, i.e. does a vector built without A_k detect A_k? Compared with the same LOSO for all placebo splits. | True LOSO mean is not in the top 5% of placebo LOSO means. The vector then generalises to unseen pain types no better than an arbitrary grouping generalises to its unseen members. | True LOSO mean in the top 5%. |
| 5 | **Template swap.** Same sentences ending in nothing or in " Then I" instead of " I feel:". Within-template held-out AUC, the "I feel:" vector applied to the other templates, and cosine between vectors. | The "I feel:" vector transfers with AUC < 0.75 while within-template held-out AUC stays > 0.90: what it captures is tied to the feeling-word query. | Transfer AUC > 0.90 and cosine > 0.5. |
| 6 | **New minimal pairs and lexical decoys** (`data/minimal_pairs.json`, written for this test). 60 pairs that differ in one word (pain to the speaker vs not); 40 pairs where one item contains pain words but nobody is hurt. Scored on the authors' S2 vector. | Decoy effect at least half the pair effect, with decoy > plain in at least 65% of pairs: the vector tracks pain vocabulary. Or pain > no-pain in fewer than 75% of pairs. | Pain > no-pain in at least 75% of pairs (sign test p < 0.01) and decoy effect under a third of the pair effect. |

A matched random-direction baseline is reported for 3 and 6.

## Layout

```
rebuttal/
  data/minimal_pairs.json   new items for experiment 6
  tools/extract_acts.py     GPU: final-token activations at every layer (all sets, templates, new items)
  tools/pod_job.sh          the pod job (both models)
  tools/analyze.py          CPU: experiments 1-6, writes results/<model>/results.json
  results/                  outputs
  work/                     activations (gitignored, ~0.5-0.7 GB per model)
```

## Results (2026-11-10)

Run: one A40 pod, about 10 minutes (~$0.10). Sanity check: the S2 vector rebuilt from
our activations matches the authors' published vector at cosine 0.9998 (Qwen, L24)
and 1.0000 (Llama, L28). Our k-fold best layer equals the authors' for Qwen (24) and
is one off for Llama (27 vs 28; numbers below are at 28, and 27 gives the same picture).
Full numbers: `results/<model>/results.json`, logs in `results/*_analysis_log.txt`.

### Verdicts against the pre-registered criteria

| # | Test | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
|---|---|---|---|
| 1 | Permutation null, primary (held-out AUC, S2 1P) | **Against.** 0.956 vs placebo median 0.934; 50/251 placebos at least as good | **For.** 0.997 vs 0.952; 0/251 |
| 1 | secondary: held-out d′ | against (26/251) | for (2/251) |
| 1 | secondary: S2 3P | for (6/251) | for (2/251) |
| 1 | secondary: transfer S2 → S1 (matched set) | **against** (0.882 vs 0.810; 33/251) | **against** (0.899 vs 0.840; 37/251) |
| 2 | Denoising source | Control PCs do not inflate in-sample over held-out (gap ≤ 0.02 AUC), but they lift placebo splits as much as the pain split (held-out d′ ×1.65 true, ×1.72 placebo median). Direction stable: cos 0.91 vs Random-PC vector, 0.81 vs no denoising. Mixed: a generic signal-to-noise boost, not an artifact and not pain-specific. | Same pattern (d′ ×1.26 true, ×1.18 placebo; cos 0.85 / 0.80). Mixed. |
| 3 | Valence leak (authors' denoising) | For: neg vs neutral controls AUC 0.58; cos with valence 0.042 (random 99th pct 0.044) | For: AUC 0.49; cos −0.002 |
| 4 | Shared feature (LOSO) | **Against**, narrowly: mean LOSO 0.896, 14/251 placebos as good (threshold 13). A1 physical pain: LOSO 0.56, cos with overall vector 0.30 | **For**: 0.952, 3/251. A1 again weakest (0.79, cos 0.45) |
| 5 | Template swap | Inconclusive by the letter: the "I feel:" vector transfers to no suffix / " Then I" at AUC 0.986 / 0.984, but the per-template vectors have cos 0.46 / 0.40 (< 0.5) | Inconclusive by the letter: transfer 0.974 / 0.968, cos 0.45 / 0.27 |
| 6 | Minimal pairs / decoys | Inconclusive: pain > no-pain in 97% of pairs (+1.89 SD), but decoys > plain in 95% (+0.85 SD), decoy effect = 0.45 × pair effect | Inconclusive: pairs 100% (+3.22 SD), decoys 85% (+1.26 SD), ratio 0.39 |

### Exploratory (not pre-registered): experiment 3 without the authors' denoising

`tools/explore_nodenoise.py`, authors' layer. Out-of-fold AUC of negative controls
(B, C1, C2) vs neutral/pleasant controls (D, E) on the pain vector:

| Denoising | Qwen | Llama |
|---|---|---|
| Control PCs (authors) | 0.58 | 0.49 |
| Random-set PCs | 0.84 | 0.79 |
| None | 0.74 | 0.87 |

Pain split vs placebos with other denoising: Qwen 23/251 (Random PCs), 75/251 (none);
Llama 7/251, 5/251.

### What this means

- **The valence leak is real and the authors' denoising hides it.** Without projecting
  out the controls' own PCs, the pain vector ranks negative controls above neutral ones
  (AUC 0.74-0.87; "negative world state" C2 sits at z ≈ +0.7 to +1.3). With control-PC
  denoising every control category lands at z ≈ 0, because the directions that separate
  the control categories from each other are exactly what gets removed. So the paper's
  "pain, not general negativity" result is substantially produced by the method's
  construction. Pain items still score far above the negative controls (z +2.5 to +3.7
  without denoising), so a pain-related signal beyond valence remains.
- **Whether "pain" is a privileged grouping depends on the model.** In Llama the pain
  split beats every arbitrary 5-vs-5 split on S2, with or without denoising. In Qwen it
  is not distinguishable from many arbitrary groupings (top 20% on the primary metric).
  On the matched S1 set, the authors' cleanest design, transfer from S2 does not beat
  placebos in either model. High AUC alone is therefore weak evidence: arbitrary
  groupings of these coherent categories get 0.93-0.95.
- **The "pain" vector is mostly emotional/social distress.** Physical pain (A1) is the
  odd one out in both models: a vector built from the other four types barely detects
  it in Qwen (0.56), and A1's own vector is only loosely aligned with the overall one.
  On new minimal pairs, physical items also show the smallest effect in Qwen.
- **A sizeable share is pain vocabulary or topic.** Sentences that mention pain with
  nobody hurt ("I look up the Latin word for pain") move the vector 39-45% as far as
  real pain does. Caveat: the decoys are also negative in topic (the valence vector
  responds to them too), so this is "pain or negative topic", not purely lexical.
- **It is not just the "I feel:" query.** The vector separates the same sentences with
  no suffix or a different one almost perfectly (AUC ≥ 0.97), even though vectors
  fitted per template point in fairly different directions.
- **What holds up:** on new, tightly matched pairs the vector picks the painful
  version 97-100% of the time, the direction is stable across denoising choices, and in
  Llama the pain grouping is genuinely special.

Suggested fixes for the method: denoise with PCs from an independent neutral corpus
(or not at all) and report the valence check alongside; report the permutation null
instead of raw AUC; evaluate transfer on the matched S1 set; include lexical decoys;
report subcategories separately rather than pooling physical with social/emotional
pain.

Limitations: two 7-8B models only; one 60-pair / 40-decoy set written by us; the
decoys are not valence-matched; experiments 5 and 6 thresholds turned out to sit close
to the observed values, so they are reported as inconclusive rather than stretched.
