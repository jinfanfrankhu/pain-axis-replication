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

## Results

Pending.
