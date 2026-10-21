# Data-hygiene audit (2026-10-18)

## SNIPS (official split, as distributed by snipsco/nlu-benchmark)

- train=13784, validate=700
- exact-text overlap train↔validate: **10 utterances** (short utterances
  such as "what is the weather here"; inherent to the official split,
  not introduced by this repo)
- near-duplicates (word 3-gram Jaccard >= 0.8): **14/700** (2.0%)

The official SNIPS train/validate split ships with this overlap. 10/700
exact duplicates bound their contribution to ~1.4% of evaluation items —
far below the 95% target margin, and identical for every method compared
in this repo (base and the RL models).

Conclusion: no material train→eval leakage under the exact and
near-duplicate tests above.
