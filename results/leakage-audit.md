# Data-hygiene audit (2026-10-18, scripts/leakage_audit.py)

## SNIPS (official split, as distributed by snipsco/nlu-benchmark)

- train=13784, validate=700
- exact-text overlap train↔validate: **10 utterances** (short utterances
  such as "what is the weather here"; inherent to the official split,
  not introduced by this repo)
- near-duplicates (word 3-gram Jaccard >= 0.8): **14/700** (2.0%)

The official SNIPS train/validate split ships with this overlap. 10/700
exact duplicates bound their contribution to ~1.4% of evaluation items —
far below the 95% target margin, and identical for every method compared
in this repo (base, supervised reference, RL arms).

## BFCL v4 simple_python (synthetic train vs real eval)

- synth-train=15840, eval=400
- exact question overlap: **0** (the generator's question-level dedup
  guard held)
- eval questions sharing a word 8-gram with synthetic train: **3/400**
  (0.75%) — all standard mathematical phrasings ("find the area of a
  triangle with a base of ..."), i.e. the shared n-gram is the canonical
  way to phrase the underlying math, not a copied evaluation question

Conclusion: no material train→eval leakage in either dataset under
exact, near-duplicate, and 8-gram tests.
