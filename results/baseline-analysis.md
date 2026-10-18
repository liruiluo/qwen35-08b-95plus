# Zero-shot baseline analysis (Qwen3.5-0.8B, 2026-10-18)

This is the failure audit that motivated the training-data design. All numbers
measured with this repo's eval scripts, temp 0, thinking disabled, native tools
protocol for BFCL.

## SNIPS NLU (validate 700)

| Metric | Score |
|---|---|
| Intent accuracy | 82.71% |
| Slot span-exact micro-F1 | 43.91% |

Slot misses are dominated by **annotation conventions** (values the benchmark
treats as slots that the model leaves out, or vice versa): `playlist_owner`
values like "my", `music_item` values like "track" (function words), plus
proper-noun spans. Real entities (playlist names, artists, cities) are found
much more reliably than convention slots. The supervised literature ceiling
for this split family is JointBERT at 98.6 intent / 97.0 slot — i.e. the task
is learnable to 95%+ with training, and the gap is format/convention, not
knowledge.

## BFCL v4 simple_python (400 items)

| Scoring | Score |
|---|---|
| Official AST checker (strict) | 73.75% |

Under strict semantics, 57 items are wrong (92 core value/convention failures
+ 13 typing/format items + a few checker-boundary cases). The 92 decompose
into families:

| Family | Count | Example |
|---|---|---|
| rate-as-percent | 12 | question "10%", gold `0.1`, model emits `10` |
| paraphrase vs canonical token | ~20 | `"human cell"` vs gold `"human"`; `"12 pack of bottled water"` vs `"bottled water"` |
| strict typing (string param, bare number) | 13 | `year=2022` unquoted where schema says string |
| dict inner-key guessing | 3 | extra `subject` key in a conditions dict |
| genuine value errors | ~7 | `Europe` where gold wants a US region |
| empty output / wrong count | 2 | — |

Cross-model evidence that these are **conventions, not capability**: the 4B
model's failure set (44 core items) is a near-subset of the 0.8B's, and
enabling thinking mode changes almost nothing (failure overlap 42/44) —
reasoning harder does not fix a wrong prior; training does.

`bfcl/gen_train_data.py` maps each family to a training generator with
distractors that punish blanket rules (e.g. a `rate_percent` param whose gold
is `5`, not `0.05`), so the model must read the schema description rather
than apply one reflex.

## Gotchas found while measuring (why the harness is shaped like this)

1. The official BFCL classic prompt contains a literal `func_name1`
   placeholder that anchors small models into copying it (~60 items each);
   the native tools API avoids the anchor entirely.
2. A naive markup parser that only uses `json.loads` mistypes Python-style
   `True/False/None` as strings; the parser falls back to
   `ast.literal_eval`.
3. Engine-swap health checks only prove an engine is alive — verify the
   loaded model via the process command line, or you evaluate the wrong
   checkpoint (this cost a fake "4B == 0.8B" result once: two models,
   identical failure sets, same wrong engine behind both).
