# Postmortem: gold-answer leak in RL rollout prompts

**Found**: 2026-10-18, before any RL number was reported. All RL results in
this repo are from after the fix.

## Symptom

Early GRPO smoke runs showed mean training reward 0.93–0.95 from step 0,
while every honest evaluation of the same base weights measured the
equivalent of ~0.43–0.51 (SNIPS: 0.5×intent 0.827 + 0.5×slot-F1 0.194).
A policy cannot start above its own ceiling — the reward was measuring
something other than the task.

## Root cause

`SNIPS/data/train.jsonl` rows carry the SFT layout
`[system, user, assistant-gold]`. The RL data loader
(`methods/sphere_grpo.py::load_rows`, SNIPS branch) passed **all three
turns** to `apply_chat_template(..., add_generation_prompt=True)`. The gold
answer was therefore part of the rollout prompt; the policy learned to
parrot the answer visible in its context, and the group-relative advantage
mostly measured parroping fidelity, not NLU.

The BFCL branch of the same loader was already clean (it rebuilt a fresh
`[system, user]` pair). Only the SNIPS branch leaked.

## Fix

`load_rows` now takes `d["messages"][:-1]` for SNIPS rollouts
(commit "Fix gold-answer leak in SNIPS rollout prompts"). Step-0 mean
reward after the fix: 0.432 at temperature 1.0 — consistent with the
evaluated base level, closing the training-vs-evaluation gap.

## Why it survived a day of investigation

The discrepancy was first misattributed to the serving protocol (a real
but separate effect: trained checkpoints collapse 0.00 under the
chat-completions renderer). Both effects produce a "training says high,
eval says low" signature. The decisive test was re-running the honest
evaluation against the exact weights the trainer was sampling from —
they disagreed, which no protocol artifact can explain.

## Checks added

- Rollout prompts exclude the assistant turn by construction; the BFCL
  loader builds prompts from `question` + `function` only.
- `results/` records step-0 rewards for every arm; a step-0 mean far above
  the evaluated base level (~0.43–0.51 reward-scale on SNIPS) indicates a
  prompt or reward bug, not learning.
