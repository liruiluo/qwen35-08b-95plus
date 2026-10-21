#!/usr/bin/env python3
"""Interpolated-start pure-RL trainer (reset -> un-reset anneal).

Trains the UN-RESET model with the standard SPHERE-GRPO recipe (pure RL,
identical to sphere_grpo.py), but for the first --interp-steps steps the
weights SERVED to the rollout engine are a linear interpolation

    W_serve(a_t) = a_t * W_reset + (1 - a_t) * W_train,
    a_t = 1 - t / interp_steps   (a_0 = 1 -> pure reset model).

Step 0 therefore rolls out the reset model exactly (0.00% initial
success, the same measured starting point), and the anneal over the
first 100 steps ramps the served policy from the reset model to the
(un-reset, RL-updated) training model — until results are above the
reset model's flatline. From step interp_steps on, the engine serves
W_train directly and the run is pure RL to --steps.

Gradients always flow into W_train (the un-reset lineage); only the
serving path is interpolated. Rollouts during the anneal come from the
interpolated policy, so the advantage signal is computed on the annealed
policy's behaviour — the standard on-policy assumption is violated only
during the anneal, by construction.

Usage:
  python rl_interp.py --data SNIPS/data-jb/train.jsonl \
      --base /root/models/Qwen3.5-0.8B \
      --interp-base /root/models/Qwen3.5-0.8B-RandomInit \
      --rollout-url http://127.0.0.1:30001 \
      --arm last_adaptive --reward snips \
      --out ckpt/jb-interp-last_adaptive --steps 3000 --interp-steps 100
"""
import argparse, json, math, os, random, sys, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from methods.sphere_grpo import (SphereHook, load_rows, push_weights,  # noqa: E402
                                  reward_bfcl, reward_snips, reward_snips_intent, rollout,
                                  sphere_grad_layer, RHO)


def blend_into(holder, reset, train, alpha):
    """holder <- alpha * reset + (1 - alpha) * train (in place, no grad)."""
    with torch.no_grad():
        for (_, pr), (_, pt), (_, ph) in zip(reset.named_parameters(),
                                              train.named_parameters(),
                                              holder.named_parameters()):
            ph.copy_(alpha * pr.float() + (1.0 - alpha) * pt.float())


def push_model(url, model):
    model.config.use_cache = True
    push_weights(url, model)
    model.config.use_cache = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True,
                    help="un-reset base checkpoint (the model being trained)")
    ap.add_argument("--interp-base", required=True,
                    help="reset (random-init) checkpoint for the anneal")
    ap.add_argument("--interp-steps", type=int, default=100)
    ap.add_argument("--rollout-url", default="http://127.0.0.1:30001")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["last_adaptive", "all_hidden_adaptive", "last_fixed"])
    ap.add_argument("--reward", required=True, choices=["snips", "snips_intent", "bfcl"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--gen-temp", type=float, default=1.0)
    ap.add_argument("--name-credit", type=float, default=0.25)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--train-pool", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sync-every", type=int, default=1)
    ap.add_argument("--microbatch", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--rho", type=float, default=1e-3,
                    help="SPHERE term scale; --rho 0 = pure GRPO baseline")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    reset_model = AutoModelForCausalLM.from_pretrained(
        args.interp_base, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda()
    serve_model = AutoModelForCausalLM.from_pretrained(
        args.interp_base, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda()
    for m in (reset_model, serve_model):
        m.config.use_cache = False
        for p in m.parameters():
            p.requires_grad_(False)

    n_layers = len(model.model.layers)
    if args.arm == "all_hidden_adaptive":
        target_layers, alpha_mode = list(range(n_layers)), "adaptive"
    elif args.arm == "last_adaptive":
        target_layers, alpha_mode = [n_layers - 1], "adaptive"
    else:
        target_layers, alpha_mode = [n_layers - 1], 1.0

    rows = load_rows(args.data, "snips" if args.reward.startswith("snips") else args.reward)
    if args.train_pool:
        rng = random.Random(1000 + args.seed)
        rows = rng.sample(rows, min(args.train_pool, len(rows)))
    print(f"arm={args.arm} layers={len(target_layers)} alpha={alpha_mode} "
          f"rows={len(rows)} reward={args.reward} base={args.base} "
          f"interp_base={args.interp_base} interp_steps={args.interp_steps}",
          flush=True)

    sphere = SphereHook(model, target_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    os.makedirs(args.out, exist_ok=True)
    evidence = {"arm": args.arm, "target_layers": target_layers,
               "n_target_layers": len(target_layers),
               "alpha_mode": str(alpha_mode), "rho": args.rho, "seed": args.seed,
               "group": args.group, "prompts_per_step": args.prompts_per_step,
               "lr": args.lr, "base": args.base, "reward": args.reward,
               "train_pool": args.train_pool, "data": args.data,
               "interp": {"base": args.interp_base,
                          "steps": args.interp_steps,
                          "schedule": "a_t = 1 - t/interp_steps on serve path"},
               "optimizer_steps": 0, "steps": 0,
               "lambda_stats": [], "reward_first10": None,
               "reward_last10": None}

    # step 0 must roll out the pure reset model
    push_model(args.rollout_url, serve_model)

    rewards_hist = []
    t0 = time.time()
    for step in range(args.steps):
        a_now = max(0.0, 1.0 - step / args.interp_steps)
        batch_rows = random.sample(rows, args.prompts_per_step)
        P, G = args.prompts_per_step, args.group

        prompt_ids, plens = [], []
        for r in batch_rows:
            kw = dict(tokenize=False, enable_thinking=False,
                      add_generation_prompt=True)
            if r["tools"]:
                kw["tools"] = r["tools"]
            p = tok.apply_chat_template(r["messages"], **kw)
            ids = tok(p, add_special_tokens=False)["input_ids"]
            prompt_ids.append(ids)
            plens.append(len(ids))

        outs = rollout(args.rollout_url, prompt_ids, args.gen_temp,
                       args.max_new_tokens, G)

        rewards = []
        seqs = []
        for pi in range(P):
            for g in range(G):
                text = outs[pi][g]
                r = batch_rows[pi]
                if args.reward == "snips":
                    rewards.append(reward_snips(text, r["gold"],
                                                args.name_credit))
                elif args.reward == "snips_intent":
                    rewards.append(reward_snips_intent(text, r["gold"]))
                else:
                    rewards.append(reward_bfcl(text, r["gold_item"]))
                resp_ids = tok(text, add_special_tokens=False)["input_ids"]
                seqs.append((prompt_ids[pi] + resp_ids, plens[pi]))
        rewards = torch.tensor(rewards, dtype=torch.float32).cuda()

        rg = rewards.view(P, G)
        adv = (rg - rg.mean(dim=1, keepdim=True)) / (rg.std(dim=1, keepdim=True) + 1e-4)
        adv = adv.view(-1)

        B = len(seqs)
        maxlen = max(len(ids) for ids, _ in seqs)
        gen_ids = torch.full((B, maxlen), pad, dtype=torch.long)
        attn_full = torch.zeros((B, maxlen), dtype=torch.long)
        labels = torch.full((B, maxlen), -100, dtype=torch.long)
        for i, (ids, plen) in enumerate(seqs):
            L = len(ids)
            gen_ids[i, :L] = torch.tensor(ids)
            attn_full[i, :L] = 1
            labels[i, plen:L] = torch.tensor(ids[plen:])
        gen_ids, attn_full = gen_ids.cuda(), attn_full.cuda()
        resp_mask = (labels[:, 1:].cuda() != -100)

        CH = args.microbatch
        sphere.clear()
        opt.zero_grad(set_to_none=True)
        phi_collect = {li: [] for li in target_layers}
        denom = resp_mask.float().sum().clamp(min=1.0)
        for c0 in range(0, B, CH):
            c1 = min(c0 + CH, B)
            logits = model(input_ids=gen_ids[c0:c1],
                           attention_mask=attn_full[c0:c1]).logits
            for li in target_layers:
                phi = sphere.stash[li]
                phi_collect[li].append(
                    phi[:, :-1, :][resp_mask[c0:c1]].detach())
            sphere.clear()
            lg = logits[:, :-1]
            tok_logp = (lg.gather(2, gen_ids[c0:c1, 1:].unsqueeze(-1))
                        .squeeze(-1).float()
                        - torch.logsumexp(lg.float(), dim=-1))
            tok_logp = torch.nan_to_num(tok_logp, nan=0.0, posinf=0.0,
                                        neginf=0.0)
            m = resp_mask[c0:c1].float()
            closs = -((tok_logp * m * adv[c0:c1].unsqueeze(1)).sum() / denom)
            closs.backward()
            del logits, lg, tok_logp
        g_ppo = {n: p.grad.detach().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        gn_ppo = math.sqrt(sum(float((g * g).sum()) for g in g_ppo.values()))
        if not math.isfinite(gn_ppo):
            print(f"step {step}: non-finite PPO gradient, skipping", flush=True)
            sphere.clear()
            continue

        opt.zero_grad(set_to_none=True)
        sph_terms, alphas = [], []
        dl_rows, off = {}, {}
        for li in target_layers:
            allrows = torch.cat(phi_collect[li], 0)
            if allrows.shape[0] == 0:
                continue
            l, a, dl = sphere_grad_layer(allrows, alpha_mode)
            sph_terms.append(l)
            alphas.append(a)
            dl_rows[li] = dl
            off[li] = 0
        g_sphere, gn_sphere = {}, 0.0
        replay_rel_l2, surr_ip_dev = None, None
        if sph_terms:
            l2n, refn = 0.0, 0.0
            ip_closed, ip_replay = 0.0, 0.0
            for ci, c0 in enumerate(range(0, B, CH)):
                c1 = min(c0 + CH, B)
                sphere.clear()
                _ = model.model(input_ids=gen_ids[c0:c1],
                                attention_mask=attn_full[c0:c1])
                surr = None
                for li in target_layers:
                    if li not in dl_rows:
                        continue
                    phi = sphere.stash[li]
                    phi_r = phi[:, :-1, :][resp_mask[c0:c1]]
                    ref = phi_collect[li][ci]
                    nr = phi_r.shape[0]
                    dl_c = dl_rows[li][off[li]:off[li] + nr]
                    off[li] += nr
                    d = phi_r.detach().float() - ref
                    l2n += float(d.pow(2).sum())
                    refn += float(ref.pow(2).sum())
                    ip_replay += float((phi_r.detach().float() * dl_c).sum())
                    ip_closed += float((ref * dl_c).sum())
                    t = (phi_r.float() * dl_c).sum()
                    surr = t if surr is None else surr + t
                if surr is not None:
                    surr.backward()
            g_sphere = {n: p.grad.detach().clone()
                        for n, p in model.named_parameters() if p.grad is not None}
            gn_sphere = math.sqrt(sum(float((g * g).sum())
                                      for g in g_sphere.values()))
            replay_rel_l2 = (l2n ** 0.5) / (refn ** 0.5 + 1e-12)
            surr_ip_dev = (abs(ip_replay - ip_closed)
                           / (abs(ip_closed) + 1e-12))
        sph_loss = sum(sph_terms) if sph_terms else 0.0

        lam = args.rho * gn_ppo / (gn_sphere + 1e-8) if gn_sphere > 0 else 0.0
        with torch.no_grad():
            for n, p in model.named_parameters():
                gp, gs = g_ppo.get(n), g_sphere.get(n)
                if gp is None and gs is None:
                    continue
                if gp is None:
                    gp = torch.zeros_like(p)
                if gs is None:
                    gs = torch.zeros_like(p)
                p.grad = gp + lam * gs
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # ---- push: interpolated serve during the anneal, then W_train
        if (step + 1) % args.sync_every == 0 or step == args.steps - 1:
            nxt = step + 1
            if nxt < args.interp_steps:
                a_next = 1.0 - nxt / args.interp_steps
                blend_into(serve_model, reset_model, model, a_next)
                push_model(args.rollout_url, serve_model)
            else:
                push_model(args.rollout_url, model)

        if (args.save_every and (step + 1) % args.save_every == 0
                and step != args.steps - 1):
            sub = os.path.join(args.out, f"step{step + 1}")
            os.makedirs(sub, exist_ok=True)
            model.config.use_cache = True
            sds = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
            model.save_pretrained(sub, state_dict=sds)
            tok.save_pretrained(sub)
            model.config.use_cache = False
            ev = dict(evidence)
            ev["saved_at_step"] = step + 1
            json.dump(ev, open(os.path.join(sub, "evidence.json"), "w"),
                      indent=1)

        mr = float(rewards.mean())
        rewards_hist.append(mr)
        evidence["optimizer_steps"] += 1
        evidence["lambda_stats"].append(
            {"step": step, "g_ppo": gn_ppo, "g_sphere": gn_sphere,
             "lambda": lam, "sphere_loss": float(sph_loss),
             "alphas": alphas, "mean_reward": mr,
             "interp_alpha": a_now,
             "action_tokens": int(resp_mask.sum().item()),
             "replay_rel_l2": replay_rel_l2, "surr_ip_dev": surr_ip_dev})
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step}/{args.steps} reward {mr:.3f} "
                  f"interp_a {a_now:.2f} "
                  f"|g_ppo| {gn_ppo:.4f} |g_sphere| {gn_sphere:.3g} "
                  f"lambda {lam:.3e} sph {float(sph_loss) if sph_terms else 0:.4g} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
        sphere.clear()

    evidence["steps"] = args.steps
    n1, n2 = rewards_hist[:10], rewards_hist[-10:]
    evidence["reward_first10"] = sum(n1) / len(n1) if n1 else None
    evidence["reward_last10"] = sum(n2) / len(n2) if n2 else None
    evidence["hook_calls"] = sphere.calls
    json.dump(evidence, open(os.path.join(args.out, "evidence.json"), "w"),
              indent=1)
    with open(os.path.join(args.out, "training-log.jsonl"), "w") as f:
        for e in evidence["lambda_stats"]:
            f.write(json.dumps(e) + "\n")

    model.config.use_cache = True
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.save_pretrained(args.out, state_dict=sd)
    tok.save_pretrained(args.out)
    sphere.remove()
    print(f"DONE arm={args.arm} reward {evidence['reward_first10']} -> "
          f"{evidence['reward_last10']} opts={evidence['optimizer_steps']} "
          f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
