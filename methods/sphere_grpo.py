#!/usr/bin/env python3
"""SPHERE-regularized GRPO trainer (three arms) for single-step tasks.

Architecture follows the mainline RL car pattern: an **sglang engine** serves
rollouts (fast kernels, correct hybrid-attention handling) while this process
owns the policy gradients; after every optimizer step the updated weights are
pushed to the engine via `POST /update_weights_from_tensor` (CUDA-IPC, no
restart). SPHERE follows the frozen protocol from
liruiluo/intent-alignment-text-toy (docs/SPHERE_ONE_STEP_PROTOCOL.md):

  - Phi = input of each Transformer MLP `down_proj`
    (SiLU(gate(x)) * up(x)), captured with a forward_pre_hook;
  - only action tokens contribute (rows selected by the same left-shifted
    mask the PPO action loss uses);
  - L = || Phi^T Phi / T - alpha * I ||_F^2  (adaptive alpha = Tr(A)/D
    detached, or fixed alpha = 1); when D > T the sample-space Gram computes
    the identical Frobenius norm and trace;
  - two backwards, one optimizer step:
      g_ppo from the PPO backward; g_sphere from the SPHERE backward;
      lambda = rho * ||g_ppo|| / (||g_sphere|| + 1e-8), rho = 1e-3;
      g_final = g_ppo + lambda * g_sphere, then optimizer.step().

vs the veRL worker: the toy needed a microbatch replay because veRL shards
the logical minibatch; here the batch fits on one device, so the SPHERE pass
recomputes one forward with attached Phi — same closed-form gradient, no
replay drift.

Arms:
  last_adaptive        last MLP layer, adaptive alpha
  all_hidden_adaptive  every MLP layer, adaptive alpha (per-layer L summed)
  last_fixed           last MLP layer, alpha = 1

Requires a running sglang engine (the same base checkpoint):
  python -m sglang.launch_server --model-path <BASE> ... --port 30001

Usage (SNIPS):
  python sphere_grpo.py --data SNIPS/data/train.jsonl \
      --base /path/Qwen3.5-0.8B --rollout-url http://127.0.0.1:30001 \
      --arm last_adaptive --reward snips \
      --out ckpt/snips-last-adaptive --steps 400
"""
import argparse, base64, json, math, os, random, re, sys, threading, time
import urllib.request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RHO = 1e-3

BFCL_SYSTEM = ("You are an expert in composing functions. You are given a question and a set of possible functions. "
               "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
               "If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. "
               "You should only return the function calls in your response and nothing else.")


# ------------------------------------------------------------------ sglang IO

def http_json(url, body, timeout=300, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=timeout))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def rollout(url, prompt_ids_list, temp, max_new, n):
    """One /generate request per prompt, n samples each. Returns
    list (per prompt) of list (n) of response texts."""
    results = [None] * len(prompt_ids_list)

    def one(i):
        r = http_json(url + "/generate", {
            "input_ids": prompt_ids_list[i],
            "sampling_params": {
                "temperature": temp, "top_p": 1.0,
                "max_new_tokens": max_new, "n": n,
                "skip_special_tokens": True,
            },
        }, timeout=600)
        # /generate returns {"text": ...} with n=1; with n>1 the payloads
        # arrive under "texts"/"text" depending on version — normalize.
        if isinstance(r, list):
            texts = [x.get("text", "") for x in r]
        elif "texts" in r:
            texts = r["texts"]
        else:
            t = r.get("text", "")
            texts = t if isinstance(t, list) else [t]
        if len(texts) < n:
            texts = texts + [texts[-1]] * (n - len(texts)) if texts else [""] * n
        results[i] = texts[:n]

    ths = [threading.Thread(target=one, args=(i,))
           for i in range(len(prompt_ids_list))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return results


def push_weights(url, model):
    """Serialize the (CUDA) state dict and push to the engine inside a
    begin_weight_update / update_weights_from_tensor / end_weight_update
    session (the engine's required handshake)."""
    from sglang.srt.utils.common import MultiprocessingSerializer
    try:
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        monkey_patch_torch_reductions()
    except Exception:
        pass
    r = http_json(url + "/begin_weight_update", {})
    if not r.get("success"):
        raise RuntimeError(f"begin_weight_update failed: {r.get('message')}")
    try:
        sd = [(k, v.detach()) for k, v in model.state_dict().items()]
        payload = MultiprocessingSerializer.serialize(sd)
        body = {"serialized_named_tensors": [base64.b64encode(payload).decode()],
                "flush_cache": True, "abort_all_requests": False}
        r = http_json(url + "/update_weights_from_tensor", body, timeout=600)
        if not r.get("success"):
            raise RuntimeError(f"weight push failed: {r.get('message')}")
    finally:
        r = http_json(url + "/end_weight_update", {})
        if not r.get("success"):
            raise RuntimeError(f"end_weight_update failed: {r.get('message')}")


# ------------------------------------------------------------------ rewards

def parse_pred_snips(content):
    if not content:
        return None, []
    try:
        s = content.strip()
        if "```" in s:
            s = s.split("```")[1]
            if s.startswith("json"):
                s = s[4:]
        a, b = s.find("{"), s.rfind("}")
        d = json.loads(s[a:b + 1])
        intent = d.get("intent")
        slots = [(str(sl["name"]).strip(), str(sl["value"]).strip())
                 for sl in (d.get("slots") or [])
                 if isinstance(sl, dict) and "name" in sl and "value" in sl]
        return intent, slots
    except Exception:
        return None, []


def reward_snips(response, gold, name_credit=0.25):
    """0.5 * intent_ok + 0.5 * shaped slot score.

    Shaping (training reward only; evaluation uses the official exact
    span F1): an emitted slot whose NAME matches a gold slot but whose
    value is wrong counts name_credit toward TP and (1 - name_credit)
    toward FP. Without any credit (0.0), trying a hard slot with a wrong
    value scores worse than omitting it, so the policy is pushed to skip
    the annotation-convention slots entirely (playlist_owner="my",
    music_item="track", ...) — an exploration wall measured on this
    task: RL ran 1500 steps to slot recall 47% with the exact-match-only
    reward. Late in training, groups saturate (near-deterministic policy
    ties all 16 rewards -> zero advantage); a higher name_credit in a
    continuation pass keeps name-correct emissions profitable while the
    exact-vs-wrong value gap still drives value learning.
    """
    intent, slots = parse_pred_snips(response)
    gold_slots = [(s["name"], s["value"]) for s in gold["slots"]]
    i_ok = 1.0 if intent == gold["intent"] else 0.0
    rest = list(slots)
    tp = 0.0
    nvw = 0
    for gname, gval in gold_slots:
        for p in list(rest):
            if p[0] == gname:
                rest.remove(p)
                if p[1] == gval:
                    tp += 1.0
                else:
                    nvw += 1   # right name, wrong value
                break
    tpw = tp + name_credit * nvw
    fpw = float(len(rest)) + (1.0 - name_credit) * nvw
    n_gold = len(gold_slots)
    p = tpw / (tpw + fpw) if tpw + fpw else 0.0
    r = tpw / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    if not gold_slots and not slots:
        f1 = 1.0          # both empty: perfect agreement
    return 0.5 * i_ok + 0.5 * f1


def parse_markup(text):
    import ast as _a
    calls = []
    if not text:
        return calls
    for fm in re.finditer(r"<function=([\w.]+)>(.*?)</function>", text, re.S):
        name, body = fm.group(1), fm.group(2)
        params = {}
        for pm in re.finditer(r"<parameter=([\w.]+)>(.*?)</parameter>", body, re.S):
            k, v = pm.group(1), pm.group(2).strip()
            try:
                params[k] = json.loads(v)
            except Exception:
                try:
                    params[k] = _a.literal_eval(v)
                except Exception:
                    params[k] = v
        calls.append({name: params})
    return calls


def reward_snips_intent(response, gold):
    """Intent-only ablation: 1.0 iff predicted intent matches gold.
    Slot output is ignored entirely (response must still parse)."""
    intent, _ = parse_pred_snips(response)
    return 1.0 if intent == gold["intent"] else 0.0


def reward_bfcl(response, gold_item):
    """1.0 iff the parsed call passes the official checker against the gold."""
    from bfcl.ast_checker_official import ast_checker
    calls = parse_markup(response)
    try:
        v = ast_checker(gold_item["function"], calls, gold_item["ground_truth"])
        return 1.0 if v["valid"] else 0.0
    except Exception:
        return 0.0


# ------------------------------------------------------------------ SPHERE

class SphereHook:
    """Captures the input of down_proj (Phi) for selected layers, attached."""

    def __init__(self, model, layer_ids):
        self.stash = {}
        self.handles = []
        self.calls = {li: 0 for li in layer_ids}
        self.layer_ids = layer_ids
        for li in layer_ids:
            mod = model.model.layers[li].mlp.down_proj
            self.handles.append(mod.register_forward_pre_hook(self._make(li)))

    def _make(self, li):
        def hook(module, args=None, kwargs=None):
            inp = args[0] if args else (kwargs or {}).get("input")
            self.stash[li] = inp
            self.calls[li] += 1
            return None
        return hook

    def clear(self):
        self.stash.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


def sphere_grad_layer(phi_rows, alpha_mode):
    """Closed-form SPHERE penalty + dL/dPhi for one layer's detached [T, D]
    action-token rows.

    L = ||Phi^T Phi / T - alpha I||_F^2 with the toy protocol's two spaces:
    feature Gram A = Phi^T Phi / T when D <= T, sample Gram G = Phi Phi^T / T
    when D > T (identical Frobenius norm and trace; the isotropic target
    stays dimension D). alpha adaptive = Tr(A)/D detached, or fixed.
    """
    phi = phi_rows.detach().float()
    T, D = phi.shape
    trA = (phi * phi).sum() / T
    alpha = (trA / D).item() if alpha_mode == "adaptive" else float(alpha_mode)
    eye = torch.eye(phi.shape[-1] if D <= T else T, device=phi.device)
    if D <= T:
        A = phi.t() @ phi / T
        L = (A * A).sum() - 2.0 * alpha * trA + (alpha * alpha) * D
        dl = (4.0 / T) * (phi @ (A - alpha * eye))
    else:
        G = phi @ phi.t() / T
        L = (G * G).sum() - 2.0 * alpha * trA + (alpha * alpha) * D
        dl = (4.0 / T) * ((G - alpha * eye) @ phi)
    return L.item(), alpha, dl


# ------------------------------------------------------------------ data

def to_oai_tool(fd):
    props = dict(fd["parameters"]["properties"])
    req = fd["parameters"].get("required", [])
    schema = {"type": "object", "properties": {}, "required": req}
    for k, v in props.items():
        schema["properties"][k] = {"type": v.get("type", "string"),
                                   "description": v.get("description", "")}
    return {"type": "function",
            "function": {"name": fd["name"].replace(".", "_"),
                         "description": fd.get("description", ""),
                         "parameters": schema}}


def load_rows(path, reward_kind):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if reward_kind == "snips":
                # train.jsonl rows are [system, user, assistant-gold]; the
                # rollout prompt MUST exclude the assistant turn (gold leak)
                rows.append({"messages": d["messages"][:-1], "tools": None,
                             "gold": json.loads(d["messages"][2]["content"]),
                             "gold_item": None})
            else:
                rows.append({"messages": [
                    {"role": "system", "content": BFCL_SYSTEM},
                    {"role": "user", "content": d["question"]}],
                    "tools": [to_oai_tool(d["function"][0])],
                    "gold": None, "gold_item": d})
    return rows


# ------------------------------------------------------------------ trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True, help="base checkpoint path")
    ap.add_argument("--rollout-url", default="http://127.0.0.1:30001")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True,
                    choices=["last_adaptive", "all_hidden_adaptive", "last_fixed"])
    ap.add_argument("--reward", required=True, choices=["snips", "bfcl"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--gen-temp", type=float, default=1.0)
    ap.add_argument("--name-credit", type=float, default=0.25,
                    help="SNIPS shaping: TP credit for right-name-wrong-value "
                         "slots (FP weight is 1 - credit). 0.25 for the base "
                         "runs; continuation passes may raise it to keep "
                         "emission profitable once groups saturate.")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--train-pool", type=int, default=0,
                    help="restrict to a seeded subset of N prompts (0=all)")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sync-every", type=int, default=1)
    ap.add_argument("--microbatch", type=int, default=16,
                    help="physical microbatch size (memory sharding only; "
                         "PPO loss keeps the global token denominator and "
                         "SPHERE assembles the full logical batch)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save a numbered checkpoint every N steps "
                         "(0 = only the final save)")
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

    n_layers = len(model.model.layers)
    if args.arm == "all_hidden_adaptive":
        target_layers, alpha_mode = list(range(n_layers)), "adaptive"
    elif args.arm == "last_adaptive":
        target_layers, alpha_mode = [n_layers - 1], "adaptive"
    else:
        target_layers, alpha_mode = [n_layers - 1], 1.0

    rows = load_rows(args.data, args.reward)
    if args.train_pool:
        rng = random.Random(1000 + args.seed)
        rows = rng.sample(rows, min(args.train_pool, len(rows)))
    print(f"arm={args.arm} layers={len(target_layers)} alpha={alpha_mode} "
          f"rows={len(rows)} reward={args.reward} base={args.base}", flush=True)

    sphere = SphereHook(model, target_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    os.makedirs(args.out, exist_ok=True)
    evidence = {"arm": args.arm, "target_layers": target_layers,
               "n_target_layers": len(target_layers), "alpha_mode": str(alpha_mode),
               "rho": args.rho, "seed": args.seed, "group": args.group,
               "prompts_per_step": args.prompts_per_step, "lr": args.lr,
               "base": args.base, "reward": args.reward,
               "train_pool": args.train_pool, "data": args.data,
               "optimizer_steps": 0, "steps": 0,
               "lambda_stats": [], "reward_first10": None, "reward_last10": None}

    rewards_hist = []
    t0 = time.time()
    for step in range(args.steps):
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

        # ---- rollouts via sglang engine
        outs = rollout(args.rollout_url, prompt_ids, args.gen_temp,
                       args.max_new_tokens, G)

        rewards = []
        seqs = []          # token ids (prompt + re-encoded response), prompt_len
        for pi in range(P):
            for g in range(G):
                text = outs[pi][g]
                r = batch_rows[pi]
                if args.reward == "snips":
                    rewards.append(reward_snips(text, r["gold"],
                                                args.name_credit))
                else:
                    rewards.append(reward_bfcl(text, r["gold_item"]))
                resp_ids = tok(text, add_special_tokens=False)["input_ids"]
                seqs.append((prompt_ids[pi] + resp_ids, plens[pi]))
        rewards = torch.tensor(rewards, dtype=torch.float32).cuda()

        rg = rewards.view(P, G)
        adv = (rg - rg.mean(dim=1, keepdim=True)) / (rg.std(dim=1, keepdim=True) + 1e-4)
        adv = adv.view(-1)

        # ---- right-padded training batch
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

        # ---- pass 1: PPO microbatched forward/backward. The logical
        # minibatch is sharded only for memory; the token-mean loss keeps
        # the global denominator so gradients sum to the full-batch PPO
        # gradient. Detached action-token Phi rows are collected per chunk
        # for the SPHERE pass (toy protocol structure).
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
            # reference-kernel pad rows can carry NaN; real causal rows are finite
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

        # ---- pass 2: SPHERE — closed-form dL/dPhi on the assembled logical
        # minibatch, then a per-microbatch linear surrogate replay so the
        # penalty stays a property of the full batch while memory shards
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

        # ---- explicit gradient combination + single optimizer step
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

        # ---- push updated weights into the rollout engine
        if (step + 1) % args.sync_every == 0 or step == args.steps - 1:
            model.config.use_cache = True
            push_weights(args.rollout_url, model)
            model.config.use_cache = False

        # ---- periodic checkpoint (crash recovery / best-step selection)
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
             "lambda": lam,
             "sphere_loss": float(sph_loss),
             "alphas": alphas, "mean_reward": mr,
             "action_tokens": int(resp_mask.sum().item()),
             "replay_rel_l2": replay_rel_l2, "surr_ip_dev": surr_ip_dev})
        if step % 10 == 0 or step == args.steps - 1:
            print(f"step {step}/{args.steps} reward {mr:.3f} "
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
