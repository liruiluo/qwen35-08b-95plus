#!/usr/bin/env python3
"""Generic single-GPU SFT trainer for small causal LMs (Qwen3.5-0.8B class).

Pure PyTorch + transformers only (no accelerate / peft / trl), so it runs in a
bare inference venv that already has torch + transformers.

Data format: JSONL, one {"messages": [{"role","content"}, ...]} per line.
Loss is computed on assistant tokens only (prompt tokens are masked).
Non-thinking chat format: the Qwen template is rendered with
enable_thinking=False so the trained target contains no <think> blocks.

Usage:
  python train_sft.py --data train.jsonl --model /path/Qwen3.5-0.8B \
      --out ckpt/snips-sft --epochs 3 --bsz 8 --accum 4 --lr 1e-5
"""
import argparse, json, math, os, random, time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


class SFTData(Dataset):
    def __init__(self, path, tok, max_len):
        self.rows = []
        skipped = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msgs = d["messages"]
                tools = d.get("tools")
                kw = dict(tokenize=False, enable_thinking=False)
                kw_p = dict(kw, add_generation_prompt=True)
                kw_f = dict(kw, add_generation_prompt=False)
                if tools:
                    kw_p["tools"] = tools
                    kw_f["tools"] = tools
                prompt = tok.apply_chat_template(msgs[:-1], **kw_p)
                full = tok.apply_chat_template(msgs, **kw_f)
                if not full.startswith(prompt):
                    # rare template edge: fall back to plain concatenation
                    skipped += 1
                    continue
                p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                f_ids = tok(full, add_special_tokens=False)["input_ids"]
                if len(f_ids) > max_len:
                    skipped += 1
                    continue
                labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
                self.rows.append((f_ids, labels))
        print(f"dataset: {len(self.rows)} usable, {skipped} skipped")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, labels = self.rows[i]
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return (torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--save-every", type=int, default=0, help="save every N steps (0=only end)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = SFTData(args.data, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=True, num_workers=2,
                    collate_fn=lambda b: collate(b, tok.pad_token_id or tok.eos_token_id))

    total_steps = math.ceil(len(dl) / args.accum) * args.epochs if args.epochs >= 1 \
        else math.ceil(len(dl) * args.epochs / args.accum)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_steps * args.warmup), total_steps)

    os.makedirs(args.out, exist_ok=True)
    step = 0
    t0 = time.time()
    micro = 0
    done = False
    epoch = 0
    while not done:
        epoch += 1
        for batch in dl:
            input_ids, labels, attn = (x.cuda() for x in batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss / args.accum
            loss.backward()
            micro += 1
            if micro % args.accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {out.loss.item():.4f} gn {gn.item():.2f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)
                if args.save_every and step % args.save_every == 0:
                    save(model, tok, os.path.join(args.out, f"step{step}"))
            if step >= total_steps:
                done = True
                break
        if args.epochs > 0 and epoch >= math.ceil(args.epochs):
            done = True

    save(model, tok, args.out)
    print(f"DONE steps={step} wall={time.time()-t0:.0f}s -> {args.out}")


def save(model, tok, path):
    os.makedirs(path, exist_ok=True)
    model.config.use_cache = True
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=sd)
    tok.save_pretrained(path)
    model.config.use_cache = False
    print(f"saved -> {path}", flush=True)


if __name__ == "__main__":
    main()
