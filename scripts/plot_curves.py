#!/usr/bin/env python3
"""Plot training curves (mean rollout reward vs step) from trainer evidence.

Scans the per-arm evidence files written by methods/sphere_grpo.py:

    results/<corpus>/<arm>/evidence.json

where <corpus> is one of the SNIPS splits (`official-split`,
`jointbert-split`). Each evidence.json holds the run config plus one record
per optimizer step (g_ppo, g_sphere, lambda, mean_reward, action_tokens, ...).

For every corpus with evidence, this script writes:

    results/<corpus>/<arm>/curves.csv   step,mean_reward (+100-step blocks)
    results/curves-<corpus>.png         one panel per arm: raw reward (thin),
                                         50-step rolling mean (thick), final
                                         intent accuracy annotated when the
                                         arm's eval-score.txt is present.

Usage:  python scripts/plot_curves.py [--results-dir results]
"""
import argparse, csv, glob, json, os


def load_arm(arm_dir):
    with open(os.path.join(arm_dir, "evidence.json")) as f:
        ev = json.load(f)
    steps = [r["step"] for r in ev["lambda_stats"]]
    rew = [r["mean_reward"] for r in ev["lambda_stats"]]
    acc = None
    score = os.path.join(arm_dir, "eval-score.txt")
    if os.path.exists(score):
        for line in open(score):
            if line.startswith("INTENT_ACC="):
                acc = line.strip().split("=")[1]
    return ev.get("arm", os.path.basename(arm_dir)), ev.get("steps", len(steps)), steps, rew, acc


def write_csv(arm_dir, steps, rew, roll):
    with open(os.path.join(arm_dir, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "mean_reward", "roll50"])
        w.writerows(zip(steps, rew, roll))
    with open(os.path.join(arm_dir, "blocks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_start", "block_mean_reward"])
        B = 100
        for i in range(0, len(rew), B):
            w.writerow([steps[i], sum(rew[i:i + B]) / len(rew[i:i + B])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corpora = sorted(d for d in glob.glob(os.path.join(args.results_dir, "*"))
                     if os.path.isdir(d)
                     and glob.glob(os.path.join(d, "*", "evidence.json")))
    for corpus_dir in corpora:
        corpus = os.path.basename(corpus_dir)
        arms = sorted(glob.glob(os.path.join(corpus_dir, "*", "evidence.json")))
        data = [load_arm(os.path.dirname(p)) for p in arms]
        if not data:
            continue
        fig, axes = plt.subplots(len(data), 1, figsize=(8, 2.2 * len(data)),
                                 sharex=True,
                                 squeeze=False)
        rolls = []
        for ax, (arm, nsteps, steps, rew, acc) in zip(axes[:, 0], data):
            ax.plot(steps, rew, lw=0.4, alpha=0.35, color="tab:blue")
            roll = [sum(rew[max(0, i - 49):i + 1]) / len(rew[max(0, i - 49):i + 1])
                    for i in range(len(rew))]
            rolls.append(roll)
            ax.plot(steps, roll, lw=1.6, color="tab:blue")
            ax.set_ylabel("reward")
            title = f"{arm}  ({nsteps} steps"
            if acc:
                title += f", final intent acc {acc}"
            ax.set_title(title + ")", fontsize=10)
            ax.grid(alpha=0.25)
        axes[-1, 0].set_xlabel("optimizer step")
        fig.suptitle(f"Training curves — SNIPS {corpus} (raw + 50-step rolling mean)",
                     fontsize=11)
        fig.tight_layout()
        out = os.path.join(args.results_dir, f"curves-{corpus}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out} ({len(data)} arms)")
        for arm_dir, (arm, _, steps, rew, _), roll in zip(
                [os.path.dirname(p) for p in arms], data, rolls):
            write_csv(arm_dir, steps, rew, roll)
            print(f"  {arm}: {len(steps)} steps, final rolling reward "
                  f"{sum(rew[-50:]) / 50:.3f}")


if __name__ == "__main__":
    main()
