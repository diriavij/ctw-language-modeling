"""
Fine-tune GPT-2 small on normalized WikiText-2 (28-char alphabet).

Goal: determine how much of the CTW ↔ GPT-2 gap is due to distribution
mismatch (pretrained on rich text, evaluated on normalized text) vs.
genuine long-range dependency advantage.

Experiment:
    1. Evaluate pretrained GPT-2 on normalized val  → baseline BPC
    2. Fine-tune on normalized train (same 2M chars as CTW training)
    3. Evaluate after each epoch                    → BPC curve
    4. Compare with CTW best (D=7) and pretrained baseline

Expected outcomes:
  - If fine-tuned BPC ≈ pretrained BPC → pretrained comparison was already fair
  - If fine-tuned BPC << pretrained BPC → gap is larger than reported (CTW is worse)

Usage:
    python experiments/finetune_gpt2.py
    python experiments/finetune_gpt2.py --epochs 5 --lr 3e-5
    python experiments/finetune_gpt2.py --model gpt2-large
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import math
import time
from datetime import datetime

from text_perplexity import normalize_text, load_wikitext2, run_gpt2


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

def make_dataloader(token_ids, block_size: int, batch_size: int, shuffle: bool = True):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class BlockDataset(Dataset):
        def __init__(self, ids):
            ids_t = torch.tensor(ids, dtype=torch.long)
            # Drop remainder so all chunks are exactly block_size
            n = (len(ids_t) // block_size) * block_size
            self.blocks = ids_t[:n].view(-1, block_size)

        def __len__(self):
            return len(self.blocks)

        def __getitem__(self, idx):
            return self.blocks[idx]

    ds     = BlockDataset(token_ids)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    print(f"    {len(ds):,} blocks × {block_size} tokens  |  {len(loader):,} steps/epoch")
    return loader


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------

def finetune(
    train_text: str,
    model_name: str,
    epochs: int,
    lr: float,
    block_size: int,
    batch_size: int,
    grad_accum: int,
    save_dir: str,
) -> None:
    import torch
    from torch.optim import AdamW
    from transformers import (
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        get_linear_schedule_with_warmup,
    )

    print(f"  Loading {model_name}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model     = GPT2LMHeadModel.from_pretrained(model_name).float()

    print(f"  Tokenizing {len(train_text):,} chars...")
    train_ids = tokenizer.encode(train_text)
    print(f"  → {len(train_ids):,} tokens  ({len(train_text)/len(train_ids):.2f} chars/token)")

    loader = make_dataloader(train_ids, block_size, batch_size)

    total_steps    = (len(loader) // grad_accum) * epochs
    warmup_steps   = max(50, total_steps // 20)
    optimizer      = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler      = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"  Fine-tuning: {epochs} epochs × {len(loader)} steps, lr={lr}, "
          f"grad_accum={grad_accum}, warmup={warmup_steps}")

    model.train()
    global_step = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss   = 0.0
        epoch_steps  = 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            outputs = model(input_ids=batch, labels=batch)
            loss    = outputs.loss / grad_accum
            loss.backward()
            epoch_loss  += outputs.loss.item()
            epoch_steps += 1

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    avg = epoch_loss / epoch_steps
                    print(f"    epoch {epoch}  step {global_step:,}  "
                          f"loss={avg:.4f}  {time.time()-t_start:.0f}s")

        avg_loss = epoch_loss / epoch_steps
        print(f"  ── Epoch {epoch} done  avg_loss={avg_loss:.4f}  "
              f"{time.time()-t_start:.0f}s elapsed")

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  Checkpoint saved → {ckpt_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="gpt2")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--block_size", type=int,   default=512)
    parser.add_argument("--batch_size", type=int,   default=4,
                        help="Tokens per gradient step (before accumulation)")
    parser.add_argument("--grad_accum", type=int,   default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--out",        default=None)
    parser.add_argument("--plot_only",  default=None, metavar="JSON",
                        help="Skip training; just plot an existing results JSON.")
    args = parser.parse_args()

    exp_dir = os.path.dirname(__file__)

    if args.plot_only:
        with open(args.plot_only) as f:
            results = json.load(f)
        out_png = args.plot_only.replace(".json", ".png")
        _plot(results, out_png)
        return

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(exp_dir, f"finetune_results_{ts}.json")

    save_dir = os.path.join(exp_dir, "gpt2_finetuned")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Data ----
    print("Loading WikiText-2 (normalized)...")
    train_raw, val_raw = load_wikitext2()
    train_text = normalize_text(train_raw)
    val_text   = normalize_text(val_raw)
    print(f"  Train: {len(train_text):,} chars  |  Val: {len(val_text):,} chars")

    results = {
        "timestamp":  datetime.now().isoformat(),
        "model":      args.model,
        "epochs":     args.epochs,
        "lr":         args.lr,
        "block_size": args.block_size,
        "evaluations": [],
    }

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---- Pretrained baseline ----
    print("\n" + "=" * 55)
    print(f"Pretrained baseline ({args.model})")
    print("=" * 55)
    r = run_gpt2(val_text, model_name=args.model)
    if r:
        r["epoch"] = 0
        r["label"] = "pretrained"
        results["evaluations"].append(r)
        print(f"  Pretrained BPC/char = {r['bpc_per_char']:.4f}  "
              f"PPL/token = {r['ppl_token']:.2f}")
        save()

    # ---- Fine-tune ----
    print("\n" + "=" * 55)
    print("Fine-tuning on normalized WikiText-2")
    print("=" * 55)
    finetune(
        train_text  = train_text,
        model_name  = args.model,
        epochs      = args.epochs,
        lr          = args.lr,
        block_size  = args.block_size,
        batch_size  = args.batch_size,
        grad_accum  = args.grad_accum,
        save_dir    = save_dir,
    )

    # ---- Evaluate each checkpoint ----
    print("\n" + "=" * 55)
    print("Evaluating fine-tuned checkpoints")
    print("=" * 55)
    for epoch in range(1, args.epochs + 1):
        ckpt = os.path.join(save_dir, f"epoch_{epoch}")
        if not os.path.exists(ckpt):
            print(f"  Checkpoint epoch_{epoch} not found, skipping.")
            continue
        print(f"\n  Epoch {epoch} checkpoint: {ckpt}")
        r = run_gpt2(val_text, model_name=ckpt)
        if r:
            r["epoch"] = epoch
            r["label"] = f"finetuned_epoch_{epoch}"
            results["evaluations"].append(r)
            print(f"  BPC/char = {r['bpc_per_char']:.4f}  PPL/token = {r['ppl_token']:.2f}")
            save()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("FINE-TUNING SUMMARY")
    print("=" * 60)
    print(f"{'Label':<25}  {'BPC/char':>9}  {'PPL/token':>10}  {'ΔBPC':>8}")
    print("-" * 58)
    base_bpc = None
    for r in results["evaluations"]:
        bpc = r.get("bpc_per_char")
        if bpc is None or not math.isfinite(bpc):
            continue
        if base_bpc is None:
            base_bpc = bpc
        delta = bpc - base_bpc
        print(f"{r['label']:<25}  {bpc:>9.4f}  {r['ppl_token']:>10.2f}  "
              f"{delta:>+8.4f}")

    print(f"\nResults saved → {args.out}")
    print(f"Checkpoints  → {save_dir}/")

    _plot(results, args.out.replace(".json", ".png"))


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def _plot(results: dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import glob as _glob
    except ImportError:
        return

    evals = [r for r in results["evaluations"]
             if r.get("bpc_per_char") is not None and math.isfinite(r["bpc_per_char"])]
    if not evals:
        return

    epochs = [r["epoch"] for r in evals]
    bpcs   = [r["bpc_per_char"] for r in evals]

    # Try to load CTW best BPC from full_results_*.json
    ctw_bpc = None
    exp_dir = os.path.dirname(__file__)
    ctw_files = sorted(_glob.glob(os.path.join(exp_dir, "full_results_*.json")))
    if ctw_files:
        try:
            with open(ctw_files[-1]) as f:
                ctw_data = json.load(f)
            ctw_results = ctw_data.get("ctw_results", [])
            if ctw_results:
                ctw_bpc = min(r["bpc"] for r in ctw_results)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # BPC curve
    ax.plot(epochs, bpcs, "o-", color="#1565C0", linewidth=2.5,
            markersize=8, zorder=4, label=f"GPT-2 small ({results['model']})")
    for ep, b in zip(epochs, bpcs):
        ax.annotate(f"{b:.4f}", (ep, b),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color="#1565C0")

    # CTW best line
    if ctw_bpc is not None:
        ax.axhline(ctw_bpc, color="#E65100", linestyle="--", linewidth=2,
                   label=f"CTW best (D=7)  {ctw_bpc:.4f}", zorder=3)
        ax.text(max(epochs) - 0.05, ctw_bpc + 0.015,
                f"{ctw_bpc:.4f}", ha="right", fontsize=8, color="#E65100")

        # Gap arrow between fine-tuned best and CTW
        best_ft  = min(bpcs[1:]) if len(bpcs) > 1 else bpcs[-1]
        best_ep  = epochs[bpcs.index(best_ft)]
        gap      = ctw_bpc - best_ft
        mid_y    = (ctw_bpc + best_ft) / 2
        x_ann    = max(epochs) + 0.25
        ax.annotate("", xy=(x_ann, ctw_bpc), xytext=(x_ann, best_ft),
                    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
        ax.text(x_ann + 0.08, mid_y,
                f"Δ={gap:+.3f}\n({2**gap:.2f}×)",
                va="center", ha="left", fontsize=8, color="#333")

    # Shannon entropy band
    SHANNON_CENTRAL, SHANNON_LO, SHANNON_HI = 1.1, 1.0, 1.3
    ax.axhline(SHANNON_CENTRAL, color="#B71C1C", linestyle="-.",
               linewidth=1.5, alpha=0.85,
               label=f"Shannon H(English) ≈ {SHANNON_CENTRAL} bpc", zorder=2)
    ax.axhspan(SHANNON_LO, SHANNON_HI, alpha=0.06, color="#B71C1C", zorder=1)

    ax.set_xlabel("Epoch  (0 = pretrained)", fontsize=10)
    ax.set_ylabel("Bits per character (BPC) ↓", fontsize=10)
    ax.set_title(
        f"GPT-2 Fine-Tuning on Normalized WikiText-2\n"
        f"lr={results['lr']}, block_size={results['block_size']}",
        fontsize=11,
    )
    ax.set_xticks(epochs)
    ax.set_xticklabels(["pretrained\n(epoch 0)"] + [f"epoch {e}" for e in epochs[1:]])
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25, linestyle="--")

    all_bpc = bpcs + ([ctw_bpc] if ctw_bpc else []) + [SHANNON_LO]
    ax.set_ylim(min(all_bpc) - 0.1, max(all_bpc) + 0.25)
    ax.set_xlim(-0.3, max(epochs) + 0.7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
