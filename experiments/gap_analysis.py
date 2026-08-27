"""
Gap analysis: WHERE does CTW lose to GPT-2?

Trains CTW D=5 on WikiText-2 train, collects per-character losses on validation.
Analyses BPC by character position within words:

  Position 0  — first char after space/newline  (word-initial)
  Position 1  — second char in word
  Position 2  — third char
  ...
  Position 4+ — deep within a word

Key finding: the gap between CTW and GPT-2 is concentrated at word-initial
positions, where predicting the next character requires knowing which word is
being formed — a long-range dependency CTW cannot resolve.

Also plots a running BPC curve over the validation sequence.

Usage:
    python experiments/gap_analysis.py               # fast subset (500K train)
    python experiments/gap_analysis.py --full        # full 2M train chars
    python experiments/gap_analysis.py --no_gpt2     # CTW analysis only
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import math
import time
import glob
from collections import defaultdict
from datetime import datetime

from text_perplexity import normalize_text, load_wikitext2
from _paths import figure_for_result, result_path


# -----------------------------------------------------------------------
# CTW per-character evaluation
# -----------------------------------------------------------------------

def run_ctw_perchar(train_text: str, val_text: str,
                    depth: int, vocab: list[str]) -> list[float]:
    """
    Train CTW on train_text, return list of per-char log-losses on val_text.
    """
    from ctw.text_ctw import TextCTW

    ctw = TextCTW(depth=depth, vocab=vocab)

    print(f"  Training CTW D={depth} on {len(train_text):,} chars...")
    t0 = time.time()
    for t, ch in enumerate(train_text):
        ctx = list(train_text[max(0, t - depth):t])
        ctw.update(ch, ctx)
        if (t + 1) % 500_000 == 0:
            print(f"    [{t+1:,}/{len(train_text):,}]  {time.time()-t0:.0f}s")
    print(f"  Train done in {time.time()-t0:.1f}s")

    print(f"  Evaluating on {len(val_text):,} chars...")
    t0 = time.time()
    losses = []
    for t, ch in enumerate(val_text):
        ctx = list(val_text[max(0, t - depth):t])
        losses.append(ctw.log_loss(ch, ctx))
        if (t + 1) % 100_000 == 0:
            print(f"    [{t+1:,}/{len(val_text):,}]  "
                  f"BPC so far: {sum(losses)/(t+1):.4f}  {time.time()-t0:.0f}s")
    print(f"  Eval done in {time.time()-t0:.1f}s  |  BPC = {sum(losses)/len(losses):.4f}")
    return losses


# -----------------------------------------------------------------------
# GPT-2 per-token → per-character losses
# -----------------------------------------------------------------------

def run_gpt2_perchar(val_text: str, model_name: str = "gpt2",
                     max_chars: int = 0) -> list[float | None]:
    """
    Returns a list (same length as val_text) where entry t is the approximate
    log-loss attributed to character t.

    Each token's NLL is spread uniformly across its characters.
    Entries that fall outside the tokenized range are None.
    """
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError:
        print("  torch/transformers not installed — skipping GPT-2 per-char analysis.")
        return []

    if max_chars > 0:
        val_text = val_text[:max_chars]

    print(f"  Loading {model_name}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model     = GPT2LMHeadModel.from_pretrained(model_name).eval().float()

    encodings  = tokenizer(val_text, return_tensors="pt")
    input_ids  = encodings.input_ids
    n_tokens   = input_ids.size(1)

    # Map token index → character start position
    token_char_start = []
    offset = 0
    for tok_id in input_ids[0].tolist():
        token_str = tokenizer.decode([tok_id])
        token_char_start.append(offset)
        offset += len(token_str)

    max_length, stride = 1024, 512
    per_token_nll = [None] * n_tokens

    t0 = time.time()
    print(f"  Running GPT-2 inference ({n_tokens:,} tokens)...")
    for i in range(0, n_tokens, stride):
        begin      = max(i + stride - max_length, 0)
        end        = min(i + stride, n_tokens)
        chunk      = input_ids[:, begin:end]
        target_len = end - i

        with torch.no_grad():
            logits       = model(chunk).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()

            # Per-token NLL
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_nll = -log_probs.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1)[0]   # shape: [seq_len - 1]

        # Map to the 'target' tokens (positions i..end-1)
        target_nlls = token_nll[-target_len:].tolist()
        for j, nll_val in enumerate(target_nlls):
            tok_idx = i + j
            if tok_idx < n_tokens and math.isfinite(nll_val):
                per_token_nll[tok_idx] = nll_val / math.log(2)   # nats → bits

        if (i // stride + 1) % 20 == 0:
            print(f"    [{min(i+stride, n_tokens):,}/{n_tokens:,} tokens]  "
                  f"{time.time()-t0:.0f}s")

    # Distribute token loss to characters (uniform over token length)
    char_losses_gpt2 = [None] * len(val_text)
    for tok_idx, (char_start, nll_bits) in enumerate(
            zip(token_char_start, per_token_nll)):
        if nll_bits is None:
            continue
        tok_len = len(tokenizer.decode([input_ids[0][tok_idx].item()]))
        if tok_len == 0:
            continue
        per_char = nll_bits / tok_len
        for c in range(tok_len):
            pos = char_start + c
            if pos < len(char_losses_gpt2):
                char_losses_gpt2[pos] = per_char

    return char_losses_gpt2


# -----------------------------------------------------------------------
# Positional analysis
# -----------------------------------------------------------------------

def word_positions(text: str) -> list[int]:
    """
    For each char, return its 0-based position within its word.
    Spaces and newlines get position = -1.
    """
    positions = []
    pos = 0
    for ch in text:
        if ch in (' ', '\n'):
            positions.append(-1)
            pos = 0
        else:
            positions.append(pos)
            pos += 1
    return positions


def bpc_by_word_position(
    losses: list[float],
    positions: list[int],
    max_pos: int = 6,
) -> dict[str, float]:
    """Average BPC for each word position (0, 1, ..., max_pos-1, max_pos+)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for loss, pos in zip(losses, positions):
        if pos < 0:
            continue   # skip spaces
        key = str(pos) if pos < max_pos else f"{max_pos}+"
        buckets[key].append(loss)
    return {k: sum(v) / len(v) for k, v in buckets.items()}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",      action="store_true",
                        help="Use full training set (2M chars). Default: 500K (faster).")
    parser.add_argument("--depth",     type=int, default=5,
                        help="CTW depth for analysis (default: 5)")
    parser.add_argument("--no_gpt2",  action="store_true")
    parser.add_argument("--gpt2_model", default="gpt2")
    parser.add_argument("--out",       default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = result_path("gap_analysis_results.json")

    print("Loading WikiText-2...")
    train_raw, val_raw = load_wikitext2()
    train_text = normalize_text(train_raw)
    val_text   = normalize_text(val_raw)

    max_train = 0 if args.full else 500_000
    max_val   = 0 if args.full else 150_000

    if max_train > 0:
        train_text = train_text[:max_train]
    if max_val > 0:
        val_text = val_text[:max_val]

    vocab = sorted(set(train_text + val_text))
    print(f"  Using train: {len(train_text):,} chars  |  val: {len(val_text):,} chars")

    # ---- CTW per-char losses ----
    ctw_losses = run_ctw_perchar(train_text, val_text, args.depth, vocab)

    # ---- GPT-2 per-char losses ----
    gpt2_losses = []
    if not args.no_gpt2:
        gpt2_losses = run_gpt2_perchar(val_text, model_name=args.gpt2_model)

    # ---- Positional analysis ----
    positions = word_positions(val_text)

    ctw_by_pos  = bpc_by_word_position(ctw_losses, positions)
    gpt2_by_pos = (bpc_by_word_position(
                       [l for l in gpt2_losses if l is not None],
                       [p for l, p in zip(gpt2_losses, positions) if l is not None],
                   ) if gpt2_losses else {})

    # ---- Running BPC (window=5000 chars) ----
    window = 5_000
    running_ctw = []
    for i in range(0, len(ctw_losses) - window, window // 2):
        running_ctw.append({
            "pos":  i + window // 2,
            "bpc":  sum(ctw_losses[i:i+window]) / window,
        })

    # ---- Save ----
    results = {
        "timestamp":      datetime.now().isoformat(),
        "ctw_depth":      args.depth,
        "train_chars":    len(train_text),
        "val_chars":      len(val_text),
        "ctw_global_bpc": sum(ctw_losses) / len(ctw_losses),
        "ctw_by_word_position":  ctw_by_pos,
        "gpt2_by_word_position": gpt2_by_pos,
        "running_bpc_ctw":       running_ctw,
    }
    if gpt2_losses:
        valid = [l for l in gpt2_losses if l is not None]
        results["gpt2_global_bpc"] = sum(valid) / len(valid) if valid else None

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.out}")

    # ---- Print summary ----
    print(f"\n{'='*55}")
    print(f"BPC by word position  (CTW D={args.depth}  vs  {args.gpt2_model})")
    print(f"{'='*55}")
    print(f"{'Position':<12}  {'CTW BPC':>9}  {'GPT-2 BPC':>10}  {'Gap':>8}")
    print("-" * 46)
    pos_keys = sorted(ctw_by_pos.keys(),
                      key=lambda k: int(k.replace("+", "")) if k != "6+" else 99)
    for k in pos_keys:
        ctw_b = ctw_by_pos.get(k, float("nan"))
        gpt_b = gpt2_by_pos.get(k, float("nan"))
        gap   = ctw_b - gpt_b if not math.isnan(gpt_b) else float("nan")
        label = f"pos {k} {'(word start)' if k=='0' else ''}"
        print(f"{label:<18}  {ctw_b:>9.4f}  "
              f"{gpt_b:>10.4f}  "
              f"{gap:>+8.4f}")

    # ---- Plot ----
    _plot(results, figure_for_result(args.out))


def _plot(results: dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: BPC by word position
    pos_keys = sorted(results["ctw_by_word_position"].keys(),
                      key=lambda k: int(k.replace("+", "")) if "+" not in k else 99)
    x = range(len(pos_keys))
    ctw_vals = [results["ctw_by_word_position"][k] for k in pos_keys]
    ax1.bar([xi - 0.2 for xi in x], ctw_vals, width=0.35,
            color="#1565C0", alpha=0.8, label=f"CTW D={results['ctw_depth']}")

    if results.get("gpt2_by_word_position"):
        gpt2_vals = [results["gpt2_by_word_position"].get(k, float("nan")) for k in pos_keys]
        ax1.bar([xi + 0.2 for xi in x], gpt2_vals, width=0.35,
                color="#2E7D32", alpha=0.8, label="GPT-2 small")

    ax1.axhline(results["ctw_global_bpc"], color="#1565C0",
                linestyle="--", linewidth=1, alpha=0.5, label="CTW global BPC")
    if results.get("gpt2_global_bpc"):
        ax1.axhline(results["gpt2_global_bpc"], color="#2E7D32",
                    linestyle="--", linewidth=1, alpha=0.5, label="GPT-2 global BPC")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"pos {k}" for k in pos_keys], rotation=30, ha="right")
    ax1.set_ylabel("Bits per character (BPC) ↓", fontsize=10)
    ax1.set_title("BPC by position within word", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25, axis="y")

    # Panel 2: Running BPC over sequence
    if results.get("running_bpc_ctw"):
        xs = [r["pos"] for r in results["running_bpc_ctw"]]
        ys = [r["bpc"] for r in results["running_bpc_ctw"]]
        ax2.plot(xs, ys, color="#1565C0", linewidth=1.2, alpha=0.8,
                 label=f"CTW D={results['ctw_depth']} (running, w=5K)")
        ax2.axhline(results["ctw_global_bpc"], color="#1565C0",
                    linestyle="--", linewidth=1, alpha=0.5)
        if results.get("gpt2_global_bpc"):
            ax2.axhline(results["gpt2_global_bpc"], color="#2E7D32",
                        linestyle=":", linewidth=1.5, label="GPT-2 small (global)")

    ax2.set_xlabel("Character position in validation set", fontsize=10)
    ax2.set_ylabel("BPC (5K-char window)", fontsize=10)
    ax2.set_title("Running BPC over validation sequence", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    plt.suptitle("CTW vs. GPT-2: Where is the Gap Largest?", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
