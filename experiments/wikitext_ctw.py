"""
Experiment: CTW BPC on WikiText-2 for different depths D.

Trains TextCTW on the WikiText-2 train split, evaluates BPC on validation.
Runs for D = 3, 5, 7, 10 and prints a summary table.

Usage:
    python experiments/wikitext_ctw.py
    python experiments/wikitext_ctw.py --max_train 200000 --max_val 50000
    python experiments/wikitext_ctw.py --max_train 0 --max_val 0   # full dataset
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import math
import time
import json
import re
import unicodedata

from ctw.text_ctw import TextCTW
from _paths import result_path


def normalize_text(text: str) -> str:
    """
    Normalize to a small alphabet: lowercase letters, space, newline.
    Result: |A| = 28 instead of 65-395.

    Why: max useful CTW depth ≈ log_A(N_train / min_visits).
    With A=65 and 2M chars: D_max ≈ 2.9  →  D>3 can't generalize.
    With A=28 and 2M chars: D_max ≈ 4.0  →  D=5 shows real improvement.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # Map anything that is not a letter, space or newline to a space
    text = re.sub(r'[^a-z \n]', ' ', text)
    # Collapse runs of spaces so they don't inflate the vocab's null contexts
    text = re.sub(r' {2,}', ' ', text)
    return text


def load_wikitext2():
    """Load WikiText-2 train and validation splits."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(ds["train"]["text"])
    val_text = "\n".join(ds["validation"]["text"])
    return train_text, val_text


def build_vocab(text: str) -> list[str]:
    """Return sorted list of unique characters."""
    return sorted(set(text))


def run_ctw_experiment(train_text: str, val_text: str, depth: int, vocab: list[str]) -> dict:
    """
    Train TextCTW on train_text, evaluate BPC on val_text.
    Returns dict with bpc, perplexity, timing info.
    """
    ctw = TextCTW(depth=depth, vocab=vocab)

    # --- Train ---
    t0 = time.time()
    for t, ch in enumerate(train_text):
        ctx = list(train_text[max(0, t - depth) : t])
        ctw.update(ch, ctx)
        if (t + 1) % 50_000 == 0:
            elapsed = time.time() - t0
            rate = (t + 1) / elapsed
            print(f"  D={depth}: train {t+1:,}/{len(train_text):,} "
                  f"({rate:.0f} char/s, {elapsed:.1f}s)")
    train_time = time.time() - t0

    # --- Evaluate ---
    t0 = time.time()
    total_bits = 0.0
    for t, ch in enumerate(val_text):
        ctx = list(val_text[max(0, t - depth) : t])
        total_bits += ctw.log_loss(ch, ctx)
        if (t + 1) % 50_000 == 0:
            bpc_so_far = total_bits / (t + 1)
            print(f"  D={depth}: eval {t+1:,}/{len(val_text):,} "
                  f"(BPC so far: {bpc_so_far:.4f})")
    eval_time = time.time() - t0

    bpc = total_bits / len(val_text)
    return {
        "depth": depth,
        "bpc": bpc,
        "perplexity": 2 ** bpc,
        "train_chars": len(train_text),
        "eval_chars": len(val_text),
        "vocab_size": len(vocab),
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="CTW BPC on WikiText-2")
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 5, 7, 10],
                        help="Context depths to evaluate (default: 3 5 7 10)")
    parser.add_argument("--max_train", type=int, default=200_000,
                        help="Max training chars (0 = full, default: 200000)")
    parser.add_argument("--max_val", type=int, default=50_000,
                        help="Max validation chars (0 = full, default: 50000)")
    args = parser.parse_args()

    print("Loading WikiText-2...")
    train_text, val_text = load_wikitext2()
    print(f"  Full train: {len(train_text):,} chars")
    print(f"  Full val:   {len(val_text):,} chars")

    print("Normalizing text (NFKC → lowercase → ASCII)...")
    train_text = normalize_text(train_text)
    val_text   = normalize_text(val_text)
    print(f"  Normalized train: {len(train_text):,} chars")
    print(f"  Normalized val:   {len(val_text):,} chars")

    if args.max_train > 0:
        train_text = train_text[:args.max_train]
    if args.max_val > 0:
        val_text = val_text[:args.max_val]

    print(f"  Using train: {len(train_text):,} chars")
    print(f"  Using val:   {len(val_text):,} chars")

    vocab = build_vocab(train_text + val_text)
    print(f"  Vocab size: {len(vocab)}")

    results = []
    for d in args.depths:
        print(f"\n{'='*50}")
        print(f"Running CTW with D = {d}")
        print(f"{'='*50}")
        r = run_ctw_experiment(train_text, val_text, d, vocab)
        results.append(r)
        print(f"  BPC = {r['bpc']:.4f}, Perplexity = {r['perplexity']:.2f}, "
              f"Train: {r['train_time_s']}s, Eval: {r['eval_time_s']}s")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY: CTW on WikiText-2")
    print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,} chars, "
          f"Vocab: {len(vocab)}")
    print(f"{'='*60}")
    print(f"{'D':>4}  {'BPC':>8}  {'Perplexity':>12}  {'Train(s)':>10}  {'Eval(s)':>10}")
    print("-" * 52)
    for r in results:
        print(f"{r['depth']:>4}  {r['bpc']:>8.4f}  {r['perplexity']:>12.2f}  "
              f"{r['train_time_s']:>10.1f}  {r['eval_time_s']:>10.1f}")

    # Save results
    out_path = result_path("wikitext_ctw_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
