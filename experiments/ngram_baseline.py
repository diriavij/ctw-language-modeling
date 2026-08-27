"""
N-gram baseline on WikiText-2 (character-level).

Fast pure-Python implementation — no NLTK. Uses KT (Dirichlet-1/2) smoothing
with backoff (same estimator as CTW leaf nodes). About 100-500x faster than
nltk.lm for evaluation because log-probs are precomputed into a lookup table.

Note on comparison with CTW:
  - N-gram of order N uses exactly the same KT estimator as CTW leaf nodes,
    but at a *fixed* context depth (order-1 chars).
  - CTW tree-weights ALL depths 0..D; this is the only structural difference.
  - Order N ↔ context depth N-1. Default [4,6,8,11] matches CTW depths [3,5,7,10].

Usage:
    python experiments/ngram_baseline.py
    python experiments/ngram_baseline.py --orders 4 6 8 11
    python experiments/ngram_baseline.py --out experiments/ngram_results.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import math
import time
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime


# -----------------------------------------------------------------------
# Text normalization (same as text_perplexity.py)
# -----------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r'[^a-z \n]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


def load_wikitext2():
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return "\n".join(ds["train"]["text"]), "\n".join(ds["validation"]["text"])


# -----------------------------------------------------------------------
# Fast character n-gram model
# -----------------------------------------------------------------------

class FastCharNgramKT:
    """
    Character-level n-gram LM with KT smoothing + backoff.

    KT estimator:
        P(w | ctx) = (C(ctx, w) + 0.5) / (C(ctx) + 0.5 * A)
    Backoff: if ctx not found, try ctx[1:], then ctx[2:], ..., then '' (unigram).

    Fit:  O(N * order)
    Eval: O(|val|)  — log-probs precomputed into a lookup table.
    """

    def __init__(self, order: int, vocab: list[str], pad: str = '\x00'):
        self.order = order
        self.vocab = vocab
        self.A = len(vocab)
        self.PAD = pad
        self._counts: dict[str, Counter] = {}
        self._totals: dict[str, int] = {}

    # ------------------------------------------------------------------

    def fit(self, train_text: str) -> None:
        order = self.order
        PAD = self.PAD
        padded = PAD * (order - 1) + train_text
        N = len(padded)

        raw: dict[str, Counter] = defaultdict(Counter)

        t0 = time.time()
        for i in range(order - 1, N):
            word = padded[i]
            for ctx_len in range(0, order):          # ctx_len = 0..order-1
                ctx = padded[i - ctx_len : i]        # rightmost ctx_len chars
                raw[ctx][word] += 1

            done = i - order + 2
            if done % 500_000 == 0:
                print(f"    [{done:,}/{len(train_text):,}]  {time.time()-t0:.0f}s")

        self._counts = dict(raw)
        self._totals = {ctx: sum(c.values()) for ctx, c in raw.items()}
        print(f"    Unique contexts stored: {len(self._counts):,}")

    # ------------------------------------------------------------------

    def _log_prob_one(self, word: str, full_ctx: str) -> float:
        """log2 P(word | full_ctx) with KT smoothing + suffix backoff."""
        for n in range(len(full_ctx), -1, -1):
            ctx = full_ctx[-n:] if n > 0 else ''
            c_table = self._counts.get(ctx)
            if c_table is not None:
                count = c_table.get(word, 0)
                total = self._totals[ctx]
                return math.log2((count + 0.5) / (total + 0.5 * self.A))
        return math.log2(1.0 / self.A)

    # ------------------------------------------------------------------

    def eval_bpc(self, val_text: str) -> float:
        """
        Compute BPC on val_text.

        Step 1 — precompute log-probs for every unique (ctx, char) pair
                  in val_text (avoids recomputing repeated pairs).
        Step 2 — accumulate in a single O(|val|) pass.
        """
        order = self.order
        PAD = self.PAD
        padded_val = PAD * (order - 1) + val_text

        # Step 1: precompute unique pairs
        print(f"  Precomputing lookup table...", end='', flush=True)
        t0 = time.time()
        unique_pairs: set[tuple[str, str]] = set()
        for i in range(order - 1, len(padded_val)):
            ctx  = padded_val[i - (order - 1) : i]
            word = padded_val[i]
            unique_pairs.add((ctx, word))

        log_prob_table: dict[tuple[str, str], float] = {}
        for ctx, word in unique_pairs:
            log_prob_table[(ctx, word)] = self._log_prob_one(word, ctx)
        print(f" {len(unique_pairs):,} pairs  ({time.time()-t0:.1f}s)")

        # Step 2: accumulate
        t0 = time.time()
        total_bits = 0.0
        for i in range(order - 1, len(padded_val)):
            ctx  = padded_val[i - (order - 1) : i]
            word = padded_val[i]
            total_bits -= log_prob_table[(ctx, word)]

            done = i - order + 2
            if done % 50_000 == 0:
                print(f"    [{done:,}/{len(val_text):,}]  "
                      f"BPC so far: {total_bits/done:.4f}  "
                      f"{time.time()-t0:.0f}s")

        return total_bits / len(val_text)


# -----------------------------------------------------------------------
# Experiment driver
# -----------------------------------------------------------------------

def run_ngram(train_text: str, val_text: str, order: int, vocab: list[str]) -> dict:
    model = FastCharNgramKT(order=order, vocab=vocab)

    print(f"  Fitting {order}-gram KT model ({len(train_text):,} chars)...")
    t0 = time.time()
    model.fit(train_text)
    train_time = time.time() - t0
    print(f"  Fit done in {train_time:.1f}s")

    print(f"  Evaluating on {len(val_text):,} chars...")
    t0 = time.time()
    bpc = model.eval_bpc(val_text)
    eval_time = time.time() - t0
    print(f"  Eval done in {eval_time:.1f}s")

    return {
        "order":         order,
        "context_depth": order - 1,
        "bpc":           bpc,
        "perplexity":    2 ** bpc,
        "train_time_s":  round(train_time, 1),
        "eval_time_s":   round(eval_time, 1),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orders", type=int, nargs="+", default=[4, 6, 8, 11],
        help="N-gram orders. Default [4,6,8,11] matches CTW depths [3,5,7,10]."
    )
    parser.add_argument("--out", default=None,
                        help="Output JSON (default: experiments/ngram_results.json)")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(__file__), "ngram_results.json")

    print("Loading WikiText-2 (full dataset)...")
    train_raw, val_raw = load_wikitext2()
    train_text = normalize_text(train_raw)
    val_text   = normalize_text(val_raw)
    vocab = sorted(set(train_text + val_text))
    print(f"  Train: {len(train_text):,} chars  |  Val: {len(val_text):,} chars  "
          f"|  Vocab: {len(vocab)}")

    results = {
        "timestamp":     datetime.now().isoformat(),
        "dataset":       "WikiText-2 (full, normalized to 28-char alphabet)",
        "smoothing":     "KT (Dirichlet-1/2) + suffix backoff",
        "ngram_results": [],
    }

    for order in args.orders:
        print(f"\n{'='*55}")
        print(f"{order}-gram  (context depth {order - 1})")
        print(f"{'='*55}")
        r = run_ngram(train_text, val_text, order, vocab)
        results["ngram_results"].append(r)
        print(f"  BPC = {r['bpc']:.4f}  |  PPL = {r['perplexity']:.2f}  "
              f"|  train {r['train_time_s']}s  eval {r['eval_time_s']}s")
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # Summary table
    print(f"\n{'='*55}")
    print("SUMMARY — N-gram KT baseline")
    print(f"{'='*55}")
    print(f"{'Order':>6}  {'Context':>9}  {'BPC':>8}  {'PPL':>8}")
    print("-" * 38)
    for r in results["ngram_results"]:
        print(f"{r['order']:>6}  "
              f"{str(r['context_depth'])+' chars':>9}  "
              f"{r['bpc']:>8.4f}  "
              f"{r['perplexity']:>8.2f}")
    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
