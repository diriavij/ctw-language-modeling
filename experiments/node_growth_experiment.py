"""
Node growth experiment: how the CTW context tree fills with training data.

The key question: what fraction of the theoretical A^D contexts are actually
seen at each depth D? This shows WHY D_max is the practical limit.

Theoretical nodes at depth D:  A^D
Empirical nodes at depth D:    unique D-grams seen in training text

For D << D_max: empirical ≈ theoretical (tree is well-filled)
For D >>  D_max: empirical << theoretical (tree is sparse; most leaves unseen)

The D_max formula:  D_max = log_A(N / n_min)
marks where expected count per context drops below n_min.

Two panels:
  1. Char-level (A=28): uses bpc_min results (precomputed)
  2. Word-level (A=vocab_size): computed here for multiple vocab sizes

Usage:
    python experiments/node_growth_experiment.py
    python experiments/node_growth_experiment.py --vocab_sizes 1000 5000 10000 20000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import math
import time
import glob
from collections import Counter
from datetime import datetime

from text_perplexity import normalize_text, load_wikitext2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dmax(A, N, n_min=15):
    """Theoretical maximum useful depth: D_max = log_A(N / n_min)."""
    return math.log(N / n_min) / math.log(A)


def theoretical_nodes(A, D):
    """Number of possible contexts of length exactly D: A^D."""
    if D == 0:
        return 1
    return A ** D


def load_char_node_counts(exp_dir):
    """
    Load empirical char-level context counts from existing bpc_min results.
    Returns {depth: n_unique_contexts} or None if no results found.
    """
    files = sorted(glob.glob(os.path.join(exp_dir, "bpc_min_results_2*.json")))
    # Skip val results
    files = [f for f in files if "val" not in os.path.basename(f)]
    if not files:
        return None
    with open(files[-1]) as f:
        data = json.load(f)
    counts = {}
    for r in data.get("bpc_min_results", []):
        counts[r["depth"]] = r["n_unique_ctx"]
    n_train = None
    # Parse train size from source label e.g. "train (10,023,770 chars)"
    source = data.get("source", "")
    import re
    m = re.search(r"([\d,]+)\s*chars", source)
    if m:
        n_train = int(m.group(1).replace(",", ""))
    return counts, n_train


def compute_word_node_counts(word_tokens, max_depth):
    """
    Count unique context tuples at each depth 0..max_depth.
    Returns {depth: n_unique_contexts}.
    """
    N = len(word_tokens)
    results = {0: 1}
    for D in range(1, max_depth + 1):
        t0 = time.time()
        ctx_set = set()
        for i in range(D, N):
            ctx = tuple(word_tokens[i - D:i])
            ctx_set.add(ctx)
        results[D] = len(ctx_set)
        print(f"    D={D}: {len(ctx_set):,} unique contexts  ({time.time()-t0:.1f}s)")
    return results


def encode_words(text, vocab_size):
    """Top-vocab_size words; rest → '<UNK>'. Returns (tokens, actual_vocab_size)."""
    words = text.split()
    counts = Counter(words)
    top = {w for w, _ in counts.most_common(vocab_size - 1)}
    tokens = [w if w in top else "<UNK>" for w in words]
    return tokens, len(top) + 1   # +1 for <UNK>


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(char_counts, n_train_chars, word_results, n_train_words, out_path):
    """
    char_counts:   {depth: n_actual_contexts}
    word_results:  {vocab_size: {depth: n_actual_contexts}}
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    A_char = 28
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # -----------------------------------------------------------------------
    # Panel 1: Char-level (A=28)
    # -----------------------------------------------------------------------
    ax = axes[0]

    depths = sorted(char_counts.keys())
    empirical    = [char_counts[d] for d in depths]
    theor        = [theoretical_nodes(A_char, d) for d in depths]
    fill_pct     = [e / t * 100 if t > 0 else 0 for e, t in zip(empirical, theor)]

    ax.semilogy(depths, theor, "o--", color="#B71C1C", lw=2, markersize=6,
                label=f"Theoretical: $A^D$  (A={A_char})", alpha=0.8, zorder=3)
    ax.semilogy(depths, empirical, "s-", color="#1565C0", lw=2.5, markersize=7,
                label="Empirical: unique contexts\n(10M training chars)", zorder=4)

    # Annotate fill %
    for d, e, t, pct in zip(depths, empirical, theor, fill_pct):
        va = "bottom" if e < t * 0.7 else "top"
        offset = 8 if va == "bottom" else -8
        ax.annotate(f"{pct:.0f}%",
                    xy=(d, e), xytext=(0, offset),
                    textcoords="offset points",
                    ha="center", fontsize=7.5, color="#1565C0")

    # D_max line
    dm = dmax(A_char, n_train_chars, n_min=15)
    ax.axvline(dm, color="#E65100", linestyle=":", lw=2.2, zorder=5,
               label=f"$D_{{max}}$ = {dm:.1f}")
    ax.text(dm + 0.08, 1.5, f"$D_{{max}}$={dm:.1f}",
            color="#E65100", fontsize=9.5, va="bottom")

    # n_min reference: horizontal line at N/n_min
    n_min = 15
    avg_count_line = n_train_chars / n_min
    ax.axhline(avg_count_line, color="#555", linestyle="--", lw=1.2, alpha=0.5,
               label=f"$N/n_{{min}}$ = {int(avg_count_line):,}\n"
                     f"(avg. count = $n_{{min}}$ at this many contexts)")

    ax.set_xlabel("Context depth D", fontsize=11)
    ax.set_ylabel("Number of unique contexts (log scale)", fontsize=10)
    ax.set_title(f"Char-level CTW tree  (A={A_char})\n"
                 f"% = empirical / theoretical nodes", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_xticks(depths)

    # -----------------------------------------------------------------------
    # Panel 2: Word-level (multiple vocab sizes)
    # -----------------------------------------------------------------------
    ax2 = axes[1]

    palette = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00838F"]

    for i, (vocab_size, depth_counts) in enumerate(
            sorted(word_results.items(), key=lambda x: x[0])):
        color = palette[i % len(palette)]
        d_arr  = sorted(depth_counts.keys())
        a_arr  = [depth_counts[d] for d in d_arr]
        t_arr  = [theoretical_nodes(vocab_size, d) for d in d_arr]
        dm_w   = dmax(vocab_size, n_train_words, n_min=15)

        # Theoretical (dashed, lighter)
        ax2.semilogy(d_arr, t_arr, "--", color=color, lw=1.2, alpha=0.45)
        # Empirical (solid)
        ax2.semilogy(d_arr, a_arr, "s-", color=color, lw=2, markersize=6,
                     label=f"vocab={vocab_size:,}  ($D_{{max}}$={dm_w:.2f})", zorder=4)
        # D_max vertical
        ax2.axvline(dm_w, color=color, lw=1, alpha=0.35, linestyle=":")

        # Annotate fill % at each depth
        for d, a, t in zip(d_arr, a_arr, t_arr):
            pct = a / t * 100 if t > 0 else 0
            ax2.annotate(f"{pct:.0f}%",
                         xy=(d, a), xytext=(0, 6),
                         textcoords="offset points",
                         ha="center", fontsize=6.5, color=color)

    ax2.set_xlabel("Context depth D (words)", fontsize=11)
    ax2.set_ylabel("Number of unique contexts (log scale)", fontsize=10)
    ax2.set_title(f"Word-level CTW tree  (multiple vocab sizes)\n"
                  f"N ≈ {n_train_words/1e6:.2f}M training words  |  dashed = theoretical $A^D$",
                  fontsize=11)
    ax2.legend(fontsize=8.5, loc="upper left")
    ax2.grid(True, which="both", alpha=0.2, linestyle="--")
    ax2.set_xticks(sorted({d for dc in word_results.values() for d in dc}))

    plt.suptitle("CTW Context Tree: Theoretical vs Empirical Node Count\n"
                 "Numbers show % of theoretical nodes that are actually populated",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    stable = os.path.join(os.path.dirname(out_path), "node_growth_plot.png")
    fig.savefig(stable, dpi=150, bbox_inches="tight")
    print(f"Saved → {stable}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_sizes", type=int, nargs="+",
                        default=[1000, 5000, 10000, 20000])
    parser.add_argument("--word_max_depth", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    exp_dir = os.path.dirname(__file__)
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(exp_dir, f"node_growth_results_{ts}.json")

    # ---- Load char-level counts from bpc_min results ----
    print("Loading char-level node counts from bpc_min results...")
    result = load_char_node_counts(exp_dir)
    if result is None:
        print("ERROR: bpc_min results not found. Run bpc_min_experiment.py first.")
        return
    char_counts, n_train_chars = result
    n_train_chars = n_train_chars or 10_023_770
    print(f"  Loaded D={sorted(char_counts.keys())[0]}..{sorted(char_counts.keys())[-1]}, "
          f"N={n_train_chars:,} chars")
    for d in sorted(char_counts.keys()):
        t = theoretical_nodes(28, d)
        e = char_counts[d]
        pct = e / t * 100 if t > 0 else 0
        print(f"  D={d}: empirical={e:>12,}  theoretical={t:>12,}  fill={pct:5.1f}%")

    # ---- Load text for word-level computation ----
    print("\nLoading WikiText-2...")
    train_raw, _ = load_wikitext2()
    train_text = normalize_text(train_raw)
    print(f"  Train: {len(train_text):,} chars")
    n_train_words = len(train_text.split())
    print(f"  Train: {n_train_words:,} words")

    # ---- Word-level node counts ----
    word_results = {}
    for vocab_size in args.vocab_sizes:
        print(f"\nVocab size = {vocab_size:,}  (D_max = {dmax(vocab_size, n_train_words):.2f})")
        tokens, actual_vocab = encode_words(train_text, vocab_size)
        print(f"  Actual vocab used: {actual_vocab:,}  |  tokens: {len(tokens):,}")
        depth_counts = compute_word_node_counts(tokens, args.word_max_depth)
        word_results[actual_vocab] = depth_counts

        print(f"  Depth breakdown:")
        for d in sorted(depth_counts.keys()):
            t = theoretical_nodes(actual_vocab, d)
            e = depth_counts[d]
            pct = e / t * 100 if t > 0 else 0
            print(f"    D={d}: empirical={e:>10,}  theoretical={t:>12,}  fill={pct:6.2f}%")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY: D_max theory vs actual sparsity")
    print(f"{'='*60}")
    print(f"\nChar-level (A=28, N={n_train_chars:,}):")
    print(f"  D_max = {dmax(28, n_train_chars):.2f}")
    for d in sorted(char_counts.keys()):
        t = theoretical_nodes(28, d)
        e = char_counts[d]
        avg_cnt = n_train_chars / e if e > 0 else 0
        print(f"  D={d}: fill={e/t*100:.1f}%  avg_count={avg_cnt:.1f}")
    print(f"\nWord-level:")
    for vs, depth_counts in sorted(word_results.items()):
        print(f"  vocab={vs:,}  D_max={dmax(vs, n_train_words):.2f}")
        for d in sorted(depth_counts.keys()):
            if d == 0:
                continue
            t = theoretical_nodes(vs, d)
            e = depth_counts[d]
            avg_cnt = n_train_words / e if e > 0 else 0
            print(f"    D={d}: fill={e/t*100:.2f}%  avg_count={avg_cnt:.1f}")

    # ---- Save results ----
    output = {
        "timestamp":      datetime.now().isoformat(),
        "n_train_chars":  n_train_chars,
        "n_train_words":  n_train_words,
        "char_counts":    {str(k): v for k, v in char_counts.items()},
        "word_results":   {str(k): {str(d): v for d, v in dc.items()}
                           for k, dc in word_results.items()},
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults → {args.out}")

    # ---- Plot ----
    out_png = args.out.replace(".json", ".png")
    _plot(char_counts, n_train_chars, word_results, n_train_words, out_png)


if __name__ == "__main__":
    main()
