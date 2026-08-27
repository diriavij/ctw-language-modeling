"""
Word-level CTW scaling experiment.

Sweeps over vocabulary sizes and context depths to show:
  1. How BPC depends on vocab size at fixed depth
  2. How BPC depends on depth at fixed vocab size
  3. That D=2 is always worse than D=1 (confirmed by D_max theory)
  4. The optimal vocab size for word-level prediction

For each (vocab_size, depth) pair:
  - Trains KT n-gram on 10M chars (≈ 1.73M words)
  - Evaluates on 1M val chars
  - Computes BPC = word_bits / total_chars  (spaces not predicted — see note)

Note: BPC here excludes space prediction cost (same convention as Word-CTW in
token_ctw_experiment.py). To get a fully comparable BPC, add ~0.169 for spaces.
GPT-2 reference and char-CTW baseline are added from existing results for context.

D_max theory:  D_max = log_A(N / n_min)
    With N ≈ 1.73M words and n_min=15:
    vocab=500:  D_max ≈ 2.11
    vocab=1000: D_max ≈ 1.84
    vocab=2000: D_max ≈ 1.62
    vocab=5000: D_max ≈ 1.40
    vocab=10000: D_max ≈ 1.27
    vocab=20000: D_max ≈ 1.17

Usage:
    python experiments/word_ctw_scaling.py
    python experiments/word_ctw_scaling.py --vocab_sizes 500 1000 2000 5000 10000 20000
    python experiments/word_ctw_scaling.py --plot_only experiments/word_ctw_scaling_*.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import math
import time
import glob
from collections import Counter
from datetime import datetime

from text_perplexity import normalize_text, load_wikitext2
from token_ctw_experiment import TokenNgramKT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dmax(A, N, n_min=15):
    return math.log(N / n_min) / math.log(A)


def build_vocab(train_words, vocab_size):
    """Top-(vocab_size-1) words + <UNK>. Returns (word2idx, vocab_list)."""
    counts = Counter(train_words)
    top = [w for w, _ in counts.most_common(vocab_size - 1)]
    vocab = ["<UNK>"] + top
    word2idx = {w: i for i, w in enumerate(vocab)}
    return word2idx, vocab


def encode(words, word2idx):
    return [w if w in word2idx else "<UNK>" for w in words]


def load_reference_bpc(exp_dir):
    """Load char-CTW D=5 and GPT-2 BPC from full_results."""
    char_ctw_bpc, gpt2_bpc = None, None
    for f in sorted(glob.glob(os.path.join(exp_dir, "full_results_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        for r in d.get("ctw_results", []):
            if r["depth"] == 5:
                char_ctw_bpc = r["bpc"]
        if d.get("gpt2_result"):
            gpt2_bpc = d["gpt2_result"]["bpc_per_char"]
    return char_ctw_bpc, gpt2_bpc


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(train_text, val_text, vocab_sizes, depths, n_min=15):
    """
    Returns nested dict: results[vocab_size][depth] = {bpc, bpt, coverage, ...}
    """
    train_words = train_text.split()
    val_words   = val_text.split()
    N_train     = len(train_words)
    N_val_chars = len(val_text)

    results = {}

    for vocab_size in vocab_sizes:
        print(f"\n{'─'*55}")
        print(f"Vocab size = {vocab_size:,}  "
              f"(D_max = {dmax(vocab_size, N_train):.2f})")
        print(f"{'─'*55}")

        word2idx, vocab = build_vocab(train_words, vocab_size)
        train_tokens = encode(train_words, word2idx)
        val_tokens   = encode(val_words,   word2idx)

        actual_vocab = len(vocab)
        cov_train = sum(1 for w in train_tokens if w != "<UNK>") / len(train_tokens)
        cov_val   = sum(1 for w in val_tokens   if w != "<UNK>") / len(val_tokens)
        print(f"  Actual vocab: {actual_vocab:,}  |  "
              f"train coverage: {cov_train*100:.1f}%  |  val coverage: {cov_val*100:.1f}%")

        results[vocab_size] = {}

        for depth in depths:
            t0 = time.time()
            model = TokenNgramKT(order=depth, vocab=vocab, alpha=0.5)
            model.fit(train_tokens)
            fit_time = time.time() - t0

            t0 = time.time()
            bpt = model.eval_bpt(val_tokens)
            eval_time = time.time() - t0

            # BPC = word_bits / total_val_chars  (no space prediction)
            n_val_words = len(val_tokens)
            bpc = (bpt * n_val_words) / N_val_chars

            print(f"  D={depth}:  BPT={bpt:.4f}  BPC≈{bpc:.4f}  "
                  f"(fit {fit_time:.1f}s  eval {eval_time:.1f}s)")

            results[vocab_size][depth] = {
                "bpt":          round(bpt, 6),
                "bpc":          round(bpc, 6),
                "bpc_with_spaces": round(bpc + 0.169, 6),  # approximate
                "vocab_actual": actual_vocab,
                "coverage_train": round(cov_train, 4),
                "coverage_val":   round(cov_val, 4),
                "ppl_word":     round(2 ** bpt, 3),
                "dmax_theory":  round(dmax(vocab_size, N_train), 4),
                "fit_time_s":   round(fit_time, 2),
                "eval_time_s":  round(eval_time, 2),
            }

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(results, char_ctw_bpc, gpt2_bpc, n_train_words, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    palette = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00838F", "#37474F"]
    vocab_sizes = sorted(results.keys())
    depths_all  = sorted({d for vr in results.values() for d in vr})

    # -----------------------------------------------------------------------
    # Panel 1: BPC vs vocab size, one line per depth
    # -----------------------------------------------------------------------
    ax = axes[0]

    depth_colors = {d: palette[i % len(palette)] for i, d in enumerate(depths_all)}

    for depth in depths_all:
        xs = []
        ys = []
        for vs in vocab_sizes:
            if depth in results[vs]:
                xs.append(vs)
                ys.append(results[vs][depth]["bpc"])
        if xs:
            ax.plot(xs, ys, "o-", color=depth_colors[depth], lw=2, markersize=7,
                    label=f"D = {depth}", zorder=4)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3f}", (x, y),
                            textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=7.5, color=depth_colors[depth])

    # D_max = 2 crossing: mark vocab sizes where D_max passes 2
    vs_arr = np.linspace(min(vocab_sizes), max(vocab_sizes), 500)
    dm_arr = [dmax(v, n_train_words) for v in vs_arr]
    # find where D_max = 2
    dm2_vs = None
    for v, dm_v in zip(vs_arr, dm_arr):
        if dm_v <= 2.0:
            dm2_vs = v
            break
    if dm2_vs:
        ax.axvline(dm2_vs, color="#B71C1C", linestyle=":", lw=1.8, alpha=0.7,
                   label=f"$D_{{max}}$=2 at vocab≈{int(dm2_vs):,}")
        ax.text(dm2_vs + max(vocab_sizes)*0.02, ax.get_ylim()[0] + 0.05,
                f"$D_{{max}}$=2", color="#B71C1C", fontsize=8.5)

    if char_ctw_bpc:
        ax.axhline(char_ctw_bpc, color="#795548", linestyle="--", lw=1.8,
                   label=f"Char-CTW D=5  {char_ctw_bpc:.3f}")
    if gpt2_bpc:
        ax.axhline(gpt2_bpc, color="#B71C1C", linestyle="--", lw=2,
                   label=f"GPT-2 small  {gpt2_bpc:.3f}")

    ax.set_xscale("log")
    ax.set_xlabel("Vocabulary size (log scale)", fontsize=11)
    ax.set_ylabel("BPC* (spaces not predicted)", fontsize=10)
    ax.set_title("Word-CTW BPC vs Vocabulary Size\n"
                 "by context depth D", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, which="both", alpha=0.2, linestyle="--")
    ax.set_xticks(vocab_sizes)
    ax.set_xticklabels([str(v) for v in vocab_sizes], fontsize=8.5)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

    # -----------------------------------------------------------------------
    # Panel 2: BPC vs depth, one line per vocab size
    # -----------------------------------------------------------------------
    ax2 = axes[1]

    for i, vs in enumerate(vocab_sizes):
        color = palette[i % len(palette)]
        dm_v  = dmax(vs, n_train_words)
        xs = sorted(results[vs].keys())
        ys = [results[vs][d]["bpc"] for d in xs]

        ax2.plot(xs, ys, "o-", color=color, lw=2, markersize=7,
                 label=f"vocab={vs:,}  ($D_{{max}}$={dm_v:.2f})", zorder=4)
        for x, y in zip(xs, ys):
            ax2.annotate(f"{y:.3f}", (x, y),
                         textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=7.5, color=color)

        # D_max vertical (per vocab size)
        ax2.axvline(dm_v, color=color, lw=1, alpha=0.3, linestyle=":")

    if char_ctw_bpc:
        ax2.axhline(char_ctw_bpc, color="#795548", linestyle="--", lw=1.8,
                    label=f"Char-CTW D=5  {char_ctw_bpc:.3f}")
    if gpt2_bpc:
        ax2.axhline(gpt2_bpc, color="#B71C1C", linestyle="--", lw=2,
                    label=f"GPT-2 small  {gpt2_bpc:.3f}")

    ax2.set_xlabel("Context depth D (words)", fontsize=11)
    ax2.set_ylabel("BPC* (spaces not predicted)", fontsize=10)
    ax2.set_title("Word-CTW BPC vs Context Depth\n"
                  "by vocabulary size  |  dashed verticals = $D_{max}$", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.set_xticks(depths_all)

    plt.suptitle("Word-level CTW Scaling: Vocabulary Size × Context Depth\n"
                 "(WikiText-2, 10M train chars ≈ 1.73M words  |  * add ≈0.169 for spaces)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    stable = os.path.join(os.path.dirname(out_path), "word_ctw_scaling_plot.png")
    fig.savefig(stable, dpi=150, bbox_inches="tight")
    print(f"Saved → {stable}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_sizes", type=int, nargs="+",
                        default=[500, 1000, 2000, 5000, 10000, 20000])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--plot_only", default=None, metavar="JSON")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    exp_dir = os.path.dirname(__file__)
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = os.path.join(exp_dir, f"word_ctw_scaling_{ts}.json")

    char_ctw_bpc, gpt2_bpc = load_reference_bpc(exp_dir)
    print(f"Reference: Char-CTW D=5 = {char_ctw_bpc}  |  GPT-2 = {gpt2_bpc}")

    if args.plot_only:
        with open(args.plot_only) as f:
            saved = json.load(f)
        # Convert str keys back to int
        results = {int(vs): {int(d): v for d, v in dv.items()}
                   for vs, dv in saved["results"].items()}
        n_train_words = saved.get("n_train_words", 1_730_000)
        out_png = args.plot_only.replace(".json", ".png")
        _plot(results, char_ctw_bpc, gpt2_bpc, n_train_words, out_png)
        return

    # ---- Load data ----
    print("\nLoading WikiText-2...")
    train_raw, val_raw = load_wikitext2()
    train_text = normalize_text(train_raw)
    val_text   = normalize_text(val_raw)
    n_train_words = len(train_text.split())
    n_val_words   = len(val_text.split())
    print(f"  Train: {len(train_text):,} chars / {n_train_words:,} words")
    print(f"  Val:   {len(val_text):,} chars / {n_val_words:,} words")

    # ---- Print D_max predictions ----
    print(f"\n{'='*55}")
    print("D_max predictions (theory):")
    print(f"{'='*55}")
    for vs in args.vocab_sizes:
        print(f"  vocab={vs:>6,}:  D_max = {dmax(vs, n_train_words):.2f}")

    # ---- Run sweep ----
    print(f"\n{'='*55}")
    print("Running sweep...")
    print(f"{'='*55}")
    results = run_sweep(train_text, val_text, args.vocab_sizes, args.depths)

    # ---- Print summary table ----
    print(f"\n{'='*70}")
    print("SUMMARY  (* BPC does not include space prediction cost ~0.169)")
    print(f"{'='*70}")
    header = f"{'vocab':>8}  {'D_max':>6}  " + "  ".join(f"D={d} BPC" for d in args.depths)
    print(header)
    print("─" * len(header))
    for vs in args.vocab_sizes:
        dm = dmax(vs, n_train_words)
        bpcs = "  ".join(
            f"{results[vs][d]['bpc']:8.4f}" if d in results[vs] else "      —"
            for d in args.depths
        )
        best_d = min(results[vs], key=lambda d: results[vs][d]["bpc"])
        print(f"  {vs:>6,}  {dm:>6.2f}  {bpcs}  ← best: D={best_d}")

    if char_ctw_bpc:
        print(f"\n  {'Char-CTW D=5':>20}: {char_ctw_bpc:.4f} BPC  (includes spaces)")
    if gpt2_bpc:
        print(f"  {'GPT-2 small':>20}: {gpt2_bpc:.4f} BPC")

    # ---- Save ----
    output = {
        "timestamp":     datetime.now().isoformat(),
        "n_train_words": n_train_words,
        "n_val_words":   n_val_words,
        "vocab_sizes":   args.vocab_sizes,
        "depths":        args.depths,
        "results":       {str(vs): {str(d): v for d, v in dv.items()}
                          for vs, dv in results.items()},
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults → {args.out}")

    # ---- Plot ----
    out_png = args.out.replace(".json", ".png")
    _plot(results, char_ctw_bpc, gpt2_bpc, n_train_words, out_png)


if __name__ == "__main__":
    main()
