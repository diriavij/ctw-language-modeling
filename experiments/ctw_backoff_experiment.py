"""
Ablation: CTW mixing strategy vs. N-gram backoff.

Compares three prediction strategies at the same context depth:
  - CTW pred_alpha=0.0  — pure backoff (use deepest available node, no mixing)
  - CTW pred_alpha=0.1  — 10% current node + 90% child  (current default)
  - CTW pred_alpha=0.5  — equal mix (original CTW formula, optimal for coding)
  - N-gram KT order N   — same KT estimator, flat backoff (no tree weighting)

Theoretical prediction:
  pred_alpha=0.0 should ≈ N-gram (both do pure backoff with KT estimator).
  pred_alpha=0.5 should be worst for offline prediction (50% weight to unigram).

Uses a subset of WikiText-2 (default 500K train / 150K val) for speed.
Saves results under experiments/results/ and figures under experiments/figures/.

Usage:
    python experiments/ctw_backoff_experiment.py
    python experiments/ctw_backoff_experiment.py --full    # 2M train chars
    python experiments/ctw_backoff_experiment.py --depth 7
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import math
import time
from datetime import datetime

from ctw.text_ctw import TextCTW
from text_perplexity import normalize_text, load_wikitext2
from ngram_baseline import FastCharNgramKT
from _paths import figure_for_result, result_path


# -----------------------------------------------------------------------
# CTW evaluation at a fixed pred_alpha
# -----------------------------------------------------------------------

def run_ctw_alpha(
    train_text: str,
    val_text: str,
    depth: int,
    vocab: list[str],
    pred_alpha: float,
    min_child_count: int = 15,
) -> dict:
    ctw = TextCTW(
        depth=depth,
        vocab=vocab,
        pred_alpha=pred_alpha,
        min_child_count=min_child_count,
    )

    t0 = time.time()
    for t, ch in enumerate(train_text):
        ctw.update(ch, list(train_text[max(0, t - depth):t]))
    train_time = time.time() - t0

    t0 = time.time()
    total_bits = 0.0
    for t, ch in enumerate(val_text):
        total_bits += ctw.log_loss(ch, list(val_text[max(0, t - depth):t]))
    eval_time = time.time() - t0

    bpc = total_bits / len(val_text)
    return {
        "model":            f"CTW α={pred_alpha}",
        "pred_alpha":       pred_alpha,
        "depth":            depth,
        "bpc":              bpc,
        "perplexity":       2 ** bpc,
        "train_time_s":     round(train_time, 1),
        "eval_time_s":      round(eval_time, 1),
    }


# -----------------------------------------------------------------------
# N-gram evaluation
# -----------------------------------------------------------------------

def run_ngram(
    train_text: str,
    val_text: str,
    order: int,
    vocab: list[str],
) -> dict:
    model = FastCharNgramKT(order=order, vocab=vocab)
    t0    = time.time()
    model.fit(train_text)
    train_time = time.time() - t0
    t0    = time.time()
    bpc   = model.eval_bpc(val_text)
    eval_time = time.time() - t0
    return {
        "model":        f"N-gram order={order}",
        "order":        order,
        "context_depth": order - 1,
        "bpc":          bpc,
        "perplexity":   2 ** bpc,
        "train_time_s": round(train_time, 1),
        "eval_time_s":  round(eval_time, 1),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth",  type=int, default=5,
                        help="CTW context depth to ablate (default 5)")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.3, 0.5],
                        help="pred_alpha values to sweep")
    parser.add_argument("--full",   action="store_true",
                        help="Use full WikiText-2 (2M train). Default: 500K (faster).")
    parser.add_argument("--out",    default=None)
    args = parser.parse_args()

    exp_dir = os.path.dirname(__file__)
    if args.out is None:
        args.out = result_path("ctw_backoff_results.json")

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
    print(f"  Train: {len(train_text):,}  Val: {len(val_text):,}  Vocab: {len(vocab)}")

    results = {
        "timestamp":   datetime.now().isoformat(),
        "depth":       args.depth,
        "train_chars": len(train_text),
        "val_chars":   len(val_text),
        "entries":     [],
    }

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---- CTW sweep over pred_alpha ----
    print(f"\n{'='*55}")
    print(f"CTW D={args.depth} — sweeping pred_alpha")
    print(f"{'='*55}")
    for alpha in args.alphas:
        print(f"\n  pred_alpha = {alpha}")
        r = run_ctw_alpha(train_text, val_text, args.depth, vocab, alpha)
        results["entries"].append(r)
        print(f"  BPC = {r['bpc']:.4f}  PPL = {r['perplexity']:.2f}  "
              f"train {r['train_time_s']}s  eval {r['eval_time_s']}s")
        save()

    # ---- N-gram at matching order ----
    order = args.depth + 1   # same context depth
    print(f"\n{'='*55}")
    print(f"N-gram order={order} (context depth {args.depth}) — KT + backoff")
    print(f"{'='*55}")
    r = run_ngram(train_text, val_text, order, vocab)
    results["entries"].append(r)
    print(f"  BPC = {r['bpc']:.4f}  PPL = {r['perplexity']:.2f}  "
          f"train {r['train_time_s']}s  eval {r['eval_time_s']}s")
    save()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY  (D={args.depth}, train={len(train_text):,})")
    print(f"{'='*60}")
    print(f"{'Model':<22}  {'BPC':>8}  {'PPL':>8}")
    print("-" * 42)
    for r in results["entries"]:
        print(f"{r['model']:<22}  {r['bpc']:>8.4f}  {r['perplexity']:>8.2f}")

    _plot(results, figure_for_result(args.out))
    print(f"\nResults saved → {args.out}")


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def _plot(results: dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    ctw_entries = [e for e in results["entries"] if "pred_alpha" in e]
    ng_entries  = [e for e in results["entries"] if "order" in e]

    fig, ax = plt.subplots(figsize=(7, 4))

    alphas = [e["pred_alpha"] for e in ctw_entries]
    bpcs   = [e["bpc"]        for e in ctw_entries]
    ax.plot(alphas, bpcs, "o-", color="#1565C0", linewidth=2,
            markersize=7, label=f"CTW D={results['depth']}")
    for a, b in zip(alphas, bpcs):
        ax.annotate(f"{b:.4f}", (a, b),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#1565C0")

    # N-gram horizontal line
    for ng in ng_entries:
        ax.axhline(ng["bpc"], color="#E65100", linestyle="--", linewidth=1.8,
                   label=f"N-gram order={ng['order']}  ({ng['bpc']:.4f})")
        ax.text(max(alphas), ng["bpc"] + 0.003,
                f"{ng['bpc']:.4f}", ha="right", fontsize=8, color="#E65100")

    ax.set_xlabel("pred_alpha  (0 = pure backoff, 0.5 = original CTW)", fontsize=10)
    ax.set_ylabel("BPC ↓", fontsize=10)
    ax.set_title(f"CTW mixing strategy vs. N-gram  (D={results['depth']})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_xlim(-0.03, max(alphas) + 0.05)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
