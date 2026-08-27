"""
Theoretical D_max analysis: at what context depth does a CTW tree saturate?

D_max = log_A(N / n_min)

Where:
  N     = training corpus size (chars)
  A     = alphabet size
  n_min = minimum child count threshold (15 in our setup)

The formula says: beyond depth D_max, the average leaf has been seen fewer than
n_min times, so backoff fires on most paths and additional depth buys nothing.

This script:
  1. Plots D_max vs. corpus size N for several alphabet sizes (A=2,28,59,95,256)
  2. Marks our WikiText-2 and text8 operating points
  3. Overlays empirical CTW saturation depths from results/full_results_*.json
  4. Shows a secondary axis: "% of leaves seen ≥ n_min" vs. depth

Saves: experiments/figures/dmax_theory.pdf / .png

Usage:
    python experiments/dmax_theory.py
    python experiments/dmax_theory.py --nmin 5 --nmin 15 --nmin 50   # compare thresholds
    python experiments/dmax_theory.py --ctw_json experiments/results/full_results_*.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import math
import glob

from _paths import FIGURES_DIR, RESULTS_DIR, ensure_artifact_dirs


# -----------------------------------------------------------------------
# Core formula
# -----------------------------------------------------------------------

def dmax(N: float, A: int, n_min: int = 15) -> float:
    """Theoretical maximum useful CTW depth."""
    if N <= 0 or A <= 1 or n_min <= 0:
        return 0.0
    return math.log(N / n_min) / math.log(A)


def expected_leaf_count(N: float, A: int, depth: int) -> float:
    """Expected number of training chars at a depth-D leaf = N / A^D."""
    return N / (A ** depth)


def fraction_seen(N: float, A: int, depth: int, n_min: int = 15) -> float:
    """
    Fraction of depth-D leaves that have been seen ≥ n_min times.
    Approximates each leaf as Poisson(lambda = N / A^D).
    """
    import math
    lam = N / (A ** depth)
    if lam == 0:
        return 0.0
    # P(Poisson(lam) >= n_min)  = 1 - CDF(n_min - 1)
    # Use incomplete gamma: P(X >= k) = Gamma(k, lam) / (k-1)!
    # Simple direct sum for small n_min:
    prob_lt = 0.0
    for k in range(n_min):
        prob_lt += math.exp(-lam) * (lam ** k) / math.factorial(k)
        if prob_lt >= 1.0:
            return 0.0
    return max(0.0, 1.0 - prob_lt)


# -----------------------------------------------------------------------
# Load empirical saturation depth from results JSON
# -----------------------------------------------------------------------

def empirical_saturation(json_path: str) -> tuple[int | None, float | None]:
    """
    Find the depth where CTW BPC stops improving (saturation).
    Returns (saturation_depth, train_chars).
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
        results = data.get("ctw_results", [])
        if len(results) < 2:
            return None, data.get("train_chars")
        # Find first depth where improvement < 0.001 bpc
        best_bpc  = results[0]["bpc"]
        best_d    = results[0]["depth"]
        sat_d     = None
        for r in results[1:]:
            improvement = best_bpc - r["bpc"]
            if improvement < 0.001:
                sat_d = r["depth"]
                break
            best_bpc = r["bpc"]
            best_d   = r["depth"]
        if sat_d is None:
            sat_d = results[-1]["depth"]
        return sat_d, data.get("train_chars")
    except Exception:
        return None, None


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmin",    type=int, nargs="+", default=[15],
                        help="min_child_count threshold(s) to plot (default: 15)")
    parser.add_argument("--ctw_json", default=None,
                        help="Path to full_results_*.json (auto-discovered if omitted)")
    parser.add_argument("--ptb_json", default=None,
                        help="Path to ptb_results_*.json (text8, auto-discovered if omitted)")
    parser.add_argument("--out",     default=None)
    args = parser.parse_args()

    ensure_artifact_dirs()
    out_prefix = args.out or os.path.join(str(FIGURES_DIR), "dmax_theory")

    # Auto-discover result files
    def find_latest(pattern):
        files = sorted(glob.glob(pattern))
        return files[-1] if files else None

    ctw_json = args.ctw_json or find_latest(os.path.join(str(RESULTS_DIR), "full_results_*.json"))
    ptb_json = args.ptb_json or find_latest(os.path.join(str(RESULTS_DIR), "ptb_results_*.json"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib")
        _print_table(args.nmin)
        return

    # -----------------------------------------------------------------------
    # Figure layout: 2 panels side by side
    # Panel 1: D_max vs. corpus size N  (several alphabets + our points)
    # Panel 2: Fraction of leaves seen  vs. depth  (fixed N=2M, A=28)
    # -----------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Corpus sizes: 10K to 100B chars ----
    N_vals = np.logspace(4, 11, 500)

    # ---- Panel 1: D_max vs. N ----
    alphabet_specs = [
        (2,   "A=2 (binary)",       "#B71C1C"),
        (28,  "A=28 (our setup)",   "#1565C0"),
        (59,  "A=59 (printable)",   "#2E7D32"),
        (95,  "A=95 (ASCII print)", "#6A1B9A"),
        (256, "A=256 (bytes)",      "#E65100"),
    ]

    n_min_primary = args.nmin[0]
    for A, label, color in alphabet_specs:
        dmaxs = [dmax(N, A, n_min_primary) for N in N_vals]
        lw = 2.5 if A == 28 else 1.5
        ax1.plot(N_vals, dmaxs, color=color, linewidth=lw, label=label)

    # Additional n_min thresholds for A=28
    colors_nmin = ["#1565C0", "#0288D1", "#26C6DA"]
    for idx, nm in enumerate(args.nmin[1:]):
        dmaxs = [dmax(N, 28, nm) for N in N_vals]
        ax1.plot(N_vals, dmaxs, color=colors_nmin[idx % 3], linewidth=1.5,
                 linestyle=":", label=f"A=28, n_min={nm}")

    # Our operating points
    points = []
    if ctw_json and os.path.exists(ctw_json):
        sat_d, train_chars = empirical_saturation(ctw_json)
        if train_chars:
            d_theory = dmax(train_chars, 28, n_min_primary)
            points.append(("WikiText-2\n(our exp.)", train_chars, d_theory,
                           sat_d, "#1565C0"))

    if ptb_json and os.path.exists(ptb_json):
        sat_d2, train_chars2 = empirical_saturation(ptb_json)
        if train_chars2:
            d_theory2 = dmax(train_chars2, 28, n_min_primary)
            points.append(("text8\n(our exp.)", train_chars2, d_theory2,
                           sat_d2, "#E65100"))

    for label, N_pt, d_theory, sat_d, color in points:
        ax1.scatter([N_pt], [d_theory], s=100, zorder=6, color=color,
                    edgecolors="white", linewidths=1.5)
        ax1.annotate(
            f"{label}\nN={N_pt/1e6:.1f}M\nD_max={d_theory:.1f}" +
            (f"\nsat≈{sat_d}" if sat_d else ""),
            xy=(N_pt, d_theory),
            xytext=(N_pt * 3.5, d_theory + 2.5),
            fontsize=7.5, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
        )
        # Vertical guide line
        ax1.axvline(N_pt, color=color, linestyle="--", linewidth=0.8, alpha=0.4)

    # Our actual depth range (D=3..10)
    ax1.axhspan(3, 10, alpha=0.06, color="#1565C0", zorder=0)
    ax1.text(1e4 * 1.3, 6.3, "Our D sweep\n(3–10)", fontsize=7.5,
             color="#1565C0", alpha=0.7)

    ax1.set_xscale("log")
    ax1.set_xlabel("Training corpus size N (chars)", fontsize=10)
    ax1.set_ylabel(f"$D_{{\\max}}$ = $\\log_A(N / n_{{\\min}})$, $n_{{\\min}}$={n_min_primary}",
                   fontsize=10)
    ax1.set_title("Theoretical Saturation Depth vs. Corpus Size", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.25, linestyle="--")
    ax1.set_xlim(N_vals[0], N_vals[-1])
    ax1.set_ylim(0, max(20, dmax(N_vals[-1], 2, n_min_primary) + 2))

    # X-axis tick labels in readable form
    ax1.set_xticks([1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11])
    ax1.set_xticklabels(["10K", "100K", "1M", "10M", "100M", "1B", "10B", "100B"],
                        fontsize=8)

    # ---- Panel 2: Fraction of leaves seen >= n_min vs. depth ----
    depths = list(range(1, 16))
    N_scenarios = [
        (500_000,   "N=500K (fast subset)", "#B0BEC5", "--"),
        (2_000_000, "N=2M (WikiText-2)",    "#1565C0", "-"),
        (90_000_000,"N=90M (text8)",        "#E65100", "-"),
    ]

    for N_scen, label, color, ls in N_scenarios:
        fracs = [fraction_seen(N_scen, 28, d, n_min_primary) * 100 for d in depths]
        ax2.plot(depths, fracs, color=color, linestyle=ls, linewidth=2, label=label)

    # Mark D_max for each scenario (where fraction drops below 50%)
    for N_scen, label, color, ls in N_scenarios:
        d_th = dmax(N_scen, 28, n_min_primary)
        ax2.axvline(d_th, color=color, linewidth=1, linestyle=":", alpha=0.7)
        ax2.text(d_th + 0.15, 55,
                 f"D_max={d_th:.1f}",
                 fontsize=7, color=color, rotation=90, va="bottom")

    ax2.axhline(50, color="#78909C", linestyle="--", linewidth=1, alpha=0.7,
                label="50% threshold")
    ax2.set_xlabel("Context depth D", fontsize=10)
    ax2.set_ylabel(f"% of leaves seen ≥ $n_{{\\min}}$={n_min_primary} times", fontsize=10)
    ax2.set_title("Coverage: Fraction of Useful Leaves vs. Depth", fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.set_xlim(1, 15)
    ax2.set_ylim(-2, 105)
    ax2.set_xticks(depths)

    plt.suptitle(
        "CTW Saturation Depth: $D_{\\max} = \\log_A(N / n_{\\min})$",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)

    # ---- Print table ----
    _print_table(args.nmin, ctw_json, ptb_json)


def _print_table(nmin_list, ctw_json=None, ptb_json=None):
    def find_latest(pattern):
        files = sorted(glob.glob(pattern))
        return files[-1] if files else None

    ctw_json = ctw_json or find_latest(os.path.join(str(RESULTS_DIR), "full_results_*.json"))
    ptb_json = ptb_json or find_latest(os.path.join(str(RESULTS_DIR), "ptb_results_*.json"))

    print(f"\n{'='*65}")
    print(f"D_max = log_A(N / n_min)  for A=28 (our 28-char alphabet)")
    print(f"{'='*65}")
    print(f"{'N':>12}  {'n_min':>6}  {'D_max':>8}  Dataset")
    print("-" * 48)

    scenarios = [
        (500_000,    "fast subset (500K)"),
        (2_000_000,  "WikiText-2 (2M)"),
        (90_000_000, "text8 (90M)"),
    ]
    for nm in nmin_list:
        for N, label in scenarios:
            d = dmax(N, 28, nm)
            print(f"{N:>12,}  {nm:>6}  {d:>8.2f}  {label}")
        if nm != nmin_list[-1]:
            print()

    if ctw_json and os.path.exists(ctw_json):
        sat_d, train_chars = empirical_saturation(ctw_json)
        print(f"\n  Empirical saturation (WikiText-2): "
              f"depth ≈ {sat_d}  (theory: {dmax(train_chars, 28, nmin_list[0]):.1f})")
    if ptb_json and os.path.exists(ptb_json):
        sat_d2, train_chars2 = empirical_saturation(ptb_json)
        print(f"  Empirical saturation (text8):       "
              f"depth ≈ {sat_d2}  (theory: {dmax(train_chars2, 28, nmin_list[0]):.1f})")


if __name__ == "__main__":
    main()
