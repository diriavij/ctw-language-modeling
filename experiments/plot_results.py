"""
Publication-quality BPC vs. context depth plot.

Reads (auto-discovered unless overridden):
  experiments/results/full_results_*.json       — CTW + GPT-2 small
  experiments/results/ngram_results.json        — N-gram KT
  experiments/results/gpt2_scaling_results.json — GPT-2 large
  experiments/results/ptb_results_*.json        — text8 results (optional)

Saves:
  experiments/figures/bpc_plot.pdf / .png — single WikiText-2 panel
  experiments/figures/bpc_plot_2panel.pdf/png — optional side-by-side panel

New in this version:
  - Shannon entropy line (~1.1 bpc) with uncertainty band
  - D_max vertical marker (theoretical saturation depth)
  - Optional PTB side-by-side panel

Usage:
    python experiments/plot_results.py
    python experiments/plot_results.py --ptb_json experiments/results/ptb_results_*.json
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import glob
import math

from _paths import FIGURES_DIR, RESULTS_DIR, ensure_artifact_dirs


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def compute_dmax(train_chars: int, A: int = 28, n_min: int = 15) -> float:
    """Theoretical maximum useful CTW depth: D_max = log_A(N / n_min)."""
    return math.log(train_chars / n_min) / math.log(A)


# -----------------------------------------------------------------------
# Core plotting function — draws one dataset panel onto ax
# -----------------------------------------------------------------------

def draw_panel(
    ax,
    ctw_data: dict,
    ngram_data: dict | None,
    gpt2_extra: dict | None,
    title: str,
    show_shannon: bool = True,
    show_dmax: bool = True,
    show_ylabel: bool = True,
    show_legend: bool = True,
):
    import matplotlib.ticker as ticker

    all_bpc = []

    # ---- CTW curve ----
    ctw_depths = [r["depth"] for r in ctw_data["ctw_results"]]
    ctw_bpc    = [r["bpc"]   for r in ctw_data["ctw_results"]]
    all_bpc.extend(ctw_bpc)
    ngram_lookup = ({r["context_depth"]: r["bpc"]
                     for r in ngram_data["ngram_results"]}
                    if ngram_data else {})

    ax.plot(ctw_depths, ctw_bpc,
            "o-", color="#1565C0", linewidth=2, markersize=7,
            label="CTW (this work)", zorder=4)
    for d, b in zip(ctw_depths, ctw_bpc):
        label_offset = -15 if d in ngram_lookup and b < ngram_lookup[d] else 9
        ax.annotate(f"{b:.3f}", (d, b),
                    textcoords="offset points", xytext=(0, label_offset),
                    ha="center", fontsize=7.5, color="#1565C0")

    # ---- N-gram curve ----
    if ngram_data:
        ng_depths = [r["context_depth"] for r in ngram_data["ngram_results"]]
        ng_bpc    = [r["bpc"]           for r in ngram_data["ngram_results"]]
        all_bpc.extend(ng_bpc)
        ax.plot(ng_depths, ng_bpc,
                "s--", color="#E65100", linewidth=2, markersize=7,
                label="N-gram KT (baseline)", zorder=4)
        for d, b in zip(ng_depths, ng_bpc):
            ctw_at_depth = dict(zip(ctw_depths, ctw_bpc)).get(d)
            label_offset = 9 if ctw_at_depth is not None and b > ctw_at_depth else -15
            ax.annotate(f"{b:.3f}", (d, b),
                        textcoords="offset points", xytext=(0, label_offset),
                        ha="center", fontsize=7.5, color="#E65100")

    # ---- GPT-2 horizontal lines ----
    gpt2_lines = []

    if ctw_data.get("gpt2_result") and ctw_data["gpt2_result"]:
        g = ctw_data["gpt2_result"]
        gpt2_lines.append(("GPT-2 small", g["bpc_per_char"], "#2E7D32"))
        all_bpc.append(g["bpc_per_char"])

    # Also check gpt2_results (list, used in ptb_experiment.py)
    for g in ctw_data.get("gpt2_results", []):
        label = g["model"].replace("gpt2-", "GPT-2 ").replace("gpt2", "GPT-2 small")
        if any(label == l for l, _, _ in gpt2_lines):
            continue
        color = {"GPT-2 small": "#2E7D32", "GPT-2 large": "#827717"}.get(label, "#4A148C")
        gpt2_lines.append((label, g["bpc_per_char"], color))
        all_bpc.append(g["bpc_per_char"])

    if gpt2_extra:
        scale_labels = {"gpt2": "GPT-2 small", "gpt2-medium": "GPT-2 medium",
                        "gpt2-large": "GPT-2 large", "gpt2-xl": "GPT-2 XL"}
        scale_colors = {"gpt2": "#2E7D32", "gpt2-medium": "#558B2F",
                        "gpt2-large": "#827717", "gpt2-xl": "#4A148C"}
        for g in gpt2_extra.get("gpt2_results", []):
            bpc = g.get("bpc_per_char")
            if bpc is None or not math.isfinite(bpc):
                continue
            label = scale_labels.get(g["model"], g["model"])
            if any(label == l for l, _, _ in gpt2_lines):
                continue
            color = scale_colors.get(g["model"], "#6A1B9A")
            gpt2_lines.append((label, bpc, color))
            all_bpc.append(bpc)

    x_min = min(ctw_depths) - 0.8
    x_max = max(ctw_depths) + 2.0

    right_label_positions = []
    for label, bpc_val, color in gpt2_lines:
        ax.axhline(bpc_val, color=color, linestyle=":", linewidth=2,
                   alpha=0.9, label=f"{label}  ({bpc_val:.4f})", zorder=3)
        label_y = bpc_val + 0.018
        while any(abs(label_y - existing) < 0.055 for existing in right_label_positions):
            label_y += 0.055
        right_label_positions.append(label_y)
        ax.text(x_max - 0.1, label_y, f"{bpc_val:.4f}",
                ha="right", fontsize=7.5, color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))

    # ---- Shannon entropy line ----
    if show_shannon:
        SHANNON_CENTRAL = 1.1   # Brown et al. (1992) / Schürmann & Grassberger
        SHANNON_LO      = 1.0
        SHANNON_HI      = 1.3   # Shannon (1951) upper bound
        ax.axhline(SHANNON_CENTRAL, color="#B71C1C", linestyle="-.",
                   linewidth=1.5, alpha=0.85,
                   label=f"Shannon H(English) ≈ {SHANNON_CENTRAL} bpc", zorder=2)
        ax.axhspan(SHANNON_LO, SHANNON_HI, alpha=0.07, color="#B71C1C", zorder=1)
        shannon_label_y = SHANNON_CENTRAL + 0.018
        while any(abs(shannon_label_y - existing) < 0.055
                  for existing in right_label_positions):
            shannon_label_y += 0.055
        ax.text(x_max - 0.1, shannon_label_y,
                f"H ≈ {SHANNON_CENTRAL}", ha="right", fontsize=7.5,
                color="#B71C1C",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))
        all_bpc.extend([SHANNON_LO, SHANNON_HI])

    # ---- D_max vertical marker ----
    if show_dmax and ctw_data.get("train_chars"):
        dmax = compute_dmax(ctw_data["train_chars"])
        ax.axvline(dmax, color="#78909C", linestyle="--", linewidth=1.2,
                   alpha=0.7, zorder=2)
        ax.text(dmax + 0.12,
                min(all_bpc) - 0.05,
                f"$D_{{\\max}}$≈{dmax:.1f}",
                fontsize=8, color="#546E7A", va="bottom")

    # ---- Gap annotation (best CTW ↔ best GPT-2) ----
    if gpt2_lines and ctw_bpc:
        best_ctw = min(ctw_bpc)
        best_d   = ctw_depths[ctw_bpc.index(best_ctw)]
        best_gpt2_bpc = min(b for _, b, _ in gpt2_lines)
        gap = best_ctw - best_gpt2_bpc
        mid_y = (best_ctw + best_gpt2_bpc) / 2
        x_ann = max(ctw_depths) + 1.2
        ax.annotate("", xy=(x_ann, best_gpt2_bpc), xytext=(x_ann, best_ctw),
                    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
        ax.text(x_ann + 0.15, mid_y,
                f"Δ={gap:+.2f}\n({2**gap:.1f}×)",
                va="center", ha="left", fontsize=8, color="#333")

    ax.set_xlim(x_min, x_max)
    y_lo = min(all_bpc) - 0.12
    y_hi = max(all_bpc) + 0.35
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Context depth (chars)", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("Bits per character (BPC) ↓", fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if show_legend:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9, edgecolor="#ccc")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctw_json",         default=None)
    parser.add_argument("--ngram_json",        default=None)
    parser.add_argument("--gpt2_scaling_json", default=None)
    parser.add_argument("--ptb_json",          default=None,
                        help="Path to ptb_results_*.json for side-by-side comparison")
    parser.add_argument("--no_shannon",  action="store_true",
                        help="Hide Shannon entropy line")
    parser.add_argument("--no_dmax",     action="store_true",
                        help="Hide D_max marker")
    parser.add_argument("--out_prefix",        default=None)
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib")
        return

    ensure_artifact_dirs()

    # Auto-discover files
    ctw_json     = args.ctw_json     or find_latest(os.path.join(str(RESULTS_DIR), "full_results_*.json"))
    ngram_json   = args.ngram_json   or os.path.join(str(RESULTS_DIR), "ngram_results.json")
    scaling_json = args.gpt2_scaling_json or os.path.join(str(RESULTS_DIR), "gpt2_scaling_results.json")
    ptb_json     = args.ptb_json     or find_latest(os.path.join(str(RESULTS_DIR), "ptb_results_*.json"))
    out_prefix   = args.out_prefix   or os.path.join(str(FIGURES_DIR), "bpc_plot")

    if not ctw_json or not os.path.exists(ctw_json):
        print("ERROR: no full_results_*.json found. Run text_perplexity.py first.")
        return

    wt2_data     = load_json(ctw_json)
    ngram_data   = load_json(ngram_json)   if os.path.exists(ngram_json)   else None
    scaling_data = load_json(scaling_json) if os.path.exists(scaling_json) else None
    ptb_data     = load_json(ptb_json)     if ptb_json and os.path.exists(ptb_json) else None

    show_shannon = not args.no_shannon
    show_dmax    = not args.no_dmax

    if ptb_data:
        # Two-panel figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
        draw_panel(ax1, wt2_data, ngram_data, scaling_data,
                   title="WikiText-2",
                   show_shannon=show_shannon, show_dmax=show_dmax,
                   show_ylabel=True, show_legend=True)

        # Build PTB ngram/gpt2 from ptb_data itself
        ptb_ngram = {"ngram_results": ptb_data.get("ngram_results", [])} if ptb_data.get("ngram_results") else None
        # For PTB GPT-2: embed gpt2_results inside a structure draw_panel can read
        ptb_for_panel = dict(ptb_data)
        draw_panel(ax2, ptb_for_panel, ptb_ngram, None,
                   title="text8 (enwiki)",
                   show_shannon=show_shannon, show_dmax=show_dmax,
                   show_ylabel=False, show_legend=True)

        plt.suptitle("Long-Range Dependency Gap: Classical vs. Neural LM",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        out_prefix_2p = out_prefix.replace("bpc_plot", "bpc_plot_2panel")
        for ext in ("pdf", "png"):
            path = f"{out_prefix_2p}.{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved → {path}")
        plt.close(fig)

    # Always also produce single-panel WikiText-2 plot
    fig, ax = plt.subplots(figsize=(7.5, 5))
    draw_panel(ax, wt2_data, ngram_data, scaling_data,
               title="WikiText-2 — Long-Range Dependency Gap",
               show_shannon=show_shannon, show_dmax=show_dmax,
               show_ylabel=True, show_legend=True)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{out_prefix}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)

    # ---- Summary table ----
    print(f"\n{'='*65}")
    print(f"{'Model':<22}  {'Dataset':>12}  {'Context':>10}  {'BPC':>8}")
    print("-" * 58)
    if ngram_data:
        for r in ngram_data["ngram_results"]:
            print(f"{'N-gram ('+str(r['order'])+'-gram)':<22}  "
                  f"{'WikiText-2':>12}  "
                  f"{str(r['context_depth'])+' chars':>10}  "
                  f"{r['bpc']:>8.4f}")
    for r in wt2_data["ctw_results"]:
        print(f"{'CTW D='+str(r['depth']):<22}  "
              f"{'WikiText-2':>12}  "
              f"{str(r['depth'])+' chars':>10}  "
              f"{r['bpc']:>8.4f}")
    if wt2_data.get("gpt2_result"):
        g = wt2_data["gpt2_result"]
        print(f"{'GPT-2 small':<22}  {'WikiText-2':>12}  "
              f"{'~5000 chars':>10}  {g['bpc_per_char']:>8.4f}")
    if scaling_data:
        for g in scaling_data.get("gpt2_results", []):
            bpc = g.get("bpc_per_char")
            if bpc and math.isfinite(bpc):
                label = g["model"].replace("gpt2-", "GPT-2 ").replace("gpt2", "GPT-2 small")
                print(f"{label:<22}  {'WikiText-2':>12}  "
                      f"{'~5000 chars':>10}  {bpc:>8.4f}")
    print(f"\n  Shannon H(English) ≈ 1.1 bpc  (range 1.0–1.3, Brown et al. 1992)")


if __name__ == "__main__":
    main()
