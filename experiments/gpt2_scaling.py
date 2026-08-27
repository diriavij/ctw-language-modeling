"""
GPT-2 scaling experiment: small → medium → large
Evaluates BPC/char on WikiText-2 validation (normalized, 28-char alphabet).
All models are pretrained (no fine-tuning).

Usage:
    python experiments/gpt2_scaling.py
    python experiments/gpt2_scaling.py --models gpt2 gpt2-medium
    python experiments/gpt2_scaling.py --models gpt2 gpt2-medium gpt2-large gpt2-xl
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))   # for text_perplexity import

import argparse
import json
from datetime import datetime

from text_perplexity import normalize_text, load_wikitext2, run_gpt2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["gpt2", "gpt2-medium", "gpt2-large"],
        help="HuggingFace model names to evaluate (default: small/medium/large)"
    )
    parser.add_argument("--out", default=None,
                        help="Output JSON (default: experiments/gpt2_scaling_results.json)")
    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(__file__), "gpt2_scaling_results.json")

    print("Loading WikiText-2 (validation only)...")
    _, val_raw = load_wikitext2()
    val_text = normalize_text(val_raw)
    print(f"  Val: {len(val_text):,} chars")

    results = {
        "timestamp":    datetime.now().isoformat(),
        "dataset":      "WikiText-2 (full, normalized to 28-char alphabet)",
        "gpt2_results": [],
    }

    for model_name in args.models:
        print(f"\n{'='*55}")
        print(f"Model: {model_name}")
        print(f"{'='*55}")
        r = run_gpt2(val_text, model_name=model_name)
        if r:
            results["gpt2_results"].append(r)
            print(f"  BPC/char  = {r['bpc_per_char']:.4f}")
            print(f"  PPL/token = {r['ppl_token']:.2f}")
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*55}")
    print("GPT-2 SCALING SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<20}  {'BPC/char':>9}  {'PPL/token':>10}  {'PPL/char':>10}")
    print("-" * 55)
    for r in results["gpt2_results"]:
        print(f"{r['model']:<20}  {r['bpc_per_char']:>9.4f}  "
              f"{r['ppl_token']:>10.2f}  {r['perplexity_char']:>10.2f}")

    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
