"""
Experiment 2 — NL Perplexity: CTW vs. GPT-2 (full WikiText-2)

Runs CTW at multiple depths AND GPT-2 on the same normalized text,
then saves everything to a JSON file for later analysis.

Usage:
    python experiments/text_perplexity.py                         # full dataset
    python experiments/text_perplexity.py --no_gpt2              # CTW only
    python experiments/text_perplexity.py --depths 3 5 7 10      # custom depths
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import math
import re
import time
import unicodedata
from datetime import datetime

from _paths import result_path


# -----------------------------------------------------------------------
# Text normalization
# -----------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """28-char alphabet: [a-z], space, newline."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r'[^a-z \n]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

def load_wikitext2() -> tuple[str, str]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(ds["train"]["text"])
    val_text   = "\n".join(ds["validation"]["text"])
    return train_text, val_text


# -----------------------------------------------------------------------
# CTW
# -----------------------------------------------------------------------

def run_ctw(train_text: str, val_text: str, depth: int, vocab: list[str]) -> dict:
    from ctw.text_ctw import TextCTW

    ctw = TextCTW(depth=depth, vocab=vocab)

    t0 = time.time()
    for t, ch in enumerate(train_text):
        ctx = list(train_text[max(0, t - depth) : t])
        ctw.update(ch, ctx)
        if (t + 1) % 500_000 == 0:
            print(f"    [{t+1:,}/{len(train_text):,}]  "
                  f"{(time.time()-t0):.0f}s elapsed")
    train_time = time.time() - t0

    t0 = time.time()
    total_bits = 0.0
    for t, ch in enumerate(val_text):
        ctx = list(val_text[max(0, t - depth) : t])
        total_bits += ctw.log_loss(ch, ctx)
    eval_time = time.time() - t0

    bpc = total_bits / len(val_text)
    return {
        "depth":       depth,
        "bpc":         bpc,
        "perplexity":  2 ** bpc,
        "train_time_s": round(train_time, 1),
        "eval_time_s":  round(eval_time, 1),
    }


# -----------------------------------------------------------------------
# GPT-2  →  bits per CHARACTER
# -----------------------------------------------------------------------

def run_gpt2(val_text: str, model_name: str = "gpt2") -> dict:
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError:
        print("  torch / transformers not installed — skipping GPT-2.")
        print("  pip install torch transformers")
        return {}

    print(f"  Loading {model_name}...")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model     = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    model     = model.float()   # ensure float32 — prevents NaN from half-precision

    # Sanity check: catch corrupted cached weights
    has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
    if has_nan:
        print(f"  ERROR: {model_name} weights contain NaN — delete cache and retry:")
        print(f"    rm -rf ~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}")
        return {}

    encodings       = tokenizer(val_text, return_tensors="pt")
    input_ids       = encodings.input_ids
    n_tokens        = input_ids.size(1)
    n_chars         = len(val_text)
    chars_per_token = n_chars / n_tokens
    print(f"  {n_chars:,} chars  |  {n_tokens:,} tokens  "
          f"|  {chars_per_token:.2f} chars/token")

    max_length, stride = 1024, 512
    nlls, counted = [], 0
    nan_batches   = 0

    t0 = time.time()
    print("  Running GPT-2 inference...")
    for i in range(0, n_tokens, stride):
        begin      = max(i + stride - max_length, 0)
        end        = min(i + stride, n_tokens)
        chunk      = input_ids[:, begin:end]
        target_len = end - i

        with torch.no_grad():
            logits       = model(chunk).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = chunk[..., 1:].contiguous()
            loss_fn      = torch.nn.CrossEntropyLoss(reduction="sum")
            nll          = loss_fn(
                shift_logits[..., -target_len:, :].view(-1, shift_logits.size(-1)),
                shift_labels[..., -target_len:].view(-1),
            )

        nll_val = nll.item()
        if not math.isfinite(nll_val):
            nan_batches += 1
            continue   # skip corrupted batch rather than poison the sum
        nlls.append(nll_val)
        counted += target_len

        if (len(nlls)) % 20 == 0:
            print(f"    [{counted:,}/{n_tokens:,} tokens]  "
                  f"{(time.time()-t0):.0f}s elapsed")

    if nan_batches:
        print(f"  Warning: {nan_batches} batches produced NaN/inf and were skipped.")
    if not nlls:
        print("  ERROR: all batches returned NaN — likely corrupted model cache.")
        return {}

    nats_per_token = sum(nlls) / counted
    bpc_per_token  = nats_per_token / math.log(2)
    bpc_per_char   = bpc_per_token / chars_per_token

    return {
        "model":            model_name,
        "n_tokens":         n_tokens,
        "n_chars":          n_chars,
        "chars_per_token":  round(chars_per_token, 3),
        "ppl_token":        round(math.exp(nats_per_token), 4),
        "bpc_per_token":    round(bpc_per_token, 6),
        "bpc_per_char":     round(bpc_per_char, 6),
        "perplexity_char":  round(2 ** bpc_per_char, 4),
        "eval_time_s":      round(time.time() - t0, 1),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths",    type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--no_gpt2",  action="store_true")
    parser.add_argument("--gpt2_model", default="gpt2",
                        help="HuggingFace model name (default: gpt2)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: auto-named in experiments/results/)")
    args = parser.parse_args()

    # Output file
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = result_path(f"full_results_{ts}.json")

    results = {
        "timestamp":   datetime.now().isoformat(),
        "dataset":     "WikiText-2 (full, normalized to 28-char alphabet)",
        "ctw_results": [],
        "gpt2_result": {},
    }

    # ---- Load & normalize ----
    print("Loading WikiText-2 (full dataset)...")
    train_raw, val_raw = load_wikitext2()
    print(f"  Raw  — train: {len(train_raw):,} chars  |  val: {len(val_raw):,} chars")

    train_text = normalize_text(train_raw)
    val_text   = normalize_text(val_raw)
    print(f"  Norm — train: {len(train_text):,} chars  |  val: {len(val_text):,} chars")

    vocab = sorted(set(train_text + val_text))
    results["train_chars"] = len(train_text)
    results["val_chars"]   = len(val_text)
    results["vocab_size"]  = len(vocab)
    print(f"  Vocab: {vocab}\n  Size: {len(vocab)}")

    # ---- CTW ----
    print("\n" + "=" * 60)
    print("CTW (character-level, 28-char alphabet)")
    print("=" * 60)
    for d in args.depths:
        print(f"\n--- D = {d} ---")
        r = run_ctw(train_text, val_text, d, vocab)
        results["ctw_results"].append(r)
        print(f"  BPC = {r['bpc']:.4f}  |  Perplexity = {r['perplexity']:.2f}  "
              f"|  train {r['train_time_s']}s  eval {r['eval_time_s']}s")
        # Save after each depth so partial results are preserved
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---- GPT-2 ----
    if not args.no_gpt2:
        print("\n" + "=" * 60)
        print(f"GPT-2 ({args.gpt2_model}, pretrained, no fine-tuning)")
        print("=" * 60)
        gpt2 = run_gpt2(val_text, model_name=args.gpt2_model)
        if gpt2:
            results["gpt2_result"] = gpt2
            print(f"\n  BPC/char  = {gpt2['bpc_per_char']:.4f}")
            print(f"  PPL/token = {gpt2['ppl_token']:.2f}")
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Long-Range Dependency Gap")
    print("=" * 70)
    print(f"{'Model':<20} {'BPC/char':>10} {'PPL/char':>10} {'Context':>12}")
    print("-" * 54)
    for r in results["ctw_results"]:
        print(f"{'CTW D='+str(r['depth']):<20} {r['bpc']:>10.4f} "
              f"{r['perplexity']:>10.2f} {str(r['depth'])+' chars':>12}")
    if results["gpt2_result"]:
        g = results["gpt2_result"]
        print(f"{'GPT-2 small':<20} {g['bpc_per_char']:>10.4f} "
              f"{g['perplexity_char']:>10.2f} {'~5000 chars':>12}")
        best_ctw = min(results["ctw_results"], key=lambda x: x["bpc"])
        gap = best_ctw["bpc"] - g["bpc_per_char"]
        print(f"\n  Best CTW (D={best_ctw['depth']}): BPC = {best_ctw['bpc']:.4f}")
        print(f"  GPT-2:                   BPC = {g['bpc_per_char']:.4f}")
        print(f"  Gap:                        = {gap:+.4f} bits/char")
        print(f"  CTW is {2**gap:.1f}x worse in char-perplexity than GPT-2")

    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
