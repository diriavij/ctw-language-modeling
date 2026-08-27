"""
Second-dataset experiment (text8) — mirrors text_perplexity.py.
Runs CTW, n-gram KT, and GPT-2 on text8 to verify the gap is dataset-agnostic.

text8 is a standard character-level LM benchmark derived from English Wikipedia
with a different preprocessing than WikiText-2 (lowercase letters + spaces only,
no punctuation, no section headers).  It is widely used in character-level LM
papers as a complement to WikiText-2.

Source: http://mattmahoney.net/dc/text8.zip (~100 MB, downloaded automatically).

Usage:
    python experiments/ptb_experiment.py
    python experiments/ptb_experiment.py --no_gpt2
    python experiments/ptb_experiment.py --depths 3 5 7 10 --orders 4 6 8 11
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import time
from datetime import datetime

from text_perplexity import normalize_text, run_ctw, run_gpt2
from ngram_baseline import FastCharNgramKT
from _paths import result_path


# -----------------------------------------------------------------------

def load_text8(max_train: int = 2_000_000, max_val: int = 200_000) -> tuple[str, str]:
    """
    Download text8 from mattmahoney.net (~100 MB) and cache locally.
    Uses first max_train chars for training, next max_val chars for validation.
    text8 is already lowercase letters + spaces; normalize_text is still applied
    for consistency with WikiText-2 experiments.
    """
    import urllib.request
    import zipfile

    cache_dir  = os.path.expanduser("~/.cache")
    cache_file = os.path.join(cache_dir, "text8.txt")
    zip_file   = cache_file + ".zip"

    if not os.path.exists(cache_file):
        url = "http://mattmahoney.net/dc/text8.zip"
        print(f"  Downloading text8 from {url} (~100 MB)...")
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(url, zip_file)
        print("  Extracting...")
        with zipfile.ZipFile(zip_file) as zf:
            text = zf.read("text8").decode("utf-8")
        with open(cache_file, "w") as f:
            f.write(text)
        os.remove(zip_file)
        print(f"  Cached to {cache_file}")
    else:
        print(f"  Loading from cache: {cache_file}")
        with open(cache_file) as f:
            text = f.read()

    print(f"  text8 total: {len(text):,} chars")
    train = text[:max_train]
    val   = text[max_train : max_train + max_val]
    return train, val


def run_ngram(train_text: str, val_text: str, order: int, vocab: list[str]) -> dict:
    model = FastCharNgramKT(order=order, vocab=vocab)
    t0 = time.time()
    model.fit(train_text)
    train_time = time.time() - t0
    t0 = time.time()
    bpc = model.eval_bpc(val_text)
    eval_time = time.time() - t0
    return {
        "order":         order,
        "context_depth": order - 1,
        "bpc":           bpc,
        "perplexity":    2 ** bpc,
        "train_time_s":  round(train_time, 1),
        "eval_time_s":   round(eval_time, 1),
    }


# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths",      type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--orders",      type=int, nargs="+", default=[4, 6, 8, 11])
    parser.add_argument("--no_gpt2",    action="store_true")
    parser.add_argument("--gpt2_models", nargs="+", default=["gpt2", "gpt2-large"])
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = result_path(f"ptb_results_{ts}.json")

    print("Loading text8 (English Wikipedia, character-level benchmark)...")
    train_raw, val_raw = load_text8()
    train_text = normalize_text(train_raw)
    val_text   = normalize_text(val_raw)
    vocab = sorted(set(train_text + val_text))
    print(f"  Train: {len(train_text):,} chars  |  Val: {len(val_text):,} chars  "
          f"|  Vocab: {len(vocab)}")

    results = {
        "timestamp":     datetime.now().isoformat(),
        "dataset":       "text8 (enwiki, normalized to 28-char alphabet)",
        "train_chars":   len(train_text),
        "val_chars":     len(val_text),
        "vocab_size":    len(vocab),
        "ctw_results":   [],
        "ngram_results": [],
        "gpt2_results":  [],
    }

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---- CTW ----
    print("\n" + "=" * 55)
    print("CTW (character-level)")
    print("=" * 55)
    for d in args.depths:
        print(f"\n--- D = {d} ---")
        r = run_ctw(train_text, val_text, d, vocab)
        results["ctw_results"].append(r)
        print(f"  BPC = {r['bpc']:.4f}  |  PPL = {r['perplexity']:.2f}  "
              f"|  train {r['train_time_s']}s  eval {r['eval_time_s']}s")
        save()

    # ---- N-gram ----
    print("\n" + "=" * 55)
    print("N-gram KT")
    print("=" * 55)
    for order in args.orders:
        print(f"\n--- {order}-gram (context depth {order - 1}) ---")
        r = run_ngram(train_text, val_text, order, vocab)
        results["ngram_results"].append(r)
        print(f"  BPC = {r['bpc']:.4f}  |  PPL = {r['perplexity']:.2f}  "
              f"|  train {r['train_time_s']}s  eval {r['eval_time_s']}s")
        save()

    # ---- GPT-2 ----
    if not args.no_gpt2:
        print("\n" + "=" * 55)
        print("GPT-2 (pretrained)")
        print("=" * 55)
        for model_name in args.gpt2_models:
            print(f"\n--- {model_name} ---")
            r = run_gpt2(val_text, model_name=model_name)
            if r:
                results["gpt2_results"].append(r)
                print(f"  BPC/char = {r['bpc_per_char']:.4f}  |  PPL/token = {r['ppl_token']:.2f}")
                save()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY — PTB")
    print(f"{'='*60}")
    print(f"{'Model':<22}  {'Context':>10}  {'BPC':>8}  {'PPL':>8}")
    print("-" * 56)
    for r in results["ngram_results"]:
        print(f"{'N-gram ('+str(r['order'])+'-gram)':<22}  "
              f"{str(r['context_depth'])+' chars':>10}  "
              f"{r['bpc']:>8.4f}  {r['perplexity']:>8.2f}")
    for r in results["ctw_results"]:
        print(f"{'CTW D='+str(r['depth']):<22}  "
              f"{str(r['depth'])+' chars':>10}  "
              f"{r['bpc']:>8.4f}  {r['perplexity']:>8.2f}")
    for r in results["gpt2_results"]:
        print(f"{r['model']:<22}  {'~5000 chars':>10}  "
              f"{r['bpc_per_char']:>8.4f}  {r['perplexity_char']:>8.2f}")

    print(f"\nResults saved → {args.out}")
    print("\nNote: to generate the two-panel plot (WikiText-2 + text8), run:")
    print("  python experiments/plot_results.py")


if __name__ == "__main__":
    main()
