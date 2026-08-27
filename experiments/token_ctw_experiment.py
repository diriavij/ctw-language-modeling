"""
Token-level CTW experiments.

Compares three token-granularity variants against the existing char-level CTW:

  Variant 1 — Word-level CTW
      KT n-gram over words (vocab ≈ 5000).  D = 1, 2, 3.
      Metric: bits per word → converted to BPC.

  Variant 2 — BPE CTW
      KT n-gram over GPT-2 BPE tokens (vocab 50 257).  D = 1.
      Requires: pip install transformers
      Metric: bits per token → BPC.

  Variant 3 — Hierarchical CTW
      Word-level model at word boundaries (first char of each word).
      Char-level CTW within words (same TextCTW as baseline).
      Directly addresses the gap found in gap_analysis.py.
      Metric: full BPC (every character predicted).

Results are saved under experiments/results/ and figures under experiments/figures/.

Usage:
    python experiments/token_ctw_experiment.py
    python experiments/token_ctw_experiment.py --word_depths 1 2 3 --vocab_size 5000
    python experiments/token_ctw_experiment.py --no_bpe       # skip BPE (no transformers)
    python experiments/token_ctw_experiment.py --full         # 2M train chars
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime

from ctw.text_ctw import TextCTW
from text_perplexity import normalize_text, load_wikitext2
from _paths import RESULTS_DIR, figure_for_result, figure_path, result_path


# -----------------------------------------------------------------------
# Vocabulary & tokenization helpers
# -----------------------------------------------------------------------

def word_split(text):
    """Split normalized text into word tokens (whitespace delimiters)."""
    return text.split()


def build_word_vocab(train_text, max_vocab=5000):
    """Return (vocab list, word→idx dict).  vocab[0] = '<UNK>'."""
    counts = Counter(word_split(train_text))
    top = [w for w, _ in counts.most_common(max_vocab - 1)]
    vocab = ['<UNK>'] + top
    word2idx = {w: i for i, w in enumerate(vocab)}
    return vocab, word2idx


def encode_words(text, word2idx):
    """Return list of word strings with OOV mapped to '<UNK>'."""
    unk = '<UNK>'
    return [w if w in word2idx else unk for w in word_split(text)]


# -----------------------------------------------------------------------
# Token-level KT n-gram  (word-level or BPE-level)
# -----------------------------------------------------------------------

class TokenNgramKT:
    """
    KT-smoothed n-gram for any token sequence.
    Standard KT: P(tok|ctx) = (count(tok,ctx) + α) / (count(ctx) + A·α)
    with α = 0.5 (same as char-level CTW leaves).
    """

    def __init__(self, order, vocab, alpha=0.5):
        self.order = order
        self.vocab = list(vocab)
        self.A     = len(vocab)
        self.alpha = alpha
        self._counts = {}   # tuple(ctx) → {tok: int}
        self._totals = {}   # tuple(ctx) → int

    def fit(self, tokens):
        """tokens: list of strings (words or BPE token strings)."""
        for i, tok in enumerate(tokens):
            for o in range(self.order + 1):
                ctx = tuple(tokens[max(0, i - o):i])
                if ctx not in self._counts:
                    self._counts[ctx] = {}
                    self._totals[ctx] = 0
                self._counts[ctx][tok] = self._counts[ctx].get(tok, 0) + 1
                self._totals[ctx] += 1

    def _best_ctx(self, ctx_tokens):
        """Return the longest context tuple that exists in the count table."""
        for o in range(min(len(ctx_tokens), self.order), -1, -1):
            ctx = tuple(ctx_tokens[-o:]) if o > 0 else ()
            if ctx in self._counts:
                return ctx
        return ()

    def log_loss(self, tok, ctx_tokens):
        """-log2 P(tok | ctx_tokens), with KT + backoff."""
        ctx   = self._best_ctx(ctx_tokens)
        n     = self._counts.get(ctx, {}).get(tok, 0)
        total = self._totals.get(ctx, 0)
        p     = (n + self.alpha) / (total + self.A * self.alpha)
        return -math.log2(p)

    def eval_bpt(self, tokens):
        """Bits per token on the full token sequence."""
        total = 0.0
        for i, tok in enumerate(tokens):
            ctx = tokens[max(0, i - self.order):i]
            total += self.log_loss(tok, ctx)
        return total / len(tokens) if tokens else float('nan')

    # --- efficient char-marginalization (used by HierarchicalCTW) ---

    def char_dist(self, ctx_tokens, char_to_words, unk_char_dist):
        """
        P(first_char | ctx) = Σ_{w: w[0]==c} P(w | ctx)

        Uses the sparse count table for efficiency:
        only iterates over words actually seen in this context.
        """
        ctx       = self._best_ctx(ctx_tokens)
        ctx_cnts  = self._counts.get(ctx, {})
        ctx_total = self._totals.get(ctx, 0)
        denom     = ctx_total + self.A * self.alpha

        # Aggregate counts by first character
        char_sum = defaultdict(float)   # c → sum of counts of words starting with c
        for word, cnt in ctx_cnts.items():
            if word != '<UNK>' and word:
                char_sum[word[0]] += cnt

        # P(c) = (sum_counts[c] + |words_starting_with_c| * alpha) / denom
        char_probs = {}
        for c, words in char_to_words.items():
            char_probs[c] = (char_sum.get(c, 0) + len(words) * self.alpha) / denom

        # UNK contribution (unknown words) → spread over chars by unk_char_dist
        unk_cnt  = ctx_cnts.get('<UNK>', 0)
        unk_prob = (unk_cnt + self.alpha) / denom
        for c, frac in unk_char_dist.items():
            char_probs[c] = char_probs.get(c, 0) + unk_prob * frac

        return char_probs


# -----------------------------------------------------------------------
# Variant 1: Word-level CTW
# -----------------------------------------------------------------------

def run_word_ctw(train_text, val_text, depth, vocab_size=5000):
    print(f"\n  Building vocab (top {vocab_size} words)...")
    vocab, word2idx = build_word_vocab(train_text, vocab_size)

    train_tokens = encode_words(train_text, word2idx)
    val_tokens   = encode_words(val_text,   word2idx)

    cov = sum(1 for w in train_tokens if w != '<UNK>') / len(train_tokens)
    print(f"  Vocab={len(vocab)}  train coverage={cov*100:.1f}%")

    t0 = time.time()
    model = TokenNgramKT(order=depth, vocab=vocab)
    model.fit(train_tokens)
    train_time = time.time() - t0

    t0 = time.time()
    bpt = model.eval_bpt(val_tokens)
    eval_time = time.time() - t0

    # BPC: word bits ÷ total val chars
    # (spaces not predicted by word model; allocated proportionally)
    n_words  = len(val_tokens)
    bpc      = (bpt * n_words) / len(val_text)
    avg_wlen = sum(len(w) for w in word_split(val_text)) / max(n_words, 1)

    print(f"  BPT={bpt:.4f}  avg_word_len={avg_wlen:.2f}  → BPC≈{bpc:.4f}  "
          f"train {train_time:.1f}s  eval {eval_time:.1f}s")

    return {
        "variant":      f"Word-CTW D={depth}",
        "type":         "word",
        "depth":        depth,
        "vocab_size":   len(vocab),
        "bpt":          bpt,
        "bpc":          bpc,
        "avg_word_len": round(avg_wlen, 3),
        "ppl_word":     2 ** bpt,
        "train_time_s": round(train_time, 1),
        "eval_time_s":  round(eval_time, 1),
    }


# -----------------------------------------------------------------------
# Variant 2: BPE CTW (GPT-2 tokenizer)
# -----------------------------------------------------------------------

def run_bpe_ctw(train_text, val_text, depth=1):
    try:
        from transformers import GPT2TokenizerFast
    except ImportError:
        print("  transformers not installed — skipping BPE variant.")
        return None

    print(f"\n  Loading GPT-2 tokenizer...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    print(f"  Encoding train ({len(train_text):,} chars)...")
    t0 = time.time()
    train_ids = tok.encode(train_text)
    val_ids   = tok.encode(val_text)
    enc_time  = time.time() - t0

    # Use decoded token strings as vocabulary keys (keeps it consistent)
    train_tokens = [tok.decode([i]) for i in train_ids]
    val_tokens   = [tok.decode([i]) for i in val_ids]

    # Build vocab from training ids
    vocab_ids  = sorted(set(train_ids) | set(val_ids))
    vocab      = [tok.decode([i]) for i in vocab_ids]
    print(f"  BPE vocab used={len(vocab)}  train tokens={len(train_ids):,}  "
          f"val tokens={len(val_ids):,}  ({enc_time:.1f}s encoding)")

    t0 = time.time()
    model = TokenNgramKT(order=depth, vocab=vocab)
    model.fit(train_tokens)
    train_time = time.time() - t0

    t0 = time.time()
    bpt = model.eval_bpt(val_tokens)
    eval_time = time.time() - t0

    chars_per_token = len(val_text) / max(len(val_ids), 1)
    bpc = bpt / chars_per_token

    print(f"  BPT={bpt:.4f}  chars/token={chars_per_token:.2f}  → BPC={bpc:.4f}  "
          f"train {train_time:.1f}s  eval {eval_time:.1f}s")

    return {
        "variant":          f"BPE-CTW D={depth}",
        "type":             "bpe",
        "depth":            depth,
        "vocab_size":       len(vocab),
        "bpt":              bpt,
        "bpc":              bpc,
        "chars_per_token":  round(chars_per_token, 3),
        "ppl_token":        2 ** bpt,
        "train_time_s":     round(train_time, 1),
        "eval_time_s":      round(eval_time, 1),
    }


# -----------------------------------------------------------------------
# Variant 3: Hierarchical CTW
# -----------------------------------------------------------------------

class HierarchicalCTW:
    """
    Char-level CTW within words  +  Word-level KT at word boundaries.

    At word-initial positions (char immediately after a space/newline):
        P(char) = Σ_{w: w[0]==char} P_word(w | word_history)

    At all other positions (spaces, within-word chars):
        P(char) = char CTW (TextCTW, same as baseline)
    """

    def __init__(self, char_depth, word_depth, vocab_size=5000, word_alpha=0.01):
        self.char_depth  = char_depth
        self.word_depth  = word_depth
        self.vocab_size  = vocab_size
        self.word_alpha  = word_alpha

    def train(self, train_text):
        # ---- char CTW ----
        self.char_vocab = sorted(set(train_text))
        self.char_ctw   = TextCTW(depth=self.char_depth, vocab=self.char_vocab)
        print(f"  Training char-CTW D={self.char_depth} on {len(train_text):,} chars...")
        t0 = time.time()
        for t, ch in enumerate(train_text):
            ctx = list(train_text[max(0, t - self.char_depth):t])
            self.char_ctw.update(ch, ctx)
            if (t + 1) % 1_000_000 == 0:
                print(f"    [{t+1:,}]  {time.time()-t0:.0f}s")
        print(f"  Char-CTW trained in {time.time()-t0:.1f}s")

        # ---- word model ----
        print(f"  Building word vocab (top {self.vocab_size})...")
        self.word_vocab, self.word2idx = build_word_vocab(train_text, self.vocab_size)
        train_tokens = encode_words(train_text, self.word2idx)

        t0 = time.time()
        self.word_model = TokenNgramKT(
            order=self.word_depth, vocab=self.word_vocab, alpha=self.word_alpha
        )
        self.word_model.fit(train_tokens)
        print(f"  Word model (alpha={self.word_alpha}) trained in {time.time()-t0:.1f}s")

        # ---- char_to_words: first char → [vocab words] ----
        self.char_to_words = defaultdict(list)
        for w in self.word_vocab:
            if w != '<UNK>' and w:
                self.char_to_words[w[0]].append(w)

        # ---- UNK first-char distribution (from OOV words in training) ----
        oov_words = [w for w in word_split(train_text)
                     if w not in self.word2idx or self.word2idx[w] == 0]
        if oov_words:
            fc_cnt = Counter(w[0] for w in oov_words if w)
            total  = sum(fc_cnt.values())
            self.unk_char_dist = {c: n / total for c, n in fc_cnt.items()}
        else:
            self.unk_char_dist = {c: 1/26 for c in 'abcdefghijklmnopqrstuvwxyz'}

        return self

    def eval(self, val_text):
        """
        Returns (losses: list[float], positions: list[int]).
        positions[t]: -1=space, 0=word-initial, 1+=within word.
        """
        losses    = []
        positions = []

        word_history = []   # last word_depth word strings
        current_word = []   # chars accumulated in current word
        char_pos     = 0    # position within current word

        for t, ch in enumerate(val_text):
            char_ctx = list(val_text[max(0, t - self.char_depth):t])

            if ch in (' ', '\n'):
                # ---- space: use char model ----
                loss = self.char_ctw.log_loss(ch, char_ctx)
                positions.append(-1)

                # finish current word, append to history
                if current_word:
                    w   = ''.join(current_word)
                    wtk = w if w in self.word2idx else '<UNK>'
                    word_history = (word_history + [wtk])[-self.word_depth:]
                    current_word = []
                char_pos = 0    # next letter is word-initial

            elif char_pos == 0:
                # ---- word-initial: marginalize word model → char probs ----
                char_probs = self.word_model.char_dist(
                    word_history[-self.word_depth:],
                    self.char_to_words,
                    self.unk_char_dist,
                )
                p    = max(char_probs.get(ch, 0.0), 1e-12)
                loss = -math.log2(p)
                positions.append(0)
                current_word.append(ch)
                char_pos = 1

            else:
                # ---- within word: use char model ----
                loss = self.char_ctw.log_loss(ch, char_ctx)
                positions.append(char_pos)
                current_word.append(ch)
                char_pos += 1

            losses.append(loss)

        return losses, positions


def run_hierarchical(train_text, val_text, char_depth=5, word_depth=2,
                     vocab_size=5000, word_alpha=0.01):
    print(f"\n  Hierarchical CTW  char_D={char_depth}  word_D={word_depth}  "
          f"vocab={vocab_size}  word_alpha={word_alpha}")

    model = HierarchicalCTW(char_depth, word_depth, vocab_size, word_alpha=word_alpha)
    t0 = time.time()
    model.train(train_text)
    train_time = time.time() - t0

    print(f"  Evaluating on {len(val_text):,} chars...")
    t0 = time.time()
    losses, positions = model.eval(val_text)
    eval_time = time.time() - t0

    bpc = sum(losses) / len(losses)

    # Positional breakdown
    pos_buckets = defaultdict(list)
    for loss, pos in zip(losses, positions):
        key = str(pos) if pos < 6 else "6+"
        pos_buckets[key].append(loss)
    bpc_by_pos = {k: sum(v) / len(v) for k, v in pos_buckets.items()}

    print(f"  BPC={bpc:.4f}  train {train_time:.1f}s  eval {eval_time:.1f}s")
    print(f"  BPC by position: " +
          "  ".join(f"pos{k}={v:.3f}" for k, v in sorted(
              bpc_by_pos.items(),
              key=lambda x: int(x[0].replace('+', '')) if x[0] != '-1' else -1
          )))

    return {
        "variant":      f"Hierarchical char_D={char_depth} word_D={word_depth}",
        "type":         "hierarchical",
        "char_depth":   char_depth,
        "word_depth":   word_depth,
        "vocab_size":   vocab_size,
        "word_alpha":   word_alpha,
        "bpc":          bpc,
        "bpc_by_pos":   bpc_by_pos,
        "train_time_s": round(train_time, 1),
        "eval_time_s":  round(eval_time, 1),
    }


# -----------------------------------------------------------------------
# Variant 4: Full-word Hierarchical CTW
# -----------------------------------------------------------------------

class FullWordHierarchicalCTW:
    """
    True hierarchical decomposition: the word model predicts the ENTIRE
    next word at every word boundary; within-word characters are free.

    Information-theoretic cost:
      - word-initial (pos 0): -log2 P(actual_word | word_history)
      - within-word  (pos 1+): 0.0 bits  (word identity fully determines them)
      - spaces/newlines      : char-CTW cost (same as baseline)

    Key advantage over HierarchicalCTW: no marginalization loss.
    The original model collapses the word distribution onto first-char
    (losing all information that distinguishes words with the same first char).
    This model avoids that by treating the full word as the atomic unit.

    Equivalent BPC formula:
        BPC = (Σ_words  -log2 P(w|ctx)  +  Σ_spaces  char_ctw_cost) / n_chars
    """

    def __init__(self, char_depth, word_depth, vocab_size=5000, word_alpha=0.01):
        self.char_depth = char_depth
        self.word_depth = word_depth
        self.vocab_size = vocab_size
        self.word_alpha = word_alpha

    def train(self, train_text):
        # ---- char CTW (used only for spaces) ----
        self.char_vocab = sorted(set(train_text))
        self.char_ctw   = TextCTW(depth=self.char_depth, vocab=self.char_vocab)
        print(f"  Training char-CTW D={self.char_depth} on {len(train_text):,} chars...")
        t0 = time.time()
        for t, ch in enumerate(train_text):
            ctx = list(train_text[max(0, t - self.char_depth):t])
            self.char_ctw.update(ch, ctx)
            if (t + 1) % 1_000_000 == 0:
                print(f"    [{t+1:,}]  {time.time()-t0:.0f}s")
        print(f"  Char-CTW trained in {time.time()-t0:.1f}s")

        # ---- word model ----
        print(f"  Building word vocab (top {self.vocab_size})...")
        self.word_vocab, self.word2idx = build_word_vocab(train_text, self.vocab_size)
        train_tokens = encode_words(train_text, self.word2idx)

        cov = sum(1 for w in train_tokens if w != '<UNK>') / max(len(train_tokens), 1)
        print(f"  Vocab={len(self.word_vocab)}  train coverage={cov*100:.1f}%")

        t0 = time.time()
        self.word_model = TokenNgramKT(
            order=self.word_depth, vocab=self.word_vocab, alpha=self.word_alpha
        )
        self.word_model.fit(train_tokens)
        print(f"  Word model (alpha={self.word_alpha}) trained in {time.time()-t0:.1f}s")

        return self

    def eval(self, val_text):
        """
        Returns (losses: list[float], positions: list[int]).

        At word-initial position:  look ahead to find the full word,
        call word_model.log_loss(actual_word, word_history).
        Within-word positions:     0.0 bits.
        Spaces/newlines:           char_ctw.log_loss.
        """
        losses    = []
        positions = []

        word_history = []
        current_word = []
        char_pos     = 0

        for t, ch in enumerate(val_text):
            char_ctx = list(val_text[max(0, t - self.char_depth):t])

            if ch in (' ', '\n'):
                # Finish the current word, update history
                if current_word:
                    w   = ''.join(current_word)
                    wtk = w if w in self.word2idx else '<UNK>'
                    word_history = (word_history + [wtk])[-self.word_depth:]
                    current_word = []
                char_pos = 0

                # Space cost: char-CTW
                loss = self.char_ctw.log_loss(ch, char_ctx)
                losses.append(loss)
                positions.append(-1)

            elif char_pos == 0:
                # Word-initial: look ahead, get full word, pay full word cost here
                end = t
                while end < len(val_text) and val_text[end] not in (' ', '\n'):
                    end += 1
                actual_word = val_text[t:end]
                actual_tok  = actual_word if actual_word in self.word2idx else '<UNK>'

                word_loss = self.word_model.log_loss(
                    actual_tok, word_history[-self.word_depth:]
                )
                losses.append(word_loss)
                positions.append(0)

                current_word.append(ch)
                char_pos = 1

            else:
                # Within-word: free (word identity known from pos-0 prediction)
                losses.append(0.0)
                positions.append(char_pos)
                current_word.append(ch)
                char_pos += 1

        return losses, positions


def run_full_hierarchical(train_text, val_text, char_depth=5, word_depth=2,
                          vocab_size=5000, word_alpha=0.01):
    print(f"\n  Full-word Hierarchical CTW  char_D={char_depth}  word_D={word_depth}  "
          f"vocab={vocab_size}  word_alpha={word_alpha}")

    model = FullWordHierarchicalCTW(char_depth, word_depth, vocab_size, word_alpha)
    t0 = time.time()
    model.train(train_text)
    train_time = time.time() - t0

    print(f"  Evaluating on {len(val_text):,} chars...")
    t0 = time.time()
    losses, positions = model.eval(val_text)
    eval_time = time.time() - t0

    bpc = sum(losses) / len(losses)

    # Positional breakdown
    pos_buckets = defaultdict(list)
    for loss, pos in zip(losses, positions):
        key = str(pos) if pos < 6 else "6+"
        pos_buckets[key].append(loss)
    bpc_by_pos = {k: sum(v) / len(v) for k, v in pos_buckets.items()}

    print(f"  BPC={bpc:.4f}  train {train_time:.1f}s  eval {eval_time:.1f}s")
    print(f"  BPC by position: " +
          "  ".join(f"pos{k}={v:.3f}" for k, v in sorted(
              bpc_by_pos.items(),
              key=lambda x: int(x[0].replace('+', '')) if x[0] != '-1' else -1
          )))

    return {
        "variant":      f"FullWord char_D={char_depth} word_D={word_depth}",
        "type":         "full_hierarchical",
        "char_depth":   char_depth,
        "word_depth":   word_depth,
        "vocab_size":   vocab_size,
        "word_alpha":   word_alpha,
        "bpc":          bpc,
        "bpc_by_pos":   bpc_by_pos,
        "train_time_s": round(train_time, 1),
        "eval_time_s":  round(eval_time, 1),
    }


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def _plot(results, out_path, gap_analysis_path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    entries  = results["entries"]
    char_bpc = results.get("char_ctw_bpc")
    gpt2_bpc = results.get("gpt2_bpc")

    gap_data = {}
    if gap_analysis_path and os.path.exists(gap_analysis_path):
        with open(gap_analysis_path) as f:
            gap_data = json.load(f)

    # Identify key entries
    word_d1   = next((e for e in entries if e["type"] == "word" and e.get("depth") == 1), None)
    full_hier = next((e for e in reversed(entries) if e["type"] == "full_hierarchical"), None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # -----------------------------------------------------------------------
    # Panel 1: BPC comparison bar chart (key models only)
    # -----------------------------------------------------------------------
    ax = axes[0]

    bar_labels, bar_bpcs, bar_colors = [], [], []

    if char_bpc:
        bar_labels.append("Char-CTW D=5\n(baseline)")
        bar_bpcs.append(char_bpc)
        bar_colors.append("#E65100")

    if word_d1:
        bar_labels.append("Word-CTW D=1 *\n(best word model)")
        bar_bpcs.append(word_d1["bpc"])
        bar_colors.append("#1565C0")

    if full_hier:
        bar_labels.append("FullWord D=2\n(no marginalization)")
        bar_bpcs.append(full_hier["bpc"])
        bar_colors.append("#00796B")

    bars = ax.bar(range(len(bar_labels)), bar_bpcs,
                  color=bar_colors, alpha=0.87, width=0.5, zorder=3)
    for bar, bpc_val in zip(bars, bar_bpcs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bpc_val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    if gpt2_bpc:
        ax.axhline(gpt2_bpc, color="#B71C1C", linestyle="--", linewidth=2,
                   label=f"GPT-2 small  {gpt2_bpc:.3f}", zorder=4)
    ax.axhline(1.1, color="#795548", linestyle=":", linewidth=1.5, alpha=0.7,
               label="Shannon H(English) ≈ 1.1")
    ax.legend(fontsize=9, loc="upper right")

    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("Bits per character (BPC) ↓", fontsize=10)
    ax.set_title("Overall BPC: Token-Granularity Models", fontsize=11)
    ax.grid(True, alpha=0.25, axis="y", zorder=0)
    ax.set_ylim(0, max(bar_bpcs) * 1.2)
    ax.text(0.01, 0.02, "* spaces not included in Word-CTW BPC",
            transform=ax.transAxes, fontsize=7.5, color="#555", va="bottom")

    # -----------------------------------------------------------------------
    # Panel 2: BPC contribution by character type
    # contribution = fraction_of_chars × avg_BPC_at_that_position_type
    # Groups: Space | Word start (pos 0) | Within word (pos 1+)
    # -----------------------------------------------------------------------
    ax2 = axes[1]

    # Character type fractions  (avg_word_len ≈ 4.89 for WikiText-2)
    AVG_WL      = 4.89
    sp_frac     = 1.0 / (AVG_WL + 1)           # ≈ 0.170  (one space per word)
    pos0_frac   = 1.0 / (AVG_WL + 1)           # ≈ 0.170  (first char per word)
    within_frac = (AVG_WL - 1) / (AVG_WL + 1)  # ≈ 0.660

    models_pos = []

    ctw_pos_data  = gap_data.get("ctw_by_word_position", {})
    gpt2_pos_data = gap_data.get("gpt2_by_word_position", {})

    if ctw_pos_data and char_bpc:
        ctw_pos0_loss = ctw_pos_data.get("0", 0)
        ctw_space_c   = sp_frac * 0.995           # char-CTW space cost (≈0.995 from experiment)
        ctw_pos0_c    = pos0_frac * ctw_pos0_loss
        ctw_within_c  = char_bpc - ctw_space_c - ctw_pos0_c
        models_pos.append({"label": "Char-CTW D=5", "color": "#E65100",
                            "space": ctw_space_c, "pos0": ctw_pos0_c, "within": ctw_within_c})

    if full_hier and full_hier.get("bpc"):
        fw_bpc   = full_hier["bpc"]
        fw_space = sp_frac * 0.995               # same char-CTW for spaces
        fw_pos0  = fw_bpc - fw_space             # within-word = 0 (word known)
        models_pos.append({"label": "FullWord D=2", "color": "#00796B",
                            "space": fw_space, "pos0": fw_pos0, "within": 0.0})

    if gpt2_bpc and gpt2_pos_data:
        gpt2_within_keys = ["1", "2", "3", "4", "5", "6+"]
        gpt2_within_vals = [gpt2_pos_data[k] for k in gpt2_within_keys if k in gpt2_pos_data]
        gpt2_within_avg  = sum(gpt2_within_vals) / len(gpt2_within_vals) if gpt2_within_vals else 0
        gpt2_pos0_c   = pos0_frac * gpt2_pos_data.get("0", 0)
        gpt2_within_c = within_frac * gpt2_within_avg
        gpt2_space_c  = gpt2_bpc - gpt2_pos0_c - gpt2_within_c
        models_pos.append({"label": "GPT-2 small", "color": "#1565C0",
                            "space": gpt2_space_c, "pos0": gpt2_pos0_c, "within": gpt2_within_c})

    if models_pos:
        n = len(models_pos)
        x = np.arange(3)
        width = 0.22
        offsets = np.linspace(-(n-1)*width/2, (n-1)*width/2, n)
        group_labels = ["Space\n(pos −1)", "Word start\n(pos 0)", "Within word\n(pos 1+, avg)"]
        group_keys   = ["space", "pos0", "within"]

        for m, offset in zip(models_pos, offsets):
            vals = [m[k] for k in group_keys]
            bars2 = ax2.bar(x + offset, vals, width, color=m["color"],
                            alpha=0.87, label=m["label"], zorder=3)
            for bar, v in zip(bars2, vals):
                if v > 0.015:
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 0.01,
                             f"{v:.3f}", ha="center", va="bottom",
                             fontsize=7.5, color=m["color"], fontweight="bold")

        # Annotate the key FullWord insight
        fw_m = next((m for m in models_pos if "FullWord" in m["label"]), None)
        if fw_m:
            fw_offset = offsets[1]  # FullWord is second model
            ax2.annotate("within-word\n= FREE\n(word known)",
                         xy=(2 + fw_offset, 0.005),
                         xytext=(2 + fw_offset + 0.35, fw_m["pos0"] * 0.35),
                         ha="left", fontsize=7.5, color="#00796B",
                         arrowprops=dict(arrowstyle="->", color="#00796B", lw=1.0))

        ax2.set_xticks(x)
        ax2.set_xticklabels(group_labels, fontsize=10)
        ax2.set_ylabel("BPC contribution\n(char_type_fraction × avg_cost) ↓", fontsize=9)
        ax2.set_title("Where Is the Cost? BPC by Character Type", fontsize=11)
        ax2.legend(fontsize=9, loc="upper right")
        ax2.grid(True, alpha=0.25, axis="y", zorder=0)
        ax2.set_ylim(0, max(m["pos0"] for m in models_pos) * 1.25)

    plt.suptitle("Token-level CTW: Word-Granularity Experiments  "
                 "(10M train, WikiText-2, vocab=20K, word α=0.01)",
                 fontsize=11, y=1.01)
    plt.tight_layout()

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Also save to stable path for slides
    stable = figure_path("token_ctw_plot.png")
    fig.savefig(stable, dpi=150, bbox_inches="tight")
    print(f"Saved → {stable}")

    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_depths",  type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--vocab_size",   type=int, default=5000)
    parser.add_argument("--char_depth",   type=int, default=5,
                        help="Char-CTW depth for hierarchical variant")
    parser.add_argument("--word_depth_h", type=int, default=2,
                        help="Word-CTW depth for hierarchical variant")
    parser.add_argument("--no_bpe",       action="store_true")
    parser.add_argument("--no_hier",      action="store_true")
    parser.add_argument("--no_full_hier", action="store_true",
                        help="Skip full-word hierarchical variant")
    parser.add_argument("--word_alpha",   type=float, default=0.01,
                        help="KT alpha for word model in hierarchical variants (default 0.01)")
    parser.add_argument("--full",         action="store_true",
                        help="Use full WikiText-2 (2M train). Default: 500K.")
    parser.add_argument("--out",          default=None)
    parser.add_argument("--plot_only",    default=None, metavar="JSON",
                        help="Skip training; regenerate plot from existing results JSON.")
    args = parser.parse_args()

    if args.plot_only:
        with open(args.plot_only) as f:
            res = json.load(f)
        exp_dir  = os.path.dirname(__file__)
        out_png  = figure_for_result(args.plot_only)
        gap_path = result_path("gap_analysis_results.json")
        _plot(res, out_png, gap_analysis_path=gap_path)
        return

    exp_dir = os.path.dirname(__file__)
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = result_path(f"token_ctw_results_{ts}.json")

    # ---- Data ----
    print("Loading WikiText-2...")
    train_raw, val_raw = load_wikitext2()
    train_text = normalize_text(train_raw)
    val_text   = normalize_text(val_raw)

    max_train = 0 if args.full else 500_000
    max_val   = 0 if args.full else 150_000
    if max_train: train_text = train_text[:max_train]
    if max_val:   val_text   = val_text[:max_val]

    print(f"  Train: {len(train_text):,}  Val: {len(val_text):,}")

    # Char-CTW BPC for reference (from existing results if available)
    char_ctw_bpc, gpt2_bpc = None, None
    import glob
    for f in sorted(glob.glob(os.path.join(str(RESULTS_DIR), "full_results_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        ctw_r = [r for r in d.get("ctw_results", []) if r["depth"] == 5]
        if ctw_r:
            char_ctw_bpc = ctw_r[0]["bpc"]
        if d.get("gpt2_result"):
            gpt2_bpc = d["gpt2_result"]["bpc_per_char"]

    results = {
        "timestamp":    datetime.now().isoformat(),
        "train_chars":  len(train_text),
        "val_chars":    len(val_text),
        "char_ctw_bpc": char_ctw_bpc,
        "gpt2_bpc":     gpt2_bpc,
        "entries":      [],
    }

    def save():
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---- Variant 1: Word-level ----
    print(f"\n{'='*55}")
    print(f"Variant 1 — Word-level CTW  (vocab={args.vocab_size})")
    print(f"{'='*55}")
    for d in args.word_depths:
        print(f"\n  D = {d}")
        r = run_word_ctw(train_text, val_text, depth=d, vocab_size=args.vocab_size)
        results["entries"].append(r)
        save()

    # ---- Variant 2: BPE ----
    if not args.no_bpe:
        print(f"\n{'='*55}")
        print("Variant 2 — BPE CTW  (GPT-2 tokenizer, D=1)")
        print(f"{'='*55}")
        r = run_bpe_ctw(train_text, val_text, depth=1)
        if r:
            results["entries"].append(r)
            save()

    # ---- Variant 3: Hierarchical (marginalization, improved alpha) ----
    if not args.no_hier:
        print(f"\n{'='*55}")
        print(f"Variant 3 — Hierarchical CTW  "
              f"(char_D={args.char_depth}, word_D={args.word_depth_h}, "
              f"word_alpha={args.word_alpha})")
        print(f"{'='*55}")
        r = run_hierarchical(
            train_text, val_text,
            char_depth=args.char_depth,
            word_depth=args.word_depth_h,
            vocab_size=args.vocab_size,
            word_alpha=args.word_alpha,
        )
        results["entries"].append(r)
        save()

    # ---- Variant 4: Full-word Hierarchical ----
    if not args.no_full_hier:
        print(f"\n{'='*55}")
        print(f"Variant 4 — Full-word Hierarchical CTW  "
              f"(char_D={args.char_depth}, word_D={args.word_depth_h}, "
              f"word_alpha={args.word_alpha})")
        print(f"{'='*55}")
        r = run_full_hierarchical(
            train_text, val_text,
            char_depth=args.char_depth,
            word_depth=args.word_depth_h,
            vocab_size=args.vocab_size,
            word_alpha=args.word_alpha,
        )
        results["entries"].append(r)
        save()

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY  (train={len(train_text):,}  val={len(val_text):,}  "
          f"word_alpha={args.word_alpha})")
    print(f"{'='*60}")
    if char_ctw_bpc:
        print(f"  {'Char-CTW D=5 (baseline)':<40}  BPC = {char_ctw_bpc:.4f}")
    for e in results["entries"]:
        extra = ""
        if "word_alpha" in e:
            extra = f"  α={e['word_alpha']}"
        print(f"  {e['variant'] + extra:<40}  BPC = {e['bpc']:.4f}")
    if gpt2_bpc:
        print(f"  {'GPT-2 small (reference)':<40}  BPC = {gpt2_bpc:.4f}")

    gap_path = result_path("gap_analysis_results.json")
    _plot(results, figure_for_result(args.out), gap_analysis_path=gap_path)
    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
