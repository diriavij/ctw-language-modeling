"""
Tests for TextCTW — multinomial character-level CTW.

Run with:  python -m pytest tests/test_text_ctw.py -v
"""

import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ctw.text_ctw import TextCTW


VOCAB_LOWER = list("abcdefghijklmnopqrstuvwxyz ")


def test_uniform_prior_text():
    """Fresh TextCTW gives equal probability to all symbols."""
    ctw = TextCTW(depth=3, vocab=VOCAB_LOWER)
    probs = ctw.predict(['\x00'] * 3)
    expected = 1.0 / len(VOCAB_LOWER)
    for sym, p in probs.items():
        assert abs(p - expected) < 1e-6, (
            f"P({sym!r}) = {p:.4f}, expected {expected:.4f}"
        )


def test_probabilities_sum_to_one_text():
    """P_w must sum to 1 at all times."""
    ctw = TextCTW(depth=3, vocab=VOCAB_LOWER)
    text = "the quick brown fox"
    for t, ch in enumerate(text):
        ctx = list(text[max(0, t - ctw.D): t])
        probs = ctw.update(ch, ctx)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-9, f"Probs sum to {total} at t={t}"


def test_text_ctw_learns_repetition():
    """
    After seeing 'aaa...a' many times, P('a' | context='aa') should be high.
    """
    ctw = TextCTW(depth=2, vocab=VOCAB_LOWER)
    train = "a" * 200
    for t in range(1, len(train)):
        ctx = list(train[max(0, t - ctw.D) : t])
        ctw.update(train[t], ctx)

    probs = ctw.predict(['a', 'a'])
    assert probs.get('a', 0) > 0.9, (
        f"Expected P('a'|'aa') > 0.9, got {probs.get('a', 0):.4f}"
    )


def test_bpc_decreases_with_more_data():
    """
    BPC on repeated text should decrease as the model sees more of it.
    Measure BPC on the first 100 chars vs last 100 chars of a 500-char text.
    """
    ctw = TextCTW(depth=5, vocab=VOCAB_LOWER)
    text = ("the cat sat on the mat " * 25)[:500]

    bits_first = []
    bits_last = []

    for t, ch in enumerate(text):
        ctx = list(text[max(0, t - ctw.D): t])
        probs = ctw.predict(ctx)
        lp = -math.log2(max(probs.get(ch, 1e-300), 1e-300))
        if t < 100:
            bits_first.append(lp)
        elif t >= 400:
            bits_last.append(lp)
        ctw.update(ch, ctx)

    bpc_first = sum(bits_first) / len(bits_first)
    bpc_last  = sum(bits_last)  / len(bits_last)
    print(f"\nBPC first 100 chars: {bpc_first:.3f}")
    print(f"BPC last  100 chars: {bpc_last:.3f}")

    assert bpc_last < bpc_first, (
        f"Model did not improve: BPC went from {bpc_first:.3f} to {bpc_last:.3f}"
    )


def test_bpc_reasonable_on_english():
    """
    On a short English text, BPC should be between 3 and 5 bpc
    (character-level; better models reach ~1.5 bpc on large data).
    """
    ctw = TextCTW(depth=6, vocab=VOCAB_LOWER)
    text = (
        "it was the best of times it was the worst of times "
        "it was the age of wisdom it was the age of foolishness "
        "it was the epoch of belief it was the epoch of incredulity "
    ).lower()
    # Keep only vocab chars
    text = "".join(c for c in text if c in ctw.vocab_set)

    total_bits = 0.0
    for t, ch in enumerate(text):
        ctx = list(text[max(0, t - ctw.D): t])
        total_bits += ctw.log_loss(ch, ctx)
        ctw.update(ch, ctx)

    bpc = total_bits / len(text)
    print(f"\nEnglish text BPC: {bpc:.3f}")
    assert 2.0 < bpc < 5.5, f"BPC {bpc:.3f} outside expected range [2, 5.5]"
