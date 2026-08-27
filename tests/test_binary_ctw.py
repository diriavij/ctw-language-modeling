"""
Tests for BinaryCTW.

Run with:  python -m pytest tests/test_binary_ctw.py -v

Each test documents WHY it passes so you can check your implementation.
The tests are ordered from simplest to most complete.
"""

import math
import random
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ctw.binary_ctw import BinaryCTW, CTWNode


# ============================================================
# Test 1: Basic sanity — fresh model gives equal probabilities
# ============================================================

def test_uniform_prior():
    """
    A fresh CTW with no data should give P(0) = P(1) = 0.5.

    Reason: KT estimator with a=b=0 gives P_e(0) = 0.5, P_e(1) = 0.5.
    The weighted probability inherits this symmetry.

    γ0 = γ1 at every level (since a=b=0 everywhere).
    """
    ctw = BinaryCTW(depth=3)
    context = [0, 0, 0]
    g0, g1 = ctw.update(0, context)
    p0 = g0 / (g0 + g1)
    assert abs(p0 - 0.5) < 1e-9, f"Expected P(0)=0.5, got {p0}"


# ============================================================
# Test 2: Leaf update — KT counts change after one symbol
# ============================================================

def test_leaf_update_shifts_probability():
    """
    After seeing symbol 0 several times in the same context,
    P(0 | context) should be > 0.5.

    KT: P_e(0 | a, b) = (a + 0.5) / (a + b + 1)
    After seeing 5 zeros and 0 ones: P_e(0) = 5.5/6 ≈ 0.917.
    """
    ctw = BinaryCTW(depth=1)
    context = [0]

    # Feed 5 zeros
    for _ in range(5):
        ctw.update(0, context)

    g0, g1 = ctw.update(0, context)   # 6th prediction
    p0 = g0 / (g0 + g1)
    assert p0 > 0.8, f"Expected P(0) > 0.8 after 5 zeros, got {p0:.4f}"


# ============================================================
# Test 3: Context dependence — different contexts give different probs
# ============================================================

def test_context_dependence():
    """
    CTW should learn different statistics for different contexts.

    Train: after context [0], always 0.
           after context [1], always 1.
    Then predict: P(0 | [0]) >> P(0 | [1]).
    """
    ctw = BinaryCTW(depth=1)

    # Train: context [0] → always 0; context [1] → always 1
    for _ in range(20):
        ctw.update(0, [0])
        ctw.update(1, [1])

    g0_given_0, g1_given_0 = ctw.update(0, [0])
    g0_given_1, g1_given_1 = ctw.update(0, [1])

    p0_given_0 = g0_given_0 / (g0_given_0 + g1_given_0)
    p0_given_1 = g0_given_1 / (g0_given_1 + g1_given_1)

    assert p0_given_0 > 0.8, f"P(0|[0]) should be high, got {p0_given_0:.4f}"
    assert p0_given_1 < 0.2, f"P(0|[1]) should be low, got {p0_given_1:.4f}"


# ============================================================
# Test 4: Predict vs update — predict() should match update()'s gammas
# ============================================================

def test_predict_matches_update():
    """
    ctw.predict(context) and the gammas from ctw.update() should agree
    on the SAME state (before and after respectively, but on a fresh tree
    they should match before any update).

    More precisely: after training on a sequence, predict() on a new
    context should give the same probs as what update() would return.
    """
    ctw_train = BinaryCTW(depth=2)
    ctw_test  = BinaryCTW(depth=2)

    # Train both on the same sequence
    sequence = [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    for t in range(1, len(sequence)):
        ctx = sequence[max(0, t - 2): t]
        ctw_train.update(sequence[t], ctx)
        ctw_test.update(sequence[t], ctx)

    # Now compare predict vs update on a new symbol
    context = [0, 1]
    p0_pred, p1_pred = ctw_test.predict(context)

    # update() returns gammas BEFORE the update
    g0, g1 = ctw_train.update(0, context)
    p0_upd = g0 / (g0 + g1)

    assert abs(p0_pred - p0_upd) < 1e-6, (
        f"predict() and update() disagree: {p0_pred:.6f} vs {p0_upd:.6f}"
    )


# ============================================================
# Test 5: Toy Setting A — order-1 binary Markov chain
# ============================================================

def test_markov_order1_approaches_entropy():
    """
    Toy Setting A from the topic document (§4, Toy Setting A).

    Generate a long sequence from a binary Markov chain of order k=1.
    CTW with D >= 1 should achieve BPC close to the true entropy.

    Chain: P(0|0) = 0.9,  P(1|0) = 0.1
           P(0|1) = 0.2,  P(1|1) = 0.8

    True entropy ≈ 0.471 bits/symbol (computed analytically below).
    CTW should achieve BPC < H + 0.05 on 10,000 symbols.
    """
    from ctw.metrics import entropy_markov_order1

    p00 = 0.9   # P(0 | prev=0)
    p11 = 0.8   # P(1 | prev=1)
    H = entropy_markov_order1(p00=p00, p11=p11)
    print(f"\nTrue entropy: {H:.4f} bpc")

    # Generate sequence
    random.seed(42)
    N = 10_000
    seq = [0]
    for _ in range(N - 1):
        prev = seq[-1]
        if prev == 0:
            seq.append(0 if random.random() < p00 else 1)
        else:
            seq.append(1 if random.random() < p11 else 0)

    # Run CTW online (update and measure simultaneously)
    ctw = BinaryCTW(depth=3)
    total_bits = 0.0
    for t in range(1, N):
        ctx = seq[max(0, t - ctw.D) : t]
        g0, g1 = ctw.update(seq[t], ctx)
        p_sym = (g0 if seq[t] == 0 else g1) / (g0 + g1)
        total_bits += -math.log2(max(p_sym, 1e-300))

    bpc = total_bits / (N - 1)
    print(f"CTW BPC:       {bpc:.4f}")
    print(f"Overhead:      {bpc - H:.4f} bpc")

    assert bpc < H + 0.05, (
        f"CTW BPC {bpc:.4f} is more than 0.05 above entropy {H:.4f}"
    )


# ============================================================
# Test 6: Order-k recovery — deeper chain needs D >= k
# ============================================================

def test_markov_order2_needs_depth2():
    """
    A Markov chain of order 2 should be poorly compressed by D=1 CTW
    but well compressed by D=2 CTW.

    Chain: XOR of last 2 bits determines next bit (plus some noise).
    Order-2 source; order-1 CTW should be close to 1 bpc (like a coin),
    while order-2 CTW should achieve << 0.5 bpc.
    """
    random.seed(123)
    N = 5_000

    # Generate: P(1 | last two bits = 00 or 11) = 0.1; otherwise = 0.9
    seq = [0, 1]
    for _ in range(N - 2):
        xor = seq[-1] ^ seq[-2]
        # xor=0 means same, =1 means different
        p1 = 0.1 if xor == 0 else 0.9
        seq.append(1 if random.random() < p1 else 0)

    def run_ctw(depth, sequence):
        ctw = BinaryCTW(depth=depth)
        total = 0.0
        for t in range(2, len(sequence)):
            ctx = sequence[max(0, t - depth) : t]
            g0, g1 = ctw.update(sequence[t], ctx)
            p = (g0 if sequence[t] == 0 else g1) / (g0 + g1)
            total += -math.log2(max(p, 1e-300))
        return total / (len(sequence) - 2)

    bpc1 = run_ctw(depth=1, sequence=seq)
    bpc2 = run_ctw(depth=2, sequence=seq)

    print(f"\nDepth-1 CTW BPC: {bpc1:.4f}")
    print(f"Depth-2 CTW BPC: {bpc2:.4f}")

    assert bpc2 < bpc1 - 0.1, (
        f"D=2 CTW ({bpc2:.4f}) should be significantly better than D=1 ({bpc1:.4f})"
    )
    assert bpc2 < 0.5, f"D=2 CTW should compress well, got {bpc2:.4f}"


# ============================================================
# Test 7: Probability normalization
# ============================================================

def test_probabilities_sum_to_one():
    """
    P(0 | context) + P(1 | context) must equal 1.0 at all times.
    """
    random.seed(7)
    ctw = BinaryCTW(depth=4)
    sequence = [random.randint(0, 1) for _ in range(200)]

    for t in range(1, len(sequence)):
        ctx = sequence[max(0, t - ctw.D) : t]
        g0, g1 = ctw.update(sequence[t], ctx)
        total = g0 + g1
        p0 = g0 / total
        p1 = g1 / total
        assert abs(p0 + p1 - 1.0) < 1e-12, f"P(0)+P(1) = {p0+p1} ≠ 1 at t={t}"


# ============================================================
# Test 8: Scaling — probabilities never hit exactly 0 or 1
# ============================================================

def test_no_zero_probabilities():
    """
    After the scaling rules (eq. 4.33-4.36), γ0 and γ1 must always be >= 1.
    This ensures no symbol ever gets probability 0 or 1.
    """
    ctw = BinaryCTW(depth=5)
    # Feed a very biased sequence: all zeros
    N = 1000
    sequence = [0] * N

    for t in range(1, N):
        ctx = sequence[max(0, t - ctw.D) : t]
        g0, g1 = ctw.update(sequence[t], ctx)
        assert g0 >= 1 and g1 >= 1, (
            f"Zero gamma at t={t}: g0={g0}, g1={g1}"
        )
        assert g0 <= ctw.MAX_VAL and g1 <= ctw.MAX_VAL, (
            f"Gamma overflow at t={t}: g0={g0}, g1={g1}, max={ctw.MAX_VAL}"
        )


# ============================================================
# Test 9: iid source — BPC approaches binary entropy
# ============================================================

def test_iid_source_entropy():
    """
    For an i.i.d. source with P(1) = 0.3, the true entropy is H(0.3) ≈ 0.881 bpc.
    CTW with any D >= 0 should approach this (since no context helps for iid).
    """
    random.seed(0)
    N = 20_000
    p1 = 0.3
    seq = [1 if random.random() < p1 else 0 for _ in range(N)]

    ctw = BinaryCTW(depth=4)
    total = 0.0
    for t in range(1, N):
        ctx = seq[max(0, t - ctw.D) : t]
        g0, g1 = ctw.update(seq[t], ctx)
        p = (g0 if seq[t] == 0 else g1) / (g0 + g1)
        total += -math.log2(max(p, 1e-300))

    bpc = total / (N - 1)
    true_H = -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1))
    print(f"\nTrue H(0.3) = {true_H:.4f}, CTW BPC = {bpc:.4f}")

    assert bpc < true_H + 0.05, (
        f"CTW BPC {bpc:.4f} too far above true entropy {true_H:.4f}"
    )
