"""
Experiment 1 — Toy Setting A: Binary Markov Chain Validation

Generates a binary Markov chain of known order k, runs CTW with
various depths D, and plots BPC vs. true entropy.

This validates CTW correctness before running on natural language.

Usage:
    python experiments/validate_markov.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import random


def generate_markov_order1(n: int, p00: float, p11: float, seed: int = 42) -> list[int]:
    """Generate n symbols from a binary order-1 Markov chain."""
    random.seed(seed)
    seq = [0]
    for _ in range(n - 1):
        if seq[-1] == 0:
            seq.append(0 if random.random() < p00 else 1)
        else:
            seq.append(1 if random.random() < p11 else 0)
    return seq


def run_ctw_bpc(sequence: list[int], depth: int) -> float:
    """Run BinaryCTW online on a sequence and return BPC."""
    from ctw.binary_ctw import BinaryCTW
    ctw = BinaryCTW(depth=depth)
    total_bits = 0.0
    for t in range(1, len(sequence)):
        ctx = sequence[max(0, t - depth) : t]
        g0, g1 = ctw.update(sequence[t], ctx)
        p = (g0 if sequence[t] == 0 else g1) / (g0 + g1)
        total_bits += -math.log2(max(p, 1e-300))
    return total_bits / (len(sequence) - 1)


def true_entropy_order1(p00: float, p11: float) -> float:
    """
    True entropy H of a binary order-1 Markov chain.
    H = π_0 * H(p01) + π_1 * H(p10)
    where π is the stationary distribution.
    """
    p01 = 1.0 - p00
    p10 = 1.0 - p11
    pi0 = p10 / (p01 + p10)
    pi1 = 1.0 - pi0

    def h(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    return pi0 * h(p01) + pi1 * h(p10)


def main():
    # ----- Chain parameters -----
    p00, p11 = 0.9, 0.8
    N = 20_000
    depths = [1, 2, 3, 5, 8, 10]

    sequence = generate_markov_order1(N, p00=p00, p11=p11)
    H = true_entropy_order1(p00=p00, p11=p11)

    print(f"Binary Markov chain: P(0|0)={p00}, P(1|1)={p11}")
    print(f"True entropy:  {H:.4f} bpc")
    print(f"Sequence length: {N}")
    print()
    print(f"{'Depth D':>10}  {'CTW BPC':>10}  {'Overhead':>10}")
    print("-" * 35)

    for d in depths:
        bpc = run_ctw_bpc(sequence, depth=d)
        print(f"{d:>10}  {bpc:>10.4f}  {bpc - H:>+10.4f}")

    print()
    print("Expected: D=1 should be close to H; D>1 should be similar or slightly worse")
    print("(since chain is order-1, higher D adds model costs without benefit)")


if __name__ == "__main__":
    main()
