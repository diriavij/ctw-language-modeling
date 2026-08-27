"""
Metrics for evaluating CTW language models.

All functions accept raw probability values (NOT log-probs).
"""

import math
from typing import Iterable


def bits_per_char(log_probs_base2: Iterable[float]) -> float:
    """
    Compute bits per character (BPC) from a sequence of log2 probabilities.

    BPC = -mean(log2(P(x_i | context_i)))

    Parameters
    ----------
    log_probs_base2 : iterable of float
        log2(P(symbol | context)) for each test symbol.
        These are NEGATIVE numbers (log of a probability < 1).

    Returns
    -------
    float : BPC (positive number; lower is better)
    """
    values = list(log_probs_base2)
    if not values:
        raise ValueError("Empty sequence")
    return -sum(values) / len(values)


def perplexity(bpc: float) -> float:
    """
    Convert bits-per-character to perplexity (base-2 definition).

    perplexity = 2 ** BPC

    This is consistent with how van Veen measures compression (bpc).
    Note: GPT-2 / HuggingFace report perplexity using natural log (e-base).
    To convert: perplexity_nats = exp(BPC * ln(2)) = 2**BPC  (same value).
    """
    return 2.0 ** bpc


def entropy_markov_order1(p00: float, p11: float) -> float:
    """
    True entropy (BPC) of a binary first-order Markov chain.

    Parameters
    ----------
    p00 : float
        P(0 | previous=0)
    p11 : float
        P(1 | previous=1)

    Returns
    -------
    float : entropy in bits per symbol
    """
    p10 = 1.0 - p11   # P(0 | prev=1)
    p01 = 1.0 - p00   # P(1 | prev=0)

    # Stationary distribution: pi_0 * p01 = pi_1 * p10
    # pi_0 + pi_1 = 1  =>  pi_0 = p10 / (p01 + p10)
    pi0 = p10 / (p01 + p10)
    pi1 = 1.0 - pi0

    def h(p: float) -> float:
        """Binary entropy."""
        if p <= 0 or p >= 1:
            return 0.0
        return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    return pi0 * h(p01) + pi1 * h(p10)


def eval_binary_ctw(ctw, sequence: list[int]) -> dict:
    """
    Evaluate a trained BinaryCTW on a sequence without updating.

    Returns dict with 'bpc', 'perplexity', and per-symbol 'log_probs'.
    """
    log_probs = []
    for t in range(1, len(sequence)):
        context = sequence[max(0, t - ctw.D) : t]
        lp = -ctw.log_loss(sequence[t], context)   # log2(P)
        log_probs.append(lp)
    bpc = bits_per_char(log_probs)
    return {
        "bpc": bpc,
        "perplexity": perplexity(bpc),
        "log_probs": log_probs,
        "n_symbols": len(log_probs),
    }


def eval_text_ctw(ctw, text: str) -> dict:
    """
    Evaluate a trained TextCTW on a text string without updating.

    Returns dict with 'bpc', 'perplexity', 'n_chars'.
    """
    log_probs = []
    for t, ch in enumerate(text):
        context = list(text[max(0, t - ctw.D) : t])
        lp = -ctw.log_loss(ch, context)   # log2(P)
        log_probs.append(lp)
    bpc = bits_per_char(log_probs)
    return {
        "bpc": bpc,
        "perplexity": perplexity(bpc),
        "n_chars": len(text),
    }
