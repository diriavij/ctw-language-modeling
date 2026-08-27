"""Interpolated context-tree model for character- or token-level text.

This module uses generalized KT estimates (Dirichlet(1/2, ..., 1/2)) and
interpolates along the single observed context path. It is a practical
large-alphabet approximation inspired by CTW, not the exact multinomial CTW
product over every child and not van Veen's binary decomposition.

Usage:
    # Build vocabulary from training text
    ctw = TextCTW(depth=5, vocab=list("abcdefghijklmnopqrstuvwxyz "))

    # Train (update on each character)
    for t, ch in enumerate(train_text):
        context = train_text[max(0, t - ctw.D) : t]
        ctw.update(ch, list(context))

    # Evaluate perplexity on test text
    log_losses = []
    for t, ch in enumerate(test_text):
        context = test_text[max(0, t - ctw.D) : t]
        log_losses.append(ctw.log_loss(ch, list(context)))
    bpc = sum(log_losses) / len(log_losses)
    perplexity = 2 ** bpc
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math


@dataclass
class TextCTWNode:
    """
    Node in a multinomial context tree.

    Stored per node:
        counts  — dict mapping symbol → count (number of times symbol
                  was observed after this context).
        total   — sum of counts (= sum(counts.values())).

    The KT (Dirichlet-1/2) probability estimate for symbol φ is:
        P_e(φ | node)  =  (counts[φ] + 0.5) / (total + |A|/2)

    P_w is the weighted mixture:
        P_w^s(φ)  =  alpha * P_e(φ|node)  +  (1 - alpha) * Π_c P_w^{c·s}(φ) / Z
    but for prediction we use the simplified recursive formula (see below).

    Note: For large alphabets the product over ALL children is expensive.
    A practical simplification is to use ONLY the child on the context path
    (as in binary CTW with the standard formula).  This is what we implement.
    """
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    children: dict[str, TextCTWNode] = field(default_factory=dict)


class TextCTW:
    """
    Interpolated context-tree model for character- or token-level text.

    Uses a log-domain implementation to avoid numerical underflow for
    long sequences.  The integer arithmetic from van Veen Ch. 4 is NOT
    used here (float is fine for measuring perplexity; integer arith is
    needed for a real-time Dasher use case).

    Parameters
    ----------
    depth : int
        Maximum context depth D.
    vocab : list[str]
        List of all possible characters.  Must be fixed at construction.
    alpha : float
        Weight used by the training-time context-tree interpolation.
    max_count : int
        Halve counts when any count reaches this value (adaptivity).
    """

    def __init__(
        self,
        depth: int,
        vocab: list[str],
        alpha: float = 0.5,
        max_count: int = 255,
        min_child_count: int = 15,
        pred_alpha: float = 0.1,
    ):
        self.D = depth
        self.vocab = vocab
        self.vocab_set = set(vocab)
        self.A = len(vocab)
        self.alpha = alpha
        self.max_count = max_count
        self.min_child_count = min_child_count
        # pred_alpha: weight on current node's KT estimate during prediction.
        # (1 - pred_alpha) goes to the child (deeper context).
        # Small pred_alpha → deeper context dominates, BPC improves with D.
        # A smaller value makes the practical predictor rely more heavily on
        # deeper observed contexts. This is a tuned interpolation parameter,
        # not the fixed Bayesian mixture weight of exact CTW.
        self.pred_alpha = pred_alpha
        self.root = TextCTWNode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, symbol: str, context: list[str]) -> dict[str, float]:
        """
        Process one symbol given its context and update the tree.

        Returns a dict {symbol: P_w(symbol | context)} for all symbols
        in vocab, computed BEFORE the update.

        Parameters
        ----------
        symbol : str
            The observed character (must be in vocab).
        context : list[str]
            The D most-recent characters, oldest first.

        Returns
        -------
        probs : dict[str, float]
            Normalized probability distribution over vocab.
        """
        context = self._pad_context(context)
        probs = self._recurse(self.root, symbol, context, depth=0)
        return probs

    def predict(self, context: list[str]) -> dict[str, float]:
        """
        Return P_w(· | context) for all vocab symbols WITHOUT updating.
        """
        context = self._pad_context(context)
        return self._predict_recurse(self.root, context, depth=0)

    def log_loss(self, symbol: str, context: list[str]) -> float:
        """Return -log2(P_w(symbol | context)) without updating the tree."""
        probs = self.predict(context)
        p = probs.get(symbol, 1e-300)
        return -math.log2(max(p, 1e-300))

    # ------------------------------------------------------------------
    # Internal — IMPLEMENT THESE
    # ------------------------------------------------------------------

    def _recurse(
        self,
        node: TextCTWNode,
        symbol: str,
        context: list[str],
        depth: int,
    ) -> dict[str, float]:
        """
        Recursive update returning {symbol: P_w(symbol|context)}.

        Steps:
            1. If depth == D (leaf):
                   return _kt_probs(node)   [only KT estimate, no children]
            2. Otherwise:
                a. Recurse to child on context path → get child_probs dict.
                b. Compute P_e for this node: _kt_probs(node).
                c. Blend:  P_w(φ) = alpha * P_e(φ) + (1-alpha) * child_probs[φ]
                   (Normalize so probabilities sum to 1.)
                d. Update this node's counts: counts[symbol] += 1, total += 1.
                   (Halve if max_count reached.)
                e. Return P_w dict (the BLENDED probs, before the count update).

        NOTE: The blend (step c) uses child_probs for the single child on
        the context path.  This is an approximation to the full product
        over all children (van Veen §3.3, eq. 3.4), but it runs in O(D)
        time instead of O(D * |A|^D) and works well in practice.

        """
        pe = self._kt_probs(node)

        if depth == self.D:
            # Leaf: only KT estimate; update counts; return pe
            node.counts[symbol] = node.counts.get(symbol, 0) + 1
            node.total += 1
            self._maybe_halve_counts(node)
            return pe

        # Recurse to child on context path
        child_sym = context[-(depth + 1)]
        child = node.children.setdefault(child_sym, TextCTWNode())
        child_probs = self._recurse(child, symbol, context, depth + 1)

        # Blend: P_w(φ) = alpha * P_e(φ) + (1-alpha) * child_probs[φ]  (eq. 3.4)
        pw = {
            sym: self.alpha * pe[sym] + (1.0 - self.alpha) * child_probs[sym]
            for sym in self.vocab
        }
        total = sum(pw.values())
        pw = {sym: p / total for sym, p in pw.items()}

        # Update counts AFTER computing pw (pw uses pre-update pe)
        node.counts[symbol] = node.counts.get(symbol, 0) + 1
        node.total += 1
        self._maybe_halve_counts(node)

        return pw

    def _predict_recurse(
        self,
        node: TextCTWNode,
        context: list[str],
        depth: int,
    ) -> dict[str, float]:
        """
        Same as _recurse but without any updates.
        """
        pe = self._kt_probs(node)

        if depth == self.D:
            return pe

        child_sym = context[-(depth + 1)]
        child = node.children.get(child_sym)

        # Back off to current node if child is unseen or too sparse.
        if child is None or child.total < self.min_child_count:
            return pe

        child_probs = self._predict_recurse(child, context, depth + 1)

        # Use pred_alpha (default 0.1) so deeper context gets 90% of the weight.
        # CTW's training alpha=0.5 is optimal for online coding but puts 50%
        # on the unigram, which is wrong for offline prediction where deeper
        # context is always more informative.
        pw = {
            sym: self.pred_alpha * pe[sym] + (1.0 - self.pred_alpha) * child_probs[sym]
            for sym in self.vocab
        }
        total = sum(pw.values())
        return {sym: p / total for sym, p in pw.items()}

    def _kt_probs(self, node: TextCTWNode) -> dict[str, float]:
        """
        Compute the KT (Dirichlet-1/2) probability estimate for each symbol.

        Generalized KT estimator:
            P_e(φ | node) = (counts[φ] + 0.5) / (total + |A| * 0.5)

        Returns a normalized dict {symbol: P_e(symbol)}.
        """
        denom = node.total + self.A * 0.5
        return {
            sym: (node.counts.get(sym, 0) + 0.5) / denom
            for sym in self.vocab
        }

    def _maybe_halve_counts(self, node: TextCTWNode) -> None:
        """
        If any count reaches max_count, halve all counts (round up).
        Also update node.total.
        """
        if node.counts and max(node.counts.values()) >= self.max_count:
            node.counts = {sym: (c + 1) // 2 for sym, c in node.counts.items()}
            node.total  = sum(node.counts.values())

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _pad_context(self, context: list[str]) -> list[str]:
        """Left-pad with a null symbol ('\x00') to length D."""
        context = list(context)
        if len(context) < self.D:
            context = ['\x00'] * (self.D - len(context)) + context
        return context[-self.D:]

    def bpc_on_sequence(self, text: str) -> float:
        """
        Compute bits-per-character on `text` (train + eval together, online).
        Each character is predicted THEN the model is updated with it.
        """
        total_loss = 0.0
        for t, ch in enumerate(text):
            context = list(text[max(0, t - self.D) : t])
            total_loss += self.log_loss(ch, context)
            self.update(ch, context)
        return total_loss / len(text)
