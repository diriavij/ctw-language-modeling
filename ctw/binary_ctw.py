"""
Binary CTW — Context-Tree Weighting for binary alphabet {0, 1}.

Implements the integer-arithmetic algorithm from van Veen (2007), Chapter 4.
All probability quantities are represented as integer (γ0, γ1) pairs so that
    P_w(0 | context) = γ0 / (γ0 + γ1)
with no floating-point divisions during tree updates.

Usage:
    ctw = BinaryCTW(depth=5)
    for t, symbol in enumerate(sequence):
        context = sequence[max(0, t - ctw.D) : t]   # D most-recent symbols
        g0, g1 = ctw.update(symbol, context)
        p_symbol = g0 / (g0 + g1) if symbol == 0 else g1 / (g0 + g1)
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

@dataclass
class CTWNode:
    """
    One node in the context tree.

    Stored values (van Veen §4.1):
        a   — count of 0s seen after this suffix context
        b   — count of 1s seen after this suffix context
        Ne  — integer numerator of P_e(a, b)  [initialized to 1, eq. 4.4]
        Nw  — integer numerator of P_w^{0s} * P_w^{1s}  [initialized to 1, eq. 4.10]

    Ne and Nw share a common denominator that is never stored explicitly.
    The invariant De(a+1,b) == Dw(0,...) keeps them aligned (eq. 4.21).

    At LEAVES (depth == D) only a and b matter; Ne and Nw are unused.
    """
    a:  int = 0   # count of 0s
    b:  int = 0   # count of 1s
    Ne: int = 1   # numerator of P_e block probability
    Nw: int = 1   # numerator of P_w^{0s} * P_w^{1s}

    # Children indexed by symbol (0 or 1).
    # Created lazily when first encountered.
    children: dict[int, CTWNode] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BinaryCTW
# ---------------------------------------------------------------------------

class BinaryCTW:
    """
    Integer-arithmetic Context-Tree Weighting for binary sequences.

    Parameters
    ----------
    depth : int
        Maximum context depth D.  Good default: 5–8 for synthetic tests,
        up to 10 for natural language (van Veen §3.4, §4.2).
    prob_bits : int
        Number of bits used to represent Ne, Nw, and γ values (c = d in
        van Veen).  Van Veen recommends 9 bits (range [1, 511]).  Table 4.1
        shows compression is best at c = d = 9.
    max_count : int
        Upper bound on counts a, b before halving.  Van Veen uses 255 (8 bits).
    """

    def __init__(self, depth: int, prob_bits: int = 9, max_count: int = 255):
        self.D = depth
        self.MAX_VAL = (1 << prob_bits) - 1   # 2^prob_bits - 1 = 511 for 9 bits
        self.max_count = max_count
        self.root = CTWNode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, symbol: int, context: list[int]) -> tuple[int, int]:
        """
        Process one symbol and update the tree.

        Returns (γ0, γ1) from the ROOT, computed *before* the update, so:
            P_w(0 | context) ≈ γ0 / (γ0 + γ1)
            P_w(1 | context) ≈ γ1 / (γ0 + γ1)

        Parameters
        ----------
        symbol : int
            Observed symbol, 0 or 1.
        context : list[int]
            The D most-recent symbols, OLDEST first.
            context = [x_{t-D}, x_{t-D+1}, ..., x_{t-1}]
            If fewer than D symbols available, left-pad with 0.

        Returns
        -------
        (γ0, γ1) : tuple[int, int]
            Integer gammas proportional to (P(0|context), P(1|context)).

        Algorithm (§3.2.1):
            1. Walk root → leaf following context[-1], context[-2], ..., context[-D].
               Create missing nodes on the way.
            2. At leaf (depth D): compute γ from KT counts, update count.
            3. Walk back leaf → root: each internal node receives child γs,
               computes own γs (eq. 4.26–4.27), updates Ne / Nw (eq. 4.15–4.16).
        """
        context = self._pad_context(context)
        return self._recurse(self.root, symbol, context, depth=0)

    def predict(self, context: list[int]) -> tuple[float, float]:
        """
        Return (P(0|context), P(1|context)) WITHOUT updating the tree.

        Useful for evaluating held-out data after training.
        """
        context = self._pad_context(context)
        g0, g1 = self._predict_recurse(self.root, context, depth=0)
        total = g0 + g1
        return g0 / total, g1 / total

    # ------------------------------------------------------------------
    # Internal helpers — IMPLEMENT THESE
    # ------------------------------------------------------------------

    def _recurse(
        self, node: CTWNode, symbol: int, context: list[int], depth: int
    ) -> tuple[int, int]:
        """
        Recursive tree walk: compute gammas AND update the tree.

        Context path at each depth (§4.4 convention):
            depth 1 uses context[-1]  (= context[D-1], most recent)
            depth 2 uses context[-2]  (= context[D-2])
            ...
            depth D uses context[-D]  (= context[0],   oldest)

        Returns (γ0, γ1) for THIS node.
        """
        if depth == self.D:
            return self._update_leaf(node, symbol)

        # ---- Recurse to child on context path ----
        child_sym = context[-(depth + 1)]    # see context path convention above
        child = node.children.setdefault(child_sym, CTWNode())
        g0_child, g1_child = self._recurse(child, symbol, context, depth + 1)

        # ---- Update this internal node and return its gammas ----
        return self._update_internal(node, symbol, g0_child, g1_child)

    def _update_leaf(self, node: CTWNode, symbol: int) -> tuple[int, int]:
        """
        Process symbol at a leaf node (depth == D).

        Steps:
            1. Compute leaf gammas from KT counts (eq. 4.31–4.32):
                   γ0 = 2*a + 1
                   γ1 = 2*b + 1
            2. Scale γ0, γ1 to [1, MAX_VAL]  (§4.2, eq. 4.33–4.34).
            3. Increment the appropriate count (a for symbol=0, b for symbol=1).
            4. Halve counts if max_count reached (§3.3 "Limiting the counts").

        Returns gammas AFTER scaling (ratio is preserved); those are what the
        parent will use.

        Equations: 4.31, 4.32, 4.33, 4.34
        """
        gamma_0 = 2 * node.a + 1
        gamma_1 = 2 * node.b + 1

        gamma_0, gamma_1 = self._scale_pair(gamma_0, gamma_1)

        if symbol == 0:
            node.a += 1
        else:
            node.b += 1
        
        self._maybe_halve_counts(node)

        return gamma_0, gamma_1

    def _update_internal(
        self,
        node: CTWNode,
        symbol: int,
        g0_child: int,
        g1_child: int,
    ) -> tuple[int, int]:
        """
        Process symbol at an internal node (depth < D).

        Receives (g0_child, g1_child) from the child on the context path.

        Steps:
            1. Compute this node's gammas (eq. 4.26–4.27):

                   g_sum  = g0_child + g1_child
                   g_x    = g0_child if symbol==0 else g1_child

                   γ0  =  Ne * (2*a + 1) * g_sum  +  Nw * g0_child * (2*a + 2*b + 2)
                   γ1  =  Ne * (2*b + 1) * g_sum  +  Nw * g1_child * (2*a + 2*b + 2)

               These are the predictions for BOTH symbols; they do NOT depend
               on the actual observed symbol.

            2. Update stored Ne, Nw based on actual symbol (eq. 4.15–4.16):

                   count   = a if symbol==0 else b
                   Ne_new  = Ne * (2*count + 1) * g_sum
                   Nw_new  = Nw * g_x           * (2*a + 2*b + 2)

            3. Update count a or b (then halve if max_count reached).

            4. Scale Ne_new, Nw_new to [1, MAX_VAL]  (eq. 4.35–4.36).

            5. Scale γ0, γ1 to [1, MAX_VAL]  (eq. 4.33–4.34).

        Returns (γ0, γ1) for this node.

        Equations: 4.26, 4.27, 4.15, 4.16, 4.33–4.36
        """
        g_sum = g0_child + g1_child
        g_x   = g0_child if symbol == 0 else g1_child
        ab2   = 2 * node.a + 2 * node.b + 2   # computed from OLD counts

        # Step 1 — gammas (eq. 4.26–4.27); same regardless of actual symbol
        gamma_0 = node.Ne * (2 * node.a + 1) * g_sum + node.Nw * g0_child * ab2
        gamma_1 = node.Ne * (2 * node.b + 1) * g_sum + node.Nw * g1_child * ab2

        # Step 2 — new Ne, Nw depend on actual symbol (eq. 4.15–4.16)
        count  = node.a if symbol == 0 else node.b
        Ne_new = node.Ne * (2 * count + 1) * g_sum
        Nw_new = node.Nw * g_x             * ab2

        # Step 3 — update count (AFTER using old counts above)
        if symbol == 0:
            node.a += 1
        else:
            node.b += 1
        self._maybe_halve_counts(node)

        # Step 4 — scale Ne, Nw (eq. 4.35–4.36)
        node.Ne, node.Nw = self._scale_pair(Ne_new, Nw_new)

        # Step 5 — scale gammas (eq. 4.33–4.34)
        gamma_0, gamma_1 = self._scale_pair(gamma_0, gamma_1)

        return gamma_0, gamma_1


    # ------------------------------------------------------------------
    # Prediction without update (traverse read-only)
    # ------------------------------------------------------------------

    def _predict_recurse(
        self, node: CTWNode, context: list[int], depth: int
    ) -> tuple[int, int]:
        """
        Traverse the tree without updating anything.
        Returns (γ0, γ1) for this node based on current counts.

        Same structure as _recurse but:
        - Does NOT modify any node fields.
        - Returns the CURRENT prediction gammas.
        - If a child doesn't exist, treat as a fresh leaf (a=b=0 → γ0=γ1=1).
        """
        if depth == self.D:
            g0 = 2 * node.a + 1
            g1 = 2 * node.b + 1
            return self._scale_pair(g0, g1)

        child_sym = context[-(depth + 1)]
        child = node.children.get(child_sym)

        if child is None:
            g0_child, g1_child = 1, 1   # fresh node: a=b=0
        else:
            g0_child, g1_child = self._predict_recurse(child, context, depth + 1)

        ab2   = 2 * node.a + 2 * node.b + 2
        g_sum = g0_child + g1_child
        gamma_0 = node.Ne * (2 * node.a + 1) * g_sum + node.Nw * g0_child * ab2
        gamma_1 = node.Ne * (2 * node.b + 1) * g_sum + node.Nw * g1_child * ab2

        return self._scale_pair(gamma_0, gamma_1)

    # ------------------------------------------------------------------
    # Scaling utilities — implement these helpers
    # ------------------------------------------------------------------

    def _scale_pair(self, x: int, y: int) -> tuple[int, int]:
        """
        Scale integers (x, y) so that max(x, y) <= MAX_VAL.
        Divide BOTH by 2 repeatedly (preserves ratio).
        Ensure min(x, y) >= 1 (set to 1 if it becomes 0).

        Used for both γ pairs (eq. 4.33–4.34) and (Ne, Nw) pairs (eq. 4.35–4.36).
        """
        while max(x, y) > self.MAX_VAL:
            x //= 2
            y //= 2
        if x == 0:
            x = 1
        if y == 0:
            y = 1
        return (x, y)

    def _maybe_halve_counts(self, node: CTWNode) -> None:
        """
        If max(a, b) >= max_count, halve both counts (round up).
        This keeps the model adaptive over time (§3.3).
        """
        if max(node.a, node.b) >= self.max_count:
            node.a = (node.a + 1) // 2
            node.b = (node.b + 1) // 2

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _pad_context(self, context: list[int]) -> list[int]:
        """Left-pad context with 0s to length D if shorter."""
        if len(context) < self.D:
            context = [0] * (self.D - len(context)) + list(context)
        return list(context[-self.D:])   # take last D symbols

    def log_loss(self, symbol: int, context: list[int]) -> float:
        """
        Compute -log2(P(symbol | context)) WITHOUT updating the tree.
        Useful for evaluating test data after training.
        """
        import math
        p0, p1 = self.predict(context)
        p = p0 if symbol == 0 else p1
        return -math.log2(max(p, 1e-300))
