# CTW Implementation Reference
## van Veen (2007) — Key Equations

All equation numbers refer to the thesis:
> van Veen, M. (2007). *Using Context-Tree Weighting as a Language Modeler in Dasher*.
> MSc thesis, TU Eindhoven.

---

## Chapter 3 — The CTW Algorithm

### 3.1 KT Estimator (binary, eq. 3.1 / 3.6)

The Krichevsky–Trofimov estimator for a memoryless binary source with `a` zeros and `b` ones:

```
P_e(0 | a, b)  =  (a + 1/2) / (a + b + 1)  =  (2a + 1) / (2a + 2b + 2)
P_e(1 | a, b)  =  (b + 1/2) / (a + b + 1)  =  (2b + 1) / (2a + 2b + 2)
```

Block probability update rule (eq. 3.8):
```
P_e(a+1, b)  =  P_e(a, b) * (2a + 1) / (2a + 2b + 2)   # after seeing a 0
P_e(a, b+1)  =  P_e(a, b) * (2b + 1) / (2a + 2b + 2)   # after seeing a 1
```

Initialized to:  P_e(0, 0) = 1 / 1  (eq. 4.4)

### 3.2 Weighted Probability (eq. 3.3 / 3.4)

```
# At LEAF (depth = D):
P_w^s  =  P_e(a_s, b_s)

# At INTERNAL NODE (depth < D):
P_w^s  =  (1/2) * P_e(a_s, b_s)  +  (1/2) * P_w^{0s} * P_w^{1s}
```

Conditional weighted probability (eq. 3.10):
```
P_w^s(0 | context)  =  [P_e(a+1, b) + P_w^{0s} * P_w^{1s} * P_w^{0s}(0|context)] / [P_e(a,b) + P_w^{0s}*P_w^{1s}]
```
(The denominator is the current P_w^s before the update.)

---

## Chapter 4 — Integer Arithmetic Implementation

### Key Idea

Instead of floating-point, each node stores two integer numerators:
- `Ne`  — numerator of  P_e(a, b),  initialized to **1**  (eq. 4.4: P_e(0,0) = 1)
- `Nw`  — numerator of  P_w^{0s} * P_w^{1s},  initialized to **1**  (eq. 4.10)

Both share a common denominator that is never stored explicitly.
Key invariant (eq. 4.21):  `De(a+1, b) == De(a, b+1) == Dw(x=0) == Dw(x=1)`.
This makes addition of P_e and P_w_children exact (no LCM needed).

### 4.1 Gamma Variables

Each node passes two integers `(γ0, γ1)` to its parent, where:
```
γ0 / (γ0 + γ1)  ≈  P_w^s(0 | context)   (eq. 4.22)
```

**At LEAF (depth = D, eq. 4.31–4.32):**
```
γ0  =  2*a + 1
γ1  =  2*b + 1
```
Only `a` and `b` are stored; `Ne` and `Nw` are not needed at leaves.

**At INTERNAL NODE (eq. 4.26–4.27):**
Receives `(g0_child, g1_child)` from the child on the context path. Then:
```
γ0  =  Ne * (2*a + 1) * (g0_child + g1_child)  +  Nw * g0_child * (2*a + 2*b + 2)
γ1  =  Ne * (2*b + 1) * (g0_child + g1_child)  +  Nw * g1_child * (2*a + 2*b + 2)
```
These are the PREDICTION gammas (proportion of P(0) and P(1)), computed before updating.

### 4.2 Update Rules

After computing γ0, γ1, update the stored values based on actual observed symbol `x`:

**For x = 0:**
```
Ne_new  =  Ne * (2*a + 1) * (g0_child + g1_child)    # eq. 4.15
Nw_new  =  Nw * g0_child  * (2*a + 2*b + 2)          # eq. 4.16
a       =  a + 1
```

**For x = 1:**
```
Ne_new  =  Ne * (2*b + 1) * (g0_child + g1_child)    # eq. 4.15 (x=1 variant)
Nw_new  =  Nw * g1_child  * (2*a + 2*b + 2)          # eq. 4.16 (x=1 variant)
b       =  b + 1
```

**Equivalently (unified form, x ∈ {0, 1}):**
```
count   =  a if x==0 else b
Ne_new  =  Ne * (2*count + 1) * (g0_child + g1_child)
Nw_new  =  Nw * g_x_child    * (2*a + 2*b + 2)      # g_x_child = g0 if x=0, g1 if x=1
```

### 4.3 Scaling (Section 4.2)

After every update, integers must stay in range `[1, 2^bits - 1]` to prevent overflow.

**Scaling gammas (eq. 4.33–4.34):**
```
while max(γ0, γ1) > 2^d - 1:
    γ0 //= 2
    γ1 //= 2
if min(γ0, γ1) == 0:
    min_val = 1   # set to 1 to avoid zero probability
```
Van Veen uses `d = 9` bits → range `[1, 511]`.

**Scaling Ne, Nw (eq. 4.35–4.36):**
```
while max(Ne, Nw) > 2^c - 1:
    Ne //= 2
    Nw //= 2
if min(Ne, Nw) == 0:
    min_val = 1
```
Van Veen uses `c = 9` bits → range `[1, 511]`.

**Scaling counts (Section 3.3 — "Limiting the counts"):**
```
if max(a, b) >= max_count:    # max_count = 255 (8 bits)
    a = (a + 1) // 2          # halve + round up
    b = (b + 1) // 2
```

### 4.4 Context Path Convention

For current position `t` with context `[x_{t-D}, ..., x_{t-2}, x_{t-1}]`:
- Root (depth 0) → child `x_{t-1}` → child `x_{t-2}` → … → leaf `x_{t-D}` (depth D)
- Most-recent symbol chooses which child of root to follow.
- In code: `context[-1]` at depth 1, `context[-2]` at depth 2, ..., `context[-D]` at leaf.

---

## Multinomial Extension (for text, Section 3.3)

For alphabet size |A| > 2, the KT estimator generalizes to:
```
P_e(a_φ + 1 | counts)  =  (a_φ + 1/2) / (total_count + |A|/2)
```
where `a_φ` is the count of symbol φ and `total_count = Σ a_φ`.

The weighted probability formula is the same but the product over children becomes a product over ALL |A| children:
```
# Internal node:
P_w^s  =  (1/2) * P_e(counts_s)  +  (1/2) * Π_{φ ∈ A} P_w^{φ·s}
```

In practice, CTW with large |A| is expensive. Options:
1. **Direct multinomial**: implement KT for arbitrary |A|  (simpler, some zero-frequency issues)
2. **Binary decomposition** (van Veen Ch. 4.3): 255 separate binary CTW trees per byte bit
   — avoids zero-frequency problem, better empirical compression

For this project, start with **direct multinomial** over ASCII lowercase (~27 symbols).

---

## Perplexity from CTW

CTW outputs `(γ0, γ1, ...)` proportional to `P_w(symbol | context)`.

```
P(symbol | context) = γ_symbol / sum(γ_all)
log2_prob           = log2(P(symbol | context))
bits_per_char (BPC) = -mean(log2_prob over all test symbols)
perplexity          = 2 ** BPC        # base-2 perplexity
```

Note: GPT-2 reports perplexity in natural log base. To compare:
```
BPC_nats = BPC * ln(2)
perplexity_nats = e ** BPC_nats
```

---

## Toy Setting A — Expected Results

Binary Markov chain of order k, run CTW with D ≥ k:
- CTW should achieve BPC close to the true entropy of the chain
- The MAP tree (CTM) should have exactly k levels

True entropy of order-1 chain with transition matrix P:
```
# Stationary distribution π: π P = π
# H = -Σ_s π_s * Σ_x P(x|s) * log2(P(x|s))
```
