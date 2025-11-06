## Worked Example: MI Estimators in Practice

This document walks through a concrete example comparing the **plug-in** and **listing** MI estimators using a simple multiple-choice question scenario.

---

## Problem Setup

- **Scenario**: Multiple-choice question with 4 possible answers {A, B, C, D}
- **Sampling**: 5 independent chains
- **Chain length**: 3 steps (iterative prompting with 3 rounds)
- **Goal**: Estimate MI to measure disagreement/uncertainty

### Observed Data (5 chains × 3 steps)

```
Chain 1: [A, B, A]
Chain 2: [A, B, A]
Chain 3: [A, C, B]
Chain 4: [B, B, A]
Chain 5: [A, C, A]
```

### Unique Tuples on Sample Support

After deduplication, we identify 4 unique tuples:

| Tuple | Count | Empirical Probability $\hat{\mu}(u)$ |
|-------|-------|------|
| (A, B, A) | 2 | 2/5 = 0.4 |
| (A, C, B) | 1 | 1/5 = 0.2 |
| (B, B, A) | 1 | 1/5 = 0.2 |
| (A, C, A) | 1 | 1/5 = 0.2 |

---

## Method 1: Plug-In Estimator

The plug-in formula computes MI as the difference between sum of marginal entropies and joint entropy:

$$I_{\text{plugin}} = \sum_{i=1}^n H(Y_i) - H(Y_1,\dots,Y_n)$$

### Step 1: Compute Marginal Entropies

**Position 1 ($Y_1$):**
- A appears: 4 times → P(A) = 0.8
- B appears: 1 time → P(B) = 0.2

$$H(Y_1) = -[0.8 \ln(0.8) + 0.2 \ln(0.2)] = -[0.8(-0.223) + 0.2(-1.609)] = 0.500 \text{ nats}$$

**Position 2 ($Y_2$):**
- B appears: 3 times → P(B) = 0.6
- C appears: 2 times → P(C) = 0.4

$$H(Y_2) = -[0.6 \ln(0.6) + 0.4 \ln(0.4)] = -[0.6(-0.511) + 0.4(-0.916)] = 0.673 \text{ nats}$$

**Position 3 ($Y_3$):**
- A appears: 4 times → P(A) = 0.8
- B appears: 1 time → P(B) = 0.2

$$H(Y_3) = 0.500 \text{ nats}$$

**Sum of marginals:**
$$\sum_{i=1}^3 H(Y_i) = 0.500 + 0.673 + 0.500 = 1.673 \text{ nats}$$

### Step 2: Compute Joint Entropy

$$H(Y_1, Y_2, Y_3) = -\sum_u \hat{\mu}(u) \ln \hat{\mu}(u)$$

$$= -[0.4 \ln(0.4) + 0.2 \ln(0.2) + 0.2 \ln(0.2) + 0.2 \ln(0.2)]$$

$$= -[0.4(-0.916) + 3(0.2)(-1.609)] = 1.332 \text{ nats}$$

### Step 3: Compute MI

$$I_{\text{plugin}} = 1.673 - 1.332 = \boxed{0.341 \text{ nats}}$$

---

## Method 2: Listing Estimator (Paper's Algorithm 1)

The listing estimator applies smoothing to both the joint and product-of-marginals distributions:

$$\widehat I_k(\gamma_1,\gamma_2) = \sum_{u\in S} \hat{\mu}(u) \log \frac{\hat{\mu}(u)+\gamma_1}{\hat{\mu}^{\otimes}(u)+\gamma_2}$$

### Step 1: Compute Marginals

Same as plug-in (from observed data):
- P(Y₁=A) = 0.8, P(Y₁=B) = 0.2
- P(Y₂=B) = 0.6, P(Y₂=C) = 0.4
- P(Y₃=A) = 0.8, P(Y₃=B) = 0.2

### Step 2: Compute Product-of-Marginals for Each Tuple

$$\hat{\mu}^{\otimes}(u) = P(Y_1) \times P(Y_2) \times P(Y_3)$$

| Tuple | $\hat{\mu}(u)$ | $\hat{\mu}^{\otimes}(u)$ |
|-------|---|---|
| (A, B, A) | 0.4 | (0.8)(0.6)(0.8) = 0.384 |
| (A, C, B) | 0.2 | (0.8)(0.4)(0.2) = 0.064 |
| (B, B, A) | 0.2 | (0.2)(0.6)(0.8) = 0.096 |
| (A, C, A) | 0.2 | (0.8)(0.4)(0.8) = 0.256 |

### Step 3: Apply Smoothing Parameters

With $\gamma_1 = \gamma_2 = 1/k = 1/5 = 0.2$:

| Tuple | $\hat{\mu}(u) + \gamma_1$ | $\hat{\mu}^{\otimes}(u) + \gamma_2$ | Ratio | $\hat{\mu}(u) \ln(\text{ratio})$ |
|-------|---|---|---|---|
| (A, B, A) | 0.4 + 0.2 = 0.6 | 0.384 + 0.2 = 0.584 | 1.027 | 0.4 × ln(1.027) = 0.011 |
| (A, C, B) | 0.2 + 0.2 = 0.4 | 0.064 + 0.2 = 0.264 | 1.515 | 0.2 × ln(1.515) = 0.083 |
| (B, B, A) | 0.2 + 0.2 = 0.4 | 0.096 + 0.2 = 0.296 | 1.351 | 0.2 × ln(1.351) = 0.060 |
| (A, C, A) | 0.2 + 0.2 = 0.4 | 0.256 + 0.2 = 0.456 | 0.877 | 0.2 × ln(0.877) = −0.026 |

### Step 4: Sum the Contributions

$$I_{\text{listing}} = 0.011 + 0.083 + 0.060 - 0.026 = \boxed{0.128 \text{ nats}}$$

---

## Comparison & Interpretation

| Metric | Plug-in | Listing | Difference |
|--------|---------|---------|-----------|
| **MI estimate** | 0.341 nats | 0.128 nats | −62.5% |
| **Calibration** | Optimistic | Conservative | — |
| **Smoothing** | None | $\gamma = 0.2$ | — |
| **Bias at k=5** | Higher | Lower | — |

### Why the Difference?

The **plug-in estimator overestimates** MI because:
1. It doesn't account for tuples that could exist but weren't observed
2. With only 5 samples from a large output space, many plausible tuples are missing
3. This creates an artificially sharp distribution → inflated MI

The **listing estimator is more conservative** because:
1. The smoothing parameters $\gamma = 0.2$ add a "virtual sample" of unobserved mass
2. This reduces extreme ratios (like observing rare tuples as if they're uniform)
3. The log-ratio becomes less dramatic → more realistic MI

### Asymptotic Behavior

As $k \to \infty$:
- $\gamma \to 0$ (smoothing disappears)
- Both estimators converge to the same value
- The true MI of the LLM's distribution is bracketed

---

## What the Original Paper Does

The paper (Section "A Computable Lower Bound" in `main_arxiv.tex`) uses **the listing estimator** (Algorithm 1 from `listing.tex`):

1. **Rejects plug-in**: Recognizes finite-sample bias from large output spaces
2. **Proposes listing**: Introduces smoothing with $\gamma = 1/k$ principled regularization
3. **Proves convergence**: Establishes concentration bounds using missing-mass theory
4. **Recommends this method**: Algorithm 1 becomes the standard for MI estimation in their framework

**Key paper insight:**
> "The listing estimator accounts for the probability mass of unobserved tuples, providing a more reliable finite-sample estimate when the support is large or effectively infinite."

For our example (k=5), the paper would report **MI ≈ 0.128 nats**, not 0.341 nats.

---

## Practical Implications

1. **For uncertainty quantification**: The listing estimate (0.128) is more trustworthy
   - Shows ~12.8% reduction from independence
   - More conservative confidence estimates

2. **For baselines**: If using plug-in (0.341), you overestimate agreement
   - May artificially boost confidence metrics
   - ECE could appear worse than it should

3. **For convergence**: Monitor both as k increases
   - Gap should shrink as $\gamma \to 0$
   - Convergence validates the MI framework

---

## References

- **Listing estimator origin**: Paper's Appendix, `listing.tex`
- **Missing-mass theory**: `missing_mass.tex`
- **Implementation**: `llm_belief_mi_test/mi_estimator.py`
  - `estimate_mi_listing_nats()` → listing method
  - `estimate_mi_nats()` → plug-in method (for comparison)

