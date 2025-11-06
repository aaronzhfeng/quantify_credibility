## Mutual Information (MI) Estimators for Iterative Prompting

This note summarizes three MI estimators used in our implementation of the DeepMind paper “To Believe or Not to Believe Your LLM” (2024):

- The paper’s original objective and lower bound (information-theoretic view)
- The paper’s finite-sample “listing” estimator (Algorithm 1)
- The simple plug‑in estimator (baseline)

We also explain why we default to the listing estimator, and how it is theoretically equivalent to the original MI definition (and asymptotically reduces to the plug‑in form).

---

### 1) Original objective in the paper (information‑theoretic view)

The iterative prompting procedure induces an LLM‑derived pseudo joint distribution over a response chain $Y_1,\dots,Y_n$. The paper shows that a task‑relevant divergence is **lower‑bounded by the MI** of this joint, making MI a computable proxy for epistemic uncertainty:

**KL divergence inequality:**
$$D_{\mathrm{KL}}(\widetilde{\mathrm{LM}}, \widetilde{P}) \;\ge\; I(\widetilde{\mathrm{LM}})$$

**MI definition in terms of KL divergence:**
$$I(\widetilde{\mathrm{LM}}) = D_{\mathrm{KL}}\big(\widetilde{\mathrm{LM}},\, \widetilde{\mathrm{LM}}^{\otimes}\big)$$

In plain terms:
- $\widetilde{\mathrm{LM}}$ is the LLM's joint distribution over response chains
- $\widetilde{\mathrm{LM}}^{\otimes}$ is the product of the LLM's marginal distributions (independence assumption)
- MI measures how much the joint distribution diverges from independence
- Computing this exactly over all strings is impossible, so the paper proposes a finite‑sample estimator (Algorithm 1) defined on the **observed** tuples.

(See paper `main_arxiv.tex`, Sec. "A computable lower bound on epistemic uncertainty".)

---

### 2) Paper's finite‑sample estimator — "listing" (Algorithm 1)

Given $k$ sampled tuples (after optional deduplication/clustering to a set $S$), define:
- $\hat{\mu}$ = empirical probability mass on the sample support
- $\hat{\mu}^{\otimes}$ = product‑of‑marginals **computed on that same support**
- $\gamma_1, \gamma_2$ = stabilization parameters (set to $1/k$ in the paper)

**The listing MI estimator:**
$$\widehat I_k(\gamma_1,\gamma_2) = \sum_{u\in S} \hat{\mu}(u) \log \frac{\hat{\mu}(u)+\gamma_1}{\hat{\mu}^{\otimes}(u)+\gamma_2}$$

**In plain terms:** Sum over all unique observed tuples. For each tuple $u$, compute the log-ratio of (smoothed joint probability) to (smoothed product-of-marginals probability), weighted by the empirical joint probability.

This mirrors Algorithm 1 in the appendix. A verbatim excerpt of the core computation appears below (from `listing.tex`):

```79:111:doc/arXiv-2406.02543v2/listing.tex
def MI_estimator(sampled_tuples, mu_on_sample, gamma_1, gamma_2):
  """Implements MI estimator (Algorithm 1).

  Args:
    sampled_tuples: A numpy array of tuples sampled from the distribution after deduplication and clustering.
    mu_on_sample: A numpy array of probabilities of the clusters.
    gamma_1: stabilization parameter.
    gamma_2: stabilization parameter.

  Returns: (float) mutual information.
  """

  # Constructing empirical distribution (\hat{\mu})
  hat_mu_on_sample = mu_on_sample / mu_on_sample.sum()

  # Constructing empirical product distribution (\hat{\mu}^{\otimes})
  hat_mu_prod_on_sample = np.zeros((len(hat_mu_on_sample),))
  for (x_i, x) in enumerate(sampled_tuples):
    hat_mu_x_prod = 1
    for i in range(len(x)):
      marg_indices = [j for (j, z) in enumerate(sampled_tuples) if z[i] == x[i]]
      hat_mu_x_prod *= hat_mu_on_sample[marg_indices].sum()

    hat_mu_prod_on_sample[x_i] = hat_mu_x_prod

  # Computing MI estimate
  mi_estimate = hat_mu_on_sample * np.log((hat_mu_on_sample + gamma_1) / (hat_mu_prod_on_sample + gamma_2) )
  mi_estimate = mi_estimate.sum()
  return mi_estimate
```

Our implementation follows the same structure (sums over unique tuples, builds product‑of‑marginals on the sample support, and applies $\gamma$-smoothing):

```61:123:llm_belief_mi_test/mi_estimator.py
def estimate_mi_listing_nats(
    chains: List[List[str]],
    gamma1: float | None = None,
    gamma2: float | None = None,
) -> float:
    """Estimate MI using the paper's Algorithm 1 structure from listing.tex.

    Adapts the code to our LLM sampling setting:
    - We observe K samples of joint tuples (the chains), deduplicate to unique tuples,
      and compute empirical cluster weights (counts).
    - Compute product-of-marginals on the sampled support using empirical weights.
    - Stabilize with gamma1, gamma2 (defaults to 1/K) and sum hat_mu * log((hat_mu+g1)/(hat_mu_prod+g2)).
    """
```

**Why smoothing?** LLMs have effectively infinite output support. With small $k$ (e.g., 10), many plausible tuples go unobserved; naïvely assigning zero probability to them biases the joint entropy downward and MI upward. The $\gamma$-terms provide principled "pseudocounts" that control this finite‑sample bias and avoid $\log 0$ issues.

---

### 3) Plug‑in estimator (baseline)

The simplest finite‑sample MI estimator is the **plug‑in formula**:

$$I_{\text{plugin}} = \sum_{i=1}^n H(Y_i) - H(Y_1,\dots,Y_n)$$

where the entropy is defined as:
$$H(X) = - \sum_x \hat{p}(x) \log \hat{p}(x)$$

**In plain terms:** MI = (sum of individual entropies) − (joint entropy). Computed with empirical counts on the observed tuples.

Our reference implementation:

```23:46:llm_belief_mi_test/mi_estimator.py
def estimate_mi_nats(chains: List[List[str]]) -> float:
    """Estimate MI of (Y1,...,Yt) as Sum_i H(Yi) - H(Y1..Yt) from samples.

    This uses a simple plug-in estimator over discrete strings. While biased for small
    samples, it is effective for a basic reproduction.
    """
```

The plug‑in is intuitive but **systematically biased** in small‑$k$ regimes with large (or infinite) support because unseen mass is treated as zero probability.

---

### 4) Why we default to the listing estimator

- **Finite‑sample robustness**: Works on the **sampled support** and smooths both the joint and product‑of‑marginals with $\gamma$ (paper uses $\gamma_1=\gamma_2=1/k$).
- **Missing‑mass aware**: Accounts for the probability of unseen tuples, which is non‑negligible when responses are open‑vocabulary strings.
- **Theoretical backing**: The appendix establishes concentration via missing‑mass arguments and shows the estimator tracks the true MI as $k$ grows.
- **Numerical stability**: Avoids $\log 0$ and excessive MI inflation from sparsity.

---

### 5) Equivalence to the original MI in the limit (and "local" identity)

All three estimators target the **same functional**:
$$I(\mu) = D_{\mathrm{KL}}(\mu,\mu^{\otimes})$$

The only difference is **which distribution $\mu$** they use:

- **Original (theory)**: $\mu = \widetilde{\mathrm{LM}}$ = the **true** LLM joint distribution
- **Listing**: $\mu = \hat{\mu}_{\text{smoothed}}$ on the **sampled support**, with $\gamma\to 0$ as $k\to\infty$
- **Plug‑in**: $\mu = \hat{\mu}$ = unsmoothed empirical distribution

**The key insight:** In the **large‑$k$ limit** (as samples accumulate), the listing estimate converges to the plug‑in estimate, and both converge to the true MI defined in the paper. The listing variant is simply the **finite‑sample, stabilized realization** of the paper's MI computation.

**Local equivalence:** On the observed support (the tuples we actually see), all three methods compute the same functional. The difference only manifests in how we handle the unobserved region.

---

### 6) Pointers

- Theory & proof context: `doc/arXiv-2406.02543v2/main_arxiv.tex`, `missing_mass.tex`
- Paper’s estimator snippet: `doc/arXiv-2406.02543v2/listing.tex`
- Implementation:
  - `llm_belief_mi_test/mi_estimator.py` → `estimate_mi_listing_nats` and `estimate_mi_nats`


