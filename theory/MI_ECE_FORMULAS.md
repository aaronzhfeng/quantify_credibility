## MI and ECE Formulas for MCQ and Open-Ended Datasets

This document provides the mathematical definitions of **Mutual Information (MI)** and **Expected Calibration Error (ECE)** as applied in our LLM uncertainty quantification framework, covering both multiple-choice questions (MCQ) and open-ended question answering.

We explain:
- How MI quantifies epistemic uncertainty in both settings
- How ECE measures calibration quality
- The differences between MCQ and open-ended evaluation metrics
- Where each formula is implemented in the codebase

---

## 1. Mutual Information (MI) for Uncertainty Quantification

MI measures epistemic uncertainty by quantifying how much a language model "disagrees with itself" across independent samples. The core idea: **higher MI = more uncertainty**.

### 1.1 MI for MCQ Datasets

For multiple-choice questions, MI is computed on the discrete choice space.

#### Setup

- **Question**: A multiple-choice question with answer choices $\mathcal{A} = \{a_1, a_2, \ldots, a_m\}$
- **Sampling**: Generate $k$ independent chains (default $k=10$ from paper), each of length $n$ (default $n=2$)
- **Chain**: $(Y_1, Y_2, \ldots, Y_n)$ where each $Y_i \in \mathcal{A}$ is a sampled choice

**In plain terms:** We ask the LLM the same MCQ question $k$ times (e.g., 10 times), and for each attempt, we use iterative prompting with $n$ steps (e.g., 2 rounds of "reconsider your answer"). This gives us $k$ chains of $n$ responses each.

#### MI Definition (Plug-in Estimator)

The simplest MI estimator uses the entropy decomposition:

$$I_{\text{plugin}} = \sum_{i=1}^{n} H(Y_i) - H(Y_1, Y_2, \ldots, Y_n)$$

where:
- $H(Y_i) = -\sum_{a \in \mathcal{A}} \hat{p}_i(a) \log \hat{p}_i(a)$ is the marginal entropy at position $i$
- $H(Y_1, \ldots, Y_n) = -\sum_{u \in S} \hat{\mu}(u) \log \hat{\mu}(u)$ is the joint entropy
- $\hat{p}_i(a)$ is the empirical frequency of answer $a$ at position $i$ across the $k$ chains
- $\hat{\mu}(u)$ is the empirical frequency of tuple $u = (y_1, \ldots, y_n)$
- $S$ is the set of unique observed tuples

**In plain terms:** MI = (sum of individual entropies) − (joint entropy). If the model always gives the same answer, the joint entropy equals the sum of marginals, so MI = 0 (no uncertainty). If the model gives diverse answers, joint entropy is lower than the marginals would suggest, so MI > 0 (epistemic uncertainty).

Our implementation:

```23:46:llm_belief_mi_test/mi_estimator.py
def estimate_mi_nats(chains: List[List[str]]) -> float:
    """Estimate MI of (Y1,...,Yt) as Sum_i H(Yi) - H(Y1..Yt) from samples.

    This uses a simple plug-in estimator over discrete strings. While biased for small
    samples, it is effective for a basic reproduction.
    """
```

#### MI Definition (Listing Estimator)

The paper's recommended estimator (Algorithm 1) uses smoothing to handle finite-sample bias:

$$\widehat{I}_k(\gamma_1, \gamma_2) = \sum_{u \in S} \hat{\mu}(u) \log \frac{\hat{\mu}(u) + \gamma_1}{\hat{\mu}^{\otimes}(u) + \gamma_2}$$

where:
- $\hat{\mu}(u) = \frac{\text{count}(u)}{k}$ is the empirical joint probability
- $\hat{\mu}^{\otimes}(u) = \prod_{i=1}^{n} \hat{p}_i(u_i)$ is the product-of-marginals (independence baseline)
- $\gamma_1, \gamma_2$ are smoothing parameters (typically $\gamma_1 = \gamma_2 = 1/k$)

**In plain terms:** For each unique tuple we observed, compute the log-ratio of (smoothed joint frequency) to (smoothed independence-assumption frequency). The $\gamma$ terms prevent division by zero and account for the probability mass of unobserved tuples.

**Why smoothing?** With small $k$ and large answer spaces, many plausible responses go unobserved. The $\gamma$ terms provide principled "pseudocounts" that control finite-sample bias.

Our implementation:

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

(See `MI_ALGORITHMS.md` for a detailed derivation and comparison.)

#### Confidence Conversion

For calibration analysis, we convert MI to a confidence score using:

$$\text{confidence} = f(\text{MI}, p_{\text{maj}})$$

where common choices include:
- **Negative MI**: $c = -\text{MI}$ (lower MI = higher confidence)
- **Inverse exponential**: $c = e^{-\text{MI}}$ (normalizes to [0, 1] range)
- **Agreement-based**: $c = p_{\text{maj}}$ where $p_{\text{maj}}$ is the proportion of chains agreeing with the majority answer

The `confidence_method` parameter in evaluation functions controls this conversion.

---

### 1.2 MI for Open-Ended Datasets

For open-ended QA, MI computation differs because the output space is infinite (arbitrary text strings). We use two approaches depending on the dataset:

#### 1.2.1 Text-Based MI (e.g., SQuAD v2)

**Setup:**
- **Question**: An extractive QA question with context passage
- **Sampling**: Generate $k$ chains of length $n$, each producing a text answer $Y_i \in \mathcal{V}^*$ (unbounded vocabulary)
- **Chain**: $(Y_1, Y_2, \ldots, Y_n)$ where each $Y_i$ is a free-form text string

**MI Computation:**

Since the output space is continuous (infinite strings), we use the **listing estimator** on the observed sample support:

$$\widehat{I}_k(\gamma_1, \gamma_2) = \sum_{u \in S} \hat{\mu}(u) \log \frac{\hat{\mu}(u) + \gamma_1}{\hat{\mu}^{\otimes}(u) + \gamma_2}$$

where:
- $S = \{u_1, u_2, \ldots, u_{|S|}\}$ is the set of unique text tuples observed
- $u = (y_1^{\text{text}}, y_2^{\text{text}}, \ldots, y_n^{\text{text}})$ where each $y_i^{\text{text}}$ is a raw text string
- Marginals $\hat{p}_i(y)$ are computed on the empirical text distribution at position $i$

**In plain terms:** We compute MI directly on the raw text outputs. If the model generates "Paris", "paris", and "PARIS" across different chains, these count as three distinct outputs for MI purposes. High lexical diversity → high MI, even if semantically the answers are similar.

Implementation:

```1630:1638:llm_belief_mi_test/calibration.py
        # Compute MI for uncertainty estimation
        chains_text = [[text for text, _ in chain] for chain in chains_with_logprobs]
        
        if mi_method == "listing":
            mi_nats = estimate_mi_listing_nats(chains_text)
        else:
            mi_nats = estimate_mi_nats(chains_text)
        
        mi_bits = nats_to_bits(mi_nats)
```

**Agreement metric:** For open-ended QA, agreement is the fraction of chains that produce the same final answer (exact string match):

$$\text{agreement} = \frac{\text{\# chains with most common answer}}{k}$$

#### 1.2.2 Correctness-Based MI (e.g., TriviaQA)

For some open-ended datasets, we care about **correctness uncertainty** rather than lexical diversity.

**Setup:**
- For each chain output $Y_i$, compute binary correctness: $C_i = \mathbb{1}[\text{answer is correct}] \in \{0, 1\}$
- Transform chain from text to correctness labels: $(Y_1, \ldots, Y_n) \rightarrow (C_1, \ldots, C_n)$

**MI Computation:**

Compute MI on the **binary correctness sequence**:

$$\widehat{I}_k^{\text{correctness}}(\gamma_1, \gamma_2) = \sum_{u \in S_{\text{binary}}} \hat{\mu}_C(u) \log \frac{\hat{\mu}_C(u) + \gamma_1}{\hat{\mu}_C^{\otimes}(u) + \gamma_2}$$

where:
- $S_{\text{binary}} \subseteq \{0,1\}^n$ is the set of unique binary tuples (e.g., $(1,1)$, $(1,0)$, $(0,1)$, $(0,0)$ for $n=2$)
- $u = (c_1, c_2, \ldots, c_n)$ where each $c_i \in \{0, 1\}$
- $\hat{\mu}_C(u)$ is the empirical probability of correctness tuple $u$
- $\hat{\mu}_C^{\otimes}(u) = \prod_{i=1}^{n} \hat{p}_i^C(u_i)$ where $\hat{p}_i^C$ is the marginal correctness rate at position $i$

**In plain terms:** Instead of computing MI on the text diversity, we compute MI on the pattern of right/wrong answers. If the model consistently gets the answer right or consistently gets it wrong, MI is low. If some chains are correct and others incorrect, MI is high (epistemic uncertainty about correctness).

**Agreement metric:** For correctness-based MI, agreement is the fraction of chains that got the answer correct:

$$\text{agreement} = \frac{\text{\# chains with correct answer}}{k}$$

**Example:** If $k=10$ chains produce:
- 7 correct answers (various phrasings like "Paris", "paris", "The answer is Paris")
- 3 incorrect answers ("London", "Berlin", "Rome")

Then: agreement = 0.7, and MI reflects uncertainty about **correctness** (not text variety).

Implementation: See `evaluate_triviaqa_with_mi` in `calibration.py` (lines ~1800-2000).

---

## 2. Expected Calibration Error (ECE)

ECE measures **calibration quality**: how well a model's predicted confidence aligns with its actual accuracy. A well-calibrated model should be correct 80% of the time on examples where it expresses 80% confidence.

### 2.1 ECE Formula (Binning Approach)

Given $N$ questions, divide the confidence range $[0, 1]$ into $B$ bins (typically $B=10$ or $B=15$). For each bin $b$:

$$\text{ECE} = \sum_{b=1}^{B} \frac{|Q_b|}{N} \left| \text{correctness}(Q_b) - \text{conf}(Q_b) \right|$$

where:
- $B$ is the number of confidence bins
- $Q_b = \{i : c_i \in [b_{\text{lower}}, b_{\text{upper}})\}$ is the set of questions in bin $b$
- $|Q_b|$ is the number of questions in bin $b$
- $\text{correctness}(Q_b)$ is the average correctness in bin $b$ (defined below for MCQ vs. open-ended)
- $\text{conf}(Q_b) = \frac{1}{|Q_b|} \sum_{i \in Q_b} c_i$ is the average confidence in bin $b$

**In plain terms:** Group all questions by predicted confidence (e.g., 0-10%, 10-20%, ..., 90-100%). For each group, compute the gap between average confidence and actual correctness rate. ECE is the weighted average of these gaps.

**Interpretation:**
- ECE = 0 → perfect calibration (confidence matches accuracy)
- ECE = 1 → maximally miscalibrated
- Lower ECE is better

Our implementation:

```75:117:llm_belief_mi_test/calibration.py
def compute_ece(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy.
    
    Args:
        predictions: Binary array of predictions (0 or 1)
        confidences: Confidence scores (0 to 1)
        labels: Ground truth labels (0 or 1)
        n_bins: Number of calibration bins
        
    Returns:
        ECE value (0 to 1, lower is better)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        
        if i == n_bins - 1:  # Last bin includes right edge
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            # Compute accuracy in this bin
            bin_accuracy = (predictions[in_bin] == labels[in_bin]).mean()
            # Compute average confidence in this bin
            bin_confidence = confidences[in_bin].mean()
            # Weight by fraction of samples in bin
            bin_weight = n_in_bin / len(confidences)
            # Add to ECE
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece
```

---

### 2.2 ECE for MCQ Datasets

For MCQ, correctness is **binary accuracy**:

$$\text{correctness}(Q_b) = \text{acc}(Q_b) = \frac{1}{|Q_b|} \sum_{i \in Q_b} \mathbb{1}[\hat{y}_i = y_i^*]$$

where:
- $\hat{y}_i$ is the predicted answer choice
- $y_i^*$ is the ground truth choice
- $\mathbb{1}[\cdot]$ is the indicator function (1 if true, 0 if false)

**In plain terms:** For MCQ, a prediction is either exactly right or exactly wrong. If the model predicts "A" and the answer is "A", correctness = 1. Otherwise, correctness = 0.

---

### 2.3 ECE for Open-Ended Datasets

For open-ended QA, we use **Exact Match (EM)** or **F1 score** as the correctness metric instead of binary accuracy.

#### Using Exact Match (Strict)

$$\text{correctness}(Q_b) = \text{EM}(Q_b) = \frac{1}{|Q_b|} \sum_{i \in Q_b} \mathbb{1}[\text{EM}(\hat{y}_i, y_i^*)]$$

where:
- $\hat{y}_i$ is the predicted text answer (normalized: lowercased, stripped, articles removed)
- $y_i^*$ is the ground truth text (or set of acceptable answers)
- $\text{EM}(\hat{y}, y^*)$ = 1 if normalized strings match exactly, 0 otherwise

**In plain terms:** Exact Match is the strict criterion. "Paris" matches "paris" (case-insensitive), but "The city of Paris" does not match "Paris".

#### Using F1 Score (Lenient)

$$\text{correctness}(Q_b) = \text{F1}(Q_b) = \frac{1}{|Q_b|} \sum_{i \in Q_b} \text{F1}(\hat{y}_i, y_i^*)$$

where:
- $\text{F1}(\hat{y}, y^*)$ is the token-level F1 score (harmonic mean of precision and recall)
- F1 ranges from 0 to 1 (1 = perfect overlap, 0 = no overlap)

**In plain terms:** F1 gives partial credit for token overlap. "The city of Paris" vs. "Paris" would get F1 = 0.5 (1 token in common, but predicted 4 tokens). This is more forgiving than EM.

**Which to use?** In our implementation:
- **ECE uses EM** (binary) for consistency with MCQ evaluation
- **F1 is reported separately** as a secondary metric to understand partial correctness

---

## 3. Summary Comparison Table

| **Aspect** | **MCQ Datasets** | **Open-Ended Datasets** |
|------------|------------------|-------------------------|
| **MI Input** | Discrete choices $Y_i \in \mathcal{A}$ | Text strings $Y_i \in \mathcal{V}^*$ or binary correctness $C_i \in \{0,1\}$ |
| **MI Estimator** | Plug-in or Listing on choice space | Listing on text (SQuAD) or Listing on correctness (TriviaQA) |
| **MI Interpretation** | Uncertainty over which choice is correct | Uncertainty over text diversity (SQuAD) or correctness (TriviaQA) |
| **Agreement** | Fraction agreeing with majority choice | Fraction with most common text (SQuAD) or correct answer (TriviaQA) |
| **Confidence Conversion** | $f(\text{MI}, p_{\text{maj}})$ | $f(\text{MI}, \text{agreement})$ |
| **Correctness Metric** | Binary accuracy $\mathbb{1}[\hat{y} = y^*]$ | Exact Match $\mathbb{1}[\text{EM}(\hat{y}, y^*)]$ or F1 score |
| **ECE Formula** | $\sum_b \frac{\|Q_b\|}{N} \| \text{acc}(Q_b) - \text{conf}(Q_b) \|$ | $\sum_b \frac{\|Q_b\|}{N} \| \text{EM}(Q_b) - \text{conf}(Q_b) \|$ |

---

## 4. Implementation Details

### MI Calculation

1. **MCQ (OpenBookQA, TruthfulQA MC1/MC2):**
   - Extract choice letter (A/B/C/D) from each chain step
   - Compute MI on discrete choice tuples using `estimate_mi_listing_nats` or `estimate_mi_nats`
   - See `evaluate_mcq_with_mi` in `calibration.py` (lines ~300-500)

2. **Open-Ended Text-Based (SQuAD v2):**
   - Collect raw text outputs from each chain step
   - Compute MI on text strings using `estimate_mi_listing_nats(chains_text)`
   - Lexical diversity (even if semantically similar) increases MI
   - See `evaluate_extractive_qa_with_mi` in `calibration.py` (lines 1550-1700)

3. **Open-Ended Correctness-Based (TriviaQA):**
   - For each chain output, evaluate if correct ("correct") or incorrect ("incorrect")
   - Compute MI on binary correctness labels using `estimate_mi_listing_nats(correctness_chains)`
   - Focuses on disagreement about correctness, ignoring lexical diversity
   - See `evaluate_triviaqa_with_mi` in `calibration.py` (lines ~1800-2000)

### ECE Calculation

1. **Binning:**
   - Divide confidence range [0, 1] into $B$ equal bins (default $B=10$)
   - Each question assigned to a bin based on its predicted confidence

2. **Correctness Metric:**
   - MCQ: Use binary accuracy (exact match on choice letter)
   - Open-Ended: Use Exact Match (normalized string comparison) for ECE
   - F1 reported separately as supplementary metric

3. **Aggregation:**
   - For each bin, compute |avg_correctness - avg_confidence|
   - Weight by bin size and sum across bins

---

## 5. Key Differences: MCQ vs. Open-Ended

### Why MI Behaves Differently

**MCQ:**
- Finite, small answer space (typically 4-5 choices)
- MI captures disagreement over discrete options
- Low MI common (model often converges to one choice)

**Open-Ended (Text-Based):**
- Infinite answer space (any text string)
- MI captures lexical diversity, not just semantic disagreement
- High MI common (many ways to phrase same answer)

**Open-Ended (Correctness-Based):**
- Binary answer space (correct/incorrect)
- MI captures disagreement about correctness
- Similar behavior to MCQ but with only 2 "choices"

### Why ECE Uses Different Metrics

**MCQ:** Binary accuracy is natural (A is either right or wrong)

**Open-Ended:** Exact Match provides binary criterion analogous to MCQ accuracy, enabling comparable ECE computation across dataset types

---

## 6. References

- **Paper:** "To Believe or Not to Believe Your LLM" (DeepMind, 2024)
  - Theory: `doc/arXiv-2406.02543v2/main_arxiv.tex`
  - Algorithm 1: `doc/arXiv-2406.02543v2/listing.tex`
- **Implementation:**
  - MI estimators: `llm_belief_mi_test/mi_estimator.py`
  - Evaluation functions: `llm_belief_mi_test/calibration.py`
  - Dataset loaders: `llm_belief_mi_test/datasets.py`
- **Related docs:**
  - `MI_ALGORITHMS.md` (detailed MI estimator derivation)
  - `MI_ESTIMATOR_EXAMPLE.md` (worked example with numbers)
