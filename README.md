# 🤖 K-Nearest Neighbors (KNN) — Research & Experiments

> A research document and experimental walkthrough of the K-Nearest Neighbors algorithm combined with Cosine Similarity, conducted by **ye-PHD Lab**.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Feature Scaling](#feature-scaling)
- [Distance Metrics](#distance-metrics)
- [Choosing K](#choosing-k)
- [Mathematical Model: KNN + Cosine Similarity](#mathematical-model-knn--cosine-similarity)
- [KNN Regression](#knn-regression)
- [How KNN Works](#how-knn-works)
- [Experiments](#experiments)
- [Results](#results)

---

## Introduction

**K-Nearest Neighbors (KNN)** is a supervised learning, non-parametric algorithm. It operates on a simple principle: similar data points tend to cluster together in feature space. KNN leverages proximity between points to perform classification or value prediction.

---

## Feature Scaling

Since KNN relies entirely on distance calculations, any feature with a larger value range will dominate the others. **Scaling features before running KNN is mandatory.**

### Z-score Standardization

Transforms data to have zero mean and unit standard deviation. Particularly suited for normally distributed data.

$$z = \frac{x - \mu}{\sigma}$$

### Min-Max Normalization

Compresses data into a fixed range, typically $[0, 1]$, while preserving the relative relationships between original values. Sensitive to outliers; best used with small datasets.

$$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### Unit Length Scaling

Transforms each sample so the resulting vector has unit norm ($\|\mathbf{x}\| = 1$), eliminating magnitude effects and retaining only directional information. Commonly used alongside Cosine Similarity.

$$x_{\text{unit}} = \frac{x}{\sqrt{\sum_{i=1}^{n} x_i^2}}$$

---

## Distance Metrics

### Euclidean Distance

Measures the straight-line distance between two points in $n$-dimensional space — the most widely used metric, performing best on normalized data.

$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}$$

### Manhattan Distance

Sum of absolute coordinate differences — well-suited for grid-like spaces and less sensitive to outliers than Euclidean.

$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

### Chebyshev Distance

Determined solely by the dimension with the largest difference between two points (*chessboard distance*). Used in robot path planning and extreme anomaly detection.

$$d(x, y) = \max_{i=1}^{n}(|x_i - y_i|)$$

### Minkowski Distance (generalization)

$$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

| Parameter $p$ | Equivalent metric |
|:---:|---|
| $p = 1$ | Manhattan |
| $p = 2$ | Euclidean |
| $p \to \infty$ | Chebyshev |

---

## Choosing K

### With a Validation Set

| Dataset size $n$ | Strategy |
|---|---|
| $n < 30$ | Leave-One-Out Cross-Validation |
| $100 \leq n \leq 10{,}000$ | Random split — ~20% for validation |
| $n \geq 100{,}000$ | Random split — 1–5% |

A common starting point: $K = \sqrt{N}$

$$\text{Accuracy} = \frac{\text{Correct predictions}}{\text{Total validation samples}}$$

### Without a Validation Set

- Use $K = \sqrt{N}$; if the result is even, add or subtract 1 to avoid tie votes.
- Many libraries default to $K = 5$.
- For clean, well-separated data: $K = 1$ or $K = 3$ may suffice.

> ⚠️ **Note:** $K$ is a **hyperparameter** — it must be set by the user before the algorithm runs; KNN cannot determine it on its own.

---

## Mathematical Model: KNN + Cosine Similarity

### Z-score Standardization (per feature)

$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

After this step, each sample is represented as a vector $\mathbf{z} \in \mathbb{R}^d$, where $d$ is the number of features.

### Cosine Similarity

Let $\mathbf{z}_q \in \mathbb{R}^d$ be the query vector and $\mathbf{z}_i \in \mathbb{R}^d$ be the $i$-th training sample. Cosine Similarity measures the angle $\theta$ between them:

$$S_C(\mathbf{z}_q, \mathbf{z}_i) = \cos\theta = \frac{\mathbf{z}_q \cdot \mathbf{z}_i}{\|\mathbf{z}_q\|\,\|\mathbf{z}_i\|} = \frac{\sum_{j=1}^{d} z_{qj}\,z_{ij}}{\sqrt{\sum_{j=1}^{d} z_{qj}^2}\;\sqrt{\sum_{j=1}^{d} z_{ij}^2}}$$

| $S_C$ value | Meaning |
|:---:|---|
| $1$ | Vectors point in the same direction |
| $0$ | Vectors are orthogonal |
| $-1$ | Vectors point in opposite directions |

### Cosine Distance

$$D_C(\mathbf{z}_q, \mathbf{z}_i) = 1 - S_C(\mathbf{z}_q, \mathbf{z}_i), \quad D_C \in [0, 2]$$

### Classification — Majority Voting

$$\hat{y}_q = \underset{c \,\in\, \mathcal{C}}{\mathrm{argmax}} \sum_{\mathbf{z}_j \,\in\, \mathcal{N}_K(\mathbf{z}_q)} \mathbf{1}[y_j = c]$$

The query sample is assigned to the class that appears **most frequently** among its $K$ nearest neighbors.

---

## KNN Regression

Instead of voting, KNN Regression computes the **average** of the $K$ nearest neighbors' target values:

$$\hat{y} = \frac{1}{K} \sum_{i \in \mathcal{N}_K(x)} y_i$$

### Comparison with Linear Regression

| Criterion | KNN Regression | Linear Regression |
|---|---|---|
| Training time | $O(1)$ — no training phase | Requires parameter optimization |
| Prediction time | Slow — computes distances to all samples | Fast — a single linear operation |
| Linearity | ✅ Non-linear, flexible | ❌ Constrained to linear relationships |
| Extrapolation | ❌ Not supported | ✅ Supported |
| Best suited for | Complex data, small scale | Simple data, high-speed requirements |

---

## How KNN Works

```
1. Choose K
   ├── K too small → prone to overfitting
   └── K too large → prone to underfitting

2. Compute distances
   └── From the query point to every sample in the training set

3. Find K nearest neighbors
   └── Sort distances in ascending order, select the K smallest

4. Make a prediction
   ├── Classification → Majority Voting
   └── Regression    → Average of K neighbors
```

---

## Experiments

### Dataset

12 samples with two features: `SystemCalls` ($X_1$) and `NetworkConnections` ($X_2$).

| SystemCalls | NetworkConnections | Label |
|:-----------:|:-----------------:|:-----:|
| 1 | 1 | 0 |
| 1 | 2 | 0 |
| 2 | 1 | 0 |
| 2 | 2 | 0 |
| 8 | 8 | 1 |
| 8 | 9 | 1 |
| 9 | 8 | 1 |
| 9 | 9 | 1 |
| 5 | 5 | 1 |
| 4 | 7 | 0 |
| 2 | 8 | 0 |
| 7 | 2 | 1 |

**Statistics:**
- $X_1$: $\mu_1 = 4.8333$, $\sigma_1 = 3.0777$
- $X_2$: $\mu_2 = 5.1667$, $\sigma_2 = 3.1842$

**Hold-out split:**
- Train: 10 samples (samples 1, 2, 3, 5, 6, 7, 9, 10, 11, 12)
- Test: Sample 4 $(2, 2)$ — Label 0 and Sample 8 $(9, 9)$ — Label 1

---

### Experiment 1 — Euclidean Distance ($K = 3$)

**Sample 4** $(\mathbf{z}_4 = [-0.92,\ -0.99])$:

| Rank | Neighbor | Distance | Label |
|:----:|----------|:--------:|:-----:|
| 1 | Sample 3 | 0.31 | 0 |
| 2 | Sample 2 | 0.32 | 0 |
| 3 | Sample 1 | 0.45 | 0 |

→ Prediction: **Label 0** ✅

**Sample 8** $(\mathbf{z}_8 = [1.35,\ 1.20])$:

| Rank | Neighbor | Distance | Label |
|:----:|----------|:--------:|:-----:|
| 1 | Sample 7 | 0.31 | 1 |
| 2 | Sample 6 | 0.32 | 1 |
| 3 | Sample 5 | 0.45 | 1 |

→ Prediction: **Label 1** ✅

| $K$ | Correct | Total | Accuracy |
|:---:|:-------:|:-----:|:--------:|
| $K = 1$ | 2 | 2 | **100%** |
| $K = 3$ | 2 | 2 | **100%** |

---

### Experiment 2 — Cosine Similarity on new point Test$(4, 6)$

**Step 1 — Z-score normalization:**

$$\mathbf{z}_{\text{test}} = \left(\frac{4 - 4.8333}{3.0777},\ \frac{6 - 5.1667}{3.1842}\right) = (-0.2708,\ 0.2617)$$

$$\|\mathbf{z}_{\text{test}}\| = \sqrt{(-0.2708)^2 + (0.2617)^2} \approx 0.3766$$

**Step 2 — Compute Cosine Similarity:**

*Sample 10* — $\mathbf{z}_{10} = (-0.2708,\ 0.5758)$, Label 0:

$$S_C(\mathbf{z}_{\text{test}}, \mathbf{z}_{10}) = \frac{0.2240}{0.3766 \times 0.6362} \approx \mathbf{0.9349}$$

*Sample 8* — $\mathbf{z}_8 = (1.3538,\ 1.2039)$, Label 1:

$$S_C(\mathbf{z}_{\text{test}}, \mathbf{z}_8) = \frac{-0.0515}{0.3766 \times 1.8117} \approx \mathbf{-0.0755}$$

**Step 3 — Majority Voting:**

The neighbors with the highest $S_C$ values all belong to **Label 0** (Sample 10 achieves $S_C = 0.9349$).

→ Outcome: Test$(4, 6)$ is classified as **Label 0** ✅

---

## Results

When the dataset exhibits high separability — clusters are well-separated and far from the decision boundary — both Euclidean Distance and Cosine Similarity achieve 100% accuracy. Cosine Similarity proves especially effective when vector magnitude carries no meaningful information and only direction matters.

---

<div align="center">
  <sub>ye-PHD Lab &nbsp;·&nbsp; K-Nearest Neighbors Research</sub>
</div>
