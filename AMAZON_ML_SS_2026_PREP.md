# Amazon ML Summer School 2026 — Selection Test PREP

**Test:** Sunday 28 Jun 2026, 2:00 PM – 6:00 PM IST (attempt anytime in window, 60 min timer)
**Format:** 20 MCQ (Math + ML) + 2 Coding (problem-solving skills)

---

## OFFICIAL SYLLABUS MAP

| Official Pillar | Weight (est.) | Jump to |
|---|---|---|
| **Machine Learning fundamentals** | ~40% of MCQs (~8 Qs) | Part 2 §2.1–2.10 |
| **Probability & Statistics** | ~25% of MCQs (~5 Qs) | Part 1 §1.3 |
| **Linear Algebra** | ~20% of MCQs (~4 Qs) | Part 1 §1.1 |
| **Calculus / Optimization** | ~15% of MCQs (~3 Qs) | Part 1 §1.2 |
| **Part B: Coding (2 problems)** | — | Part 3 (DSA) + Part 4 (SQL) |

**Study order if short on time:** Part 2 → Part 1 §1.3 → Part 1 §1.1 → Part 6 → Part 3 → Part 4

---

## PART 1 — MATHEMATICS FOR ML (MCQ)

### 1.1 Linear Algebra

| Concept | Key Fact |
|---|---|
| Matrix multiply AB | A is (m×k), B is (k×n) → result (m×n). NOT commutative. |
| Transpose | (AB)ᵀ = BᵀAᵀ |
| Inverse | (AB)⁻¹ = B⁻¹A⁻¹. Exists iff det ≠ 0. |
| Rank | Max number of linearly independent rows/cols. rank(AB) ≤ min(rank(A), rank(B)) |
| Determinant | det(AB) = det(A)·det(B). det(Aᵀ) = det(A). |
| Eigenvalue | Av = λv. Characteristic polynomial: det(A - λI) = 0 |
| Trace | tr(A) = Σ eigenvalues. tr(AB) = tr(BA). |
| PSD matrix | All eigenvalues ≥ 0. xᵀAx ≥ 0 for all x. Covariance matrices are PSD. |
| SVD | A = UΣVᵀ. U,V orthogonal. σᵢ = singular values. rank = # nonzero σᵢ |
| PCA via SVD | Principal components = columns of V (or eigenvectors of AᵀA). |
| Orthogonal matrix | QᵀQ = I → Q⁻¹ = Qᵀ. Preserves norms and angles. |
| Norms | L1: Σ\|xᵢ\|. L2: √(ΣxᵢΣ). L∞: max\|xᵢ\|. Frobenius: √(Σσᵢ²) |

**Eigendecomposition:** A = QΛQ⁻¹ (Q = eigenvectors as columns, Λ = diagonal eigenvalues)

**PCA steps:**  
1. Center data (subtract mean)  
2. Compute covariance: Σ = (1/n)XᵀX  
3. Eigendecompose Σ → top-k eigenvectors  
4. Project: Z = XW  

---

### 1.2 Calculus & Optimization

| Concept | Key Fact |
|---|---|
| Gradient | Vector of partial derivatives. Points in direction of steepest ascent. |
| Chain rule | d/dx[f(g(x))] = f'(g(x))·g'(x). The backbone of backprop. |
| Jacobian | Matrix of all first-order partial derivatives. Shape: (m×n) for f:ℝⁿ→ℝᵐ |
| Hessian | Matrix of second-order partial derivatives. PSD → local min. NSD → local max. |
| Convex function | f(λx+(1-λ)y) ≤ λf(x)+(1-λ)f(y). Any local min = global min. |
| Gradient descent | θ ← θ - α·∇L(θ). Converges for convex with small enough α. |
| SGD vs GD | SGD uses 1 sample per step — noisy but faster. Mini-batch = best of both. |
| Learning rate | Too large → diverge. Too small → slow. |
| Momentum | v ← βv + ∇L; θ ← θ - αv. Reduces oscillation. β typically 0.9 |
| Adam | Adaptive moments. m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g². θ -= α·m̂/√v̂. β₁=0.9, β₂=0.999 |
| Convex vs non-convex | MSE + linear model = convex. Neural nets = non-convex. |
| Saddle points | Gradient = 0 but not min/max. Hessian has mixed eigenvalue signs. Common in deep learning. |
| Lagrangian | Constrained optimization: L(x,λ) = f(x) + λg(x). KKT conditions. |

**Sigmoid derivative:** σ'(x) = σ(x)(1 - σ(x))  
**Softmax derivative:** ∂sᵢ/∂zⱼ = sᵢ(δᵢⱼ - sⱼ)  
**ReLU derivative:** 1 if x > 0, else 0 (subgradient at 0)  

---

### 1.3 Probability & Statistics

| Concept | Key Fact |
|---|---|
| Bayes' theorem | P(A\|B) = P(B\|A)·P(A) / P(B) |
| Independence | P(A∩B) = P(A)·P(B). P(A\|B) = P(A). |
| Conditional | P(A\|B) = P(A∩B) / P(B) |
| Expectation | E[X] = Σ x·P(x). E[aX+b] = aE[X]+b |
| Variance | Var(X) = E[X²] - (E[X])². Var(aX) = a²Var(X) |
| Covariance | Cov(X,Y) = E[XY] - E[X]E[Y]. Zero cov ≠ independence (unless Gaussian). |
| Normal dist | N(μ,σ²). Bell curve. 68-95-99.7 rule. Sum of normals = normal. |
| Bernoulli | P(X=1)=p, P(X=0)=1-p. Mean=p, Var=p(1-p). |
| Binomial | n trials. P(X=k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ. Mean=np, Var=np(1-p). |
| Poisson | Events per interval. P(X=k) = λᵏe⁻ˡ/k!. Mean=Var=λ. |
| MLE | Maximize P(data\|θ). For Gaussian: μ_MLE = sample mean. |
| MAP | Maximize P(θ\|data) ∝ P(data\|θ)·P(θ). Adds prior. |
| Central Limit Theorem | Sample mean → Normal as n→∞ regardless of distribution. |
| Bias-Variance | MSE = Bias² + Variance + Irreducible noise. |
| Hypothesis testing | p-value: prob of seeing result if H₀ true. p < 0.05 → reject H₀. |

**MLE for Gaussian:** μ = (1/n)Σxᵢ, σ² = (1/n)Σ(xᵢ-μ)² (biased) or 1/(n-1) (unbiased)

---

### 1.4 Information Theory

| Concept | Formula | Meaning |
|---|---|---|
| Entropy | H(X) = -Σ p(x)log₂p(x) | Uncertainty in X. Max when uniform. |
| Cross-entropy | H(p,q) = -Σ p(x)log q(x) | Used as loss in classification. |
| KL Divergence | KL(p\|\|q) = Σ p(x)log(p(x)/q(x)) | Always ≥ 0. Not symmetric. |
| Mutual Information | I(X;Y) = H(X) - H(X\|Y) | Info shared between X and Y. |
| NLL Loss | -Σ yᵢlog(ŷᵢ) | Same as cross-entropy when p is one-hot. |

**Cross-entropy loss for binary classification:** L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]  
**Cross-entropy loss for multiclass:** L = -Σ yᵢ·log(ŷᵢ)

---

## PART 2 — MACHINE LEARNING (MCQ)

### 2.1 Linear & Logistic Regression

| Topic | Key Fact |
|---|---|
| Linear regression | ŷ = Xw + b. Loss: MSE = (1/n)Σ(yᵢ-ŷᵢ)². Normal equation: w = (XᵀX)⁻¹Xᵀy |
| Ridge (L2) | Loss + λΣwᵢ². Shrinks weights toward 0. Closed form: w = (XᵀX + λI)⁻¹Xᵀy |
| Lasso (L1) | Loss + λΣ\|wᵢ\|. Sparse solutions (exact zeros). Feature selection. |
| ElasticNet | L1 + L2. Balance between sparse and grouped. |
| R² score | 1 - SS_res/SS_tot. 1 = perfect, 0 = predicting mean. Can be negative. |
| Logistic regression | P(y=1) = σ(wᵀx + b). Decision boundary: wᵀx + b = 0. |
| Logistic loss | Binary cross-entropy. Convex. |
| Softmax | For multiclass: P(y=k) = exp(zₖ)/Σexp(zⱼ). |
| Multicollinearity | Correlated features → unstable w. Ridge helps. |

**Why L1 gives sparsity:** The L1 constraint creates "corners" in weight space at axes, where the loss contour first touches.

---

### 2.2 SVMs

| Topic | Key Fact |
|---|---|
| Margin | Distance between decision boundary and nearest points (support vectors). |
| Hard-margin SVM | Maximize 2/\|\|w\|\|. Only works when linearly separable. |
| Soft-margin SVM | Allows violations via slack ξᵢ ≥ 0. C = penalty for violations. Large C → less slack. |
| Kernel trick | Map to high-dim without computing features explicitly. k(x,x') = φ(x)ᵀφ(x'). |
| RBF kernel | k(x,x') = exp(-γ\|\|x-x'\|\|²). Most common. |
| Polynomial kernel | k(x,x') = (xᵀx' + c)ᵈ |
| Dual form | Only depends on dot products → kernel can replace them. |
| Support vectors | Only the points on/inside the margin affect the decision boundary. |
| SVM vs Logistic | SVM maximizes margin; LR maximizes likelihood. Both linear boundaries without kernel. |

---

### 2.3 Trees & Ensembles

| Topic | Key Fact |
|---|---|
| Gini impurity | 1 - Σpᵢ². Range [0, 0.5]. 0 = pure node. |
| Information gain | H(parent) - weighted_avg H(children). Split on max gain. |
| Overfitting in trees | Deep trees overfit. Pruning, max_depth, min_samples_leaf control this. |
| Bagging | Train N trees on bootstrapped subsets. Average predictions. Reduces variance. |
| Random Forest | Bagging + random feature subset at each split. Decorrelates trees. |
| Boosting | Sequential trees. Each tree corrects previous errors. Reduces bias. |
| AdaBoost | Reweight misclassified samples. Weak learners (shallow trees). |
| Gradient Boosting | Fit new tree to residuals. XGBoost adds L2 regularization on leaf weights. |
| XGBoost | Second-order Taylor expansion, regularization, column subsampling, parallel. |
| Feature importance | In RF/XGBoost: avg reduction in impurity or gain across all splits. |
| Out-of-bag error | ~37% samples not in each bootstrap. Use as validation without holdout set. |

---

### 2.4 Unsupervised Learning

| Topic | Key Fact |
|---|---|
| K-means | Minimize intra-cluster variance Σ\|\|xᵢ - μₖ\|\|². Sensitive to init & outliers. |
| K-means++ | Smart init: choose centers proportional to dist² from existing centers. |
| Elbow method | Plot inertia vs K. Pick "elbow" — where gain diminishes. |
| DBSCAN | Density-based. No need to specify K. Handles non-convex shapes. Labels outliers. |
| Hierarchical | Agglomerative (bottom-up) or divisive. Dendrogram shows merge hierarchy. |
| PCA | Maximize variance. First PC has most variance. Components are orthogonal. |
| Explained variance ratio | Sum of top-k eigenvalues / total. Pick k for 95% variance retained. |
| Autoencoder | Encoder → bottleneck → decoder. Unsupervised representation learning. |
| GMM | Soft K-means. Fits mixture of Gaussians. Uses EM algorithm. |

---

### 2.5 Model Evaluation

| Metric | Formula | Use when |
|---|---|---|
| Accuracy | TP+TN / Total | Balanced classes |
| Precision | TP / (TP+FP) | Cost of false positive is high |
| Recall (Sensitivity) | TP / (TP+FN) | Cost of false negative is high (cancer, fraud) |
| F1 | 2·(P·R)/(P+R) | Imbalanced classes |
| F-beta | (1+β²)·P·R / (β²·P + R) | β>1: recall matters more; β<1: precision |
| AUC-ROC | Area under ROC curve. 0.5 = random, 1.0 = perfect. | Ranking quality, threshold-invariant |
| PR-AUC | Better than ROC for highly imbalanced datasets | Rare positive class |
| MSE | Mean squared error. Penalizes outliers heavily. | Regression |
| MAE | Mean absolute error. Robust to outliers. | Regression |
| R² | Explained variance fraction. | Regression |
| Log loss | Cross-entropy. Penalizes confident wrong predictions. | Classification |

**K-fold cross-validation:** Split data into K folds. Train on K-1, validate on 1. Rotate. Average scores.  
**Stratified K-fold:** Preserves class distribution in each fold. Use for imbalanced data.  
**Train/Val/Test split:** Train to fit, Val for hyperparameters, Test for final unbiased estimate.  

---

### 2.6 Neural Networks & Deep Learning

| Topic | Key Fact |
|---|---|
| Perceptron | Linear classifier. Can't solve XOR. Multi-layer can. |
| Universal approximation | A single hidden layer with enough neurons can approximate any continuous function. |
| Backpropagation | Chain rule applied backwards through the computation graph. |
| Vanishing gradient | Gradients → 0 in deep nets with sigmoid/tanh. ReLU mitigates. |
| Exploding gradient | Gradients → ∞. Gradient clipping helps. Common in RNNs. |
| Weight init | Xavier/Glorot: σ = √(2/(fan_in+fan_out)). He init: σ = √(2/fan_in) for ReLU. |
| Batch Norm | Normalize layer inputs to N(0,1). Then scale/shift with γ,β. Reduces covariate shift. |
| Layer Norm | Normalize across features per sample. Used in Transformers. |
| Dropout | Randomly zero out neurons during training. Reduces overfitting. Not used at inference (scale by 1-p). |
| L2 regularization | = weight decay. Penalizes large weights. Equivalent to MAP with Gaussian prior. |
| L1 regularization | Promotes sparsity. Equivalent to MAP with Laplace prior. |
| Early stopping | Stop when validation loss starts increasing. Implicit regularization. |

**Activation functions:**

| Name | Formula | Use |
|---|---|---|
| Sigmoid | 1/(1+e⁻ˣ) | Output layer for binary classification |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | Range (-1,1). Saturates. |
| ReLU | max(0,x) | Hidden layers. Fast. Can die (outputs always 0). |
| Leaky ReLU | max(0.01x, x) | Fixes dying ReLU. |
| GELU | x·Φ(x) | Default in BERT/GPT. |
| Softmax | exp(zₖ)/Σexp(zⱼ) | Output layer for multiclass |
| Swish | x·σ(x) | Used in EfficientNet, modern models. |

---

### 2.7 Transformers & Attention (High-likelihood for Amazon!)

| Topic | Key Fact |
|---|---|
| Attention | Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V |
| Why √dₖ? | Prevents dot products from getting too large → softmax saturation. |
| Multi-head attention | Run attention h times in parallel with different projections. Captures different subspaces. |
| Self-attention | Q, K, V all come from same sequence. Each token attends to all others. |
| Cross-attention | Q from decoder, K/V from encoder. Used in encoder-decoder transformers. |
| Positional encoding | Sinusoidal or learned. Adds position info since attention is permutation-invariant. |
| BERT | Encoder-only. Masked Language Modeling + Next Sentence Prediction. Bidirectional. |
| GPT | Decoder-only. Causal (left-to-right) language modeling. Autoregressive. |
| Encoder-decoder | T5, BART. Good for seq2seq tasks (translation, summarization). |
| Complexity | Self-attention is O(n²d). Linear variants approximate this. |
| KV cache | Store K,V from previous tokens during generation. Avoids recomputation. |
| Transformer vs RNN | Transformer: parallel training, long-range dependencies. RNN: sequential, vanishing gradient. |

---

### 2.8 Overfitting, Regularization, Bias-Variance

```
High Bias (underfitting):                High Variance (overfitting):
- Training error HIGH                    - Training error LOW
- Validation error HIGH                  - Validation error HIGH (gap)
- Model too simple                       - Model too complex

Solutions for high bias:                 Solutions for high variance:
- More complex model                     - More training data
- More features                          - Regularization (L1, L2, dropout)
- Less regularization                    - Feature selection
                                         - Ensemble methods
                                         - Early stopping
```

**Bias-Variance decomposition:** Expected MSE = Bias² + Variance + Noise

---

### 2.9 Optimization Algorithms

| Algorithm | Update Rule | Key Property |
|---|---|---|
| SGD | θ -= α·g | Simple. Noisy. Needs tuning. |
| SGD+Momentum | v = βv + g; θ -= αv | Dampens oscillations. β~0.9 |
| Nesterov | Look-ahead gradient. | Faster convergence than momentum. |
| AdaGrad | θ -= α·g/√(G+ε). G accumulates g². | Adaptive LR. LR decreases over time. Good for sparse. |
| RMSProp | G = βG + (1-β)g². θ -= α·g/√(G+ε) | Fixes AdaGrad's diminishing LR. |
| Adam | m=β₁m+(1-β₁)g; v=β₂v+(1-β₂)g²; θ -= α·m̂/√(v̂+ε) | Bias-corrected. Default for deep learning. |
| AdamW | Adam + proper weight decay (decoupled). | Better generalization. Preferred in LLMs. |

---

### 2.10 CNNs

| Topic | Key Fact |
|---|---|
| Convolution | Filter slides over input. Detects local patterns. Parameter sharing. |
| Padding | 'same': output same size. 'valid': no padding, output shrinks. |
| Output size | (W - F + 2P) / S + 1. W=input, F=filter, P=padding, S=stride. |
| Pooling | Max pool: takes max in window. Reduces spatial dims. No params. |
| Receptive field | Region of input that affects a given output neuron. Grows with depth. |
| Depthwise separable | Depthwise conv + 1×1 conv. Much fewer params. Used in MobileNet. |
| ResNet | Skip connections: y = F(x) + x. Solves vanishing gradient in deep nets. |
| BatchNorm in CNNs | Normalize across N,H,W per channel. |

---

## PART 3 — DSA CODING PATTERNS

### 3.1 Must-Know Patterns

#### Two Pointers
```python
# Find pair summing to target in sorted array
left, right = 0, len(arr) - 1
while left < right:
    s = arr[left] + arr[right]
    if s == target: return (left, right)
    elif s < target: left += 1
    else: right -= 1
```

#### Sliding Window
```python
# Maximum sum subarray of size k
window_sum = sum(arr[:k])
max_sum = window_sum
for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i - k]
    max_sum = max(max_sum, window_sum)

# Variable window: longest substring without repeating chars
left = 0
seen = {}
max_len = 0
for right, c in enumerate(s):
    if c in seen and seen[c] >= left:
        left = seen[c] + 1
    seen[c] = right
    max_len = max(max_len, right - left + 1)
```

#### Binary Search
```python
# Standard binary search
lo, hi = 0, len(arr) - 1
while lo <= hi:
    mid = (lo + hi) // 2
    if arr[mid] == target: return mid
    elif arr[mid] < target: lo = mid + 1
    else: hi = mid - 1

# Lower bound (first index >= target)
lo, hi = 0, len(arr)
while lo < hi:
    mid = (lo + hi) // 2
    if arr[mid] < target: lo = mid + 1
    else: hi = mid
```

#### Fast & Slow Pointers (Floyd's cycle detection)
```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:  # cycle detected
        break
```

#### HashMap Pattern
```python
# Two Sum
seen = {}
for i, num in enumerate(nums):
    complement = target - num
    if complement in seen:
        return [seen[complement], i]
    seen[num] = i
```

#### Prefix Sum
```python
prefix = [0] * (n + 1)
for i, v in enumerate(nums):
    prefix[i+1] = prefix[i] + v
# Range sum [l, r] = prefix[r+1] - prefix[l]
```

---

### 3.2 Dynamic Programming

**Framework:** Define state → recurrence → base case → order of computation

```python
# Fibonacci (bottom-up)
dp = [0, 1]
for i in range(2, n+1):
    dp.append(dp[-1] + dp[-2])

# 0/1 Knapsack
dp = [[0]*(W+1) for _ in range(n+1)]
for i in range(1, n+1):
    for w in range(W+1):
        dp[i][w] = dp[i-1][w]
        if weights[i-1] <= w:
            dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])

# Longest Common Subsequence
dp = [[0]*(m+1) for _ in range(n+1)]
for i in range(1, n+1):
    for j in range(1, m+1):
        if s1[i-1] == s2[j-1]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

# Coin change (min coins)
dp = [float('inf')] * (amount + 1)
dp[0] = 0
for coin in coins:
    for x in range(coin, amount + 1):
        dp[x] = min(dp[x], dp[x - coin] + 1)
```

**Classic DP problems:**
- Longest Increasing Subsequence: O(n log n) with patience sorting
- Edit Distance: 2D DP on two strings
- Matrix Chain Multiplication: O(n³)
- Rod Cutting / Unbounded Knapsack
- Word Break: `dp[i] = any(dp[j] and s[j:i] in word_set)`

---

### 3.3 Graphs

```python
# BFS — shortest path in unweighted graph
from collections import deque
def bfs(graph, start, end):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        node, path = queue.popleft()
        if node == end: return path
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))

# DFS (iterative)
def dfs(graph, start):
    stack, visited = [start], set()
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node)
        for nb in graph[node]:
            stack.append(nb)

# Topological sort (Kahn's algorithm)
from collections import deque
def topo_sort(n, edges):
    graph = [[] for _ in range(n)]
    in_degree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    return order if len(order) == n else []  # empty if cycle
```

---

### 3.4 Heap / Priority Queue

```python
import heapq

# Min-heap (default)
heap = []
heapq.heappush(heap, val)
heapq.heappop(heap)  # returns smallest

# Max-heap: negate values
heapq.heappush(heap, -val)

# K largest elements
import heapq
heapq.nlargest(k, nums)  # or use min-heap of size k

# Merge K sorted lists
heap = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
heapq.heapify(heap)
result = []
while heap:
    val, i, j = heapq.heappop(heap)
    result.append(val)
    if j + 1 < len(lists[i]):
        heapq.heappush(heap, (lists[i][j+1], i, j+1))
```

---

### 3.5 String Problems

```python
# Anagram check
from collections import Counter
Counter(s) == Counter(t)

# KMP (pattern search) — O(n+m)
def kmp(text, pattern):
    # Build failure function
    lps = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j and pattern[i] != pattern[j]: j = lps[j-1]
        if pattern[i] == pattern[j]: j += 1
        lps[i] = j
    # Search
    j = 0
    for i, c in enumerate(text):
        while j and c != pattern[j]: j = lps[j-1]
        if c == pattern[j]: j += 1
        if j == len(pattern):
            return i - j + 1  # found at index
    return -1

# Palindrome check
s == s[::-1]
# Longest palindromic substring — expand around center
def expand(s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1; r += 1
    return s[l+1:r]
```

---

### 3.6 Tree Traversals

```python
# Inorder (left, root, right) → sorted order for BST
def inorder(node):
    if not node: return []
    return inorder(node.left) + [node.val] + inorder(node.right)

# Level-order BFS
from collections import deque
def level_order(root):
    if not root: return []
    queue, result = deque([root]), []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result

# BST operations: search/insert are O(h). Balanced BST: O(log n).
```

---

## PART 4 — SQL CHEAT SHEET

### 4.1 Query Structure (order of execution)

```
FROM → JOIN → WHERE → GROUP BY → HAVING → SELECT → DISTINCT → ORDER BY → LIMIT
```

### 4.2 Essential Queries

```sql
-- Basic aggregation
SELECT department, COUNT(*) as cnt, AVG(salary) as avg_sal
FROM employees
GROUP BY department
HAVING COUNT(*) > 5
ORDER BY avg_sal DESC;

-- INNER JOIN
SELECT e.name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.id;

-- LEFT JOIN (keep all left, NULL for non-matching right)
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id;

-- Find employees with no department
SELECT e.name FROM employees e
LEFT JOIN departments d ON e.dept_id = d.id
WHERE d.id IS NULL;

-- Subquery
SELECT name FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Correlated subquery: employees earning more than their dept avg
SELECT name, salary FROM employees e1
WHERE salary > (
    SELECT AVG(salary) FROM employees e2
    WHERE e1.dept_id = e2.dept_id
);
```

### 4.3 Window Functions (VERY likely to appear)

```sql
-- ROW_NUMBER, RANK, DENSE_RANK
SELECT name, salary, department,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) as rn,
    RANK()       OVER (PARTITION BY department ORDER BY salary DESC) as rnk,
    DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as drnk
FROM employees;

-- RANK vs DENSE_RANK:
-- Salaries: 100, 90, 90, 80
-- RANK:        1,  2,  2,  4  (gap after tie)
-- DENSE_RANK:  1,  2,  2,  3  (no gap)

-- Top N per group (e.g., top 3 salaries per department)
SELECT * FROM (
    SELECT *, DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dr
    FROM employees
) t WHERE dr <= 3;

-- Running total
SELECT name, salary,
    SUM(salary) OVER (PARTITION BY department ORDER BY hire_date) as running_total
FROM employees;

-- LAG / LEAD (access previous/next row)
SELECT name, salary,
    LAG(salary, 1, 0) OVER (ORDER BY hire_date) as prev_salary,
    LEAD(salary, 1, 0) OVER (ORDER BY hire_date) as next_salary
FROM employees;

-- NTILE (split into N buckets)
SELECT name, salary,
    NTILE(4) OVER (ORDER BY salary) as quartile
FROM employees;

-- Moving average
SELECT date, revenue,
    AVG(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as ma7
FROM sales;
```

### 4.4 CTEs (Common Table Expressions)

```sql
-- Basic CTE
WITH dept_avg AS (
    SELECT dept_id, AVG(salary) as avg_sal
    FROM employees
    GROUP BY dept_id
)
SELECT e.name, e.salary, d.avg_sal
FROM employees e
JOIN dept_avg d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;

-- Recursive CTE (org chart / hierarchy)
WITH RECURSIVE org AS (
    SELECT id, name, manager_id, 0 as level
    FROM employees WHERE manager_id IS NULL  -- root
    UNION ALL
    SELECT e.id, e.name, e.manager_id, o.level + 1
    FROM employees e
    JOIN org o ON e.manager_id = o.id
)
SELECT * FROM org ORDER BY level;
```

### 4.5 Classic SQL Interview Patterns

```sql
-- Nth highest salary (e.g., 2nd highest)
SELECT MAX(salary) FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
-- Or with DENSE_RANK:
SELECT salary FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) as dr FROM employees
) t WHERE dr = 2 LIMIT 1;

-- Duplicate detection
SELECT email, COUNT(*) FROM users GROUP BY email HAVING COUNT(*) > 1;

-- Delete duplicates (keep lowest id)
DELETE FROM users WHERE id NOT IN (
    SELECT MIN(id) FROM users GROUP BY email
);

-- Self-join (employees with same salary)
SELECT a.name, b.name, a.salary
FROM employees a JOIN employees b
ON a.salary = b.salary AND a.id < b.id;

-- Pivot (manual with CASE WHEN)
SELECT
    student_id,
    SUM(CASE WHEN subject = 'Math' THEN score END) as math,
    SUM(CASE WHEN subject = 'Science' THEN score END) as science
FROM grades
GROUP BY student_id;

-- Consecutive events (using LAG)
SELECT * FROM (
    SELECT id, date, status,
        LAG(status) OVER (ORDER BY date) as prev_status
    FROM events
) t WHERE status = 'fail' AND prev_status = 'fail';
```

### 4.6 Quick SQL Reference

```sql
-- String functions
UPPER(s), LOWER(s), LENGTH(s), TRIM(s)
SUBSTRING(s, start, len)  -- 1-indexed!
CONCAT(a, b), REPLACE(s, old, new)
LIKE '%pattern%'  (% = any, _ = one char)

-- Date functions
DATE_DIFF(end, start, unit)  -- varies by DB
YEAR(date), MONTH(date), DAY(date)
DATE_ADD(date, INTERVAL 7 DAY)
NOW(), CURRENT_DATE, CURRENT_TIMESTAMP

-- NULL handling
COALESCE(a, b, c)  -- first non-null
NULLIF(a, b)       -- NULL if a=b, else a
IS NULL, IS NOT NULL  -- never use = NULL

-- CASE WHEN
CASE
    WHEN score >= 90 THEN 'A'
    WHEN score >= 80 THEN 'B'
    ELSE 'C'
END AS grade
```

---

## PART 5 — TEST STRATEGY

### Time Allocation (60 min total)
```
MCQ Section (20 questions):     ~25 minutes  → avg 75 sec/question
Coding Q1 (DSA):                ~17 minutes
Coding Q2 (SQL):                ~13 minutes
Review buffer:                   ~5 minutes
```

### MCQ Strategy
1. **First pass (20 min):** Answer all questions you know confidently. Flag uncertain ones.
2. **Second pass (5 min):** Review flagged ones. Eliminate obviously wrong options.
3. **No negative marking:** Always guess — never leave blank.

### Likely MCQ Topics (Amazon's pattern from past editions)
- Gradient descent + learning rate behavior
- Bias-variance tradeoff (what increases which)
- Attention mechanism formula and complexity
- Precision vs Recall tradeoff
- Regularization effects (L1 vs L2)
- Eigenvalues and PCA
- Bayes theorem calculation
- Cross-entropy loss
- Confusion matrix calculations (TP, FP, FN, TN)
- K-means convergence
- Decision tree splitting criterion
- Backpropagation (chain rule application)
- Softmax and sigmoid properties
- Overfitting indicators and solutions

### Coding Strategy
1. **Read both problems first** (30 sec) — start with whichever you're more confident about.
2. **DSA:** Identify the pattern (DP? Graph? Sliding window?). Write brute force first if needed, then optimize.
3. **SQL:** Write the skeleton (SELECT...FROM...WHERE...GROUP BY), then add complexity.
4. **Edge cases:** empty input, single element, all same values, negative numbers.
5. **Partial credit matters:** Even a brute-force O(n²) solution gets partial marks.

### Common Pitfalls to Avoid
- Off-by-one errors in binary search
- Forgetting to handle NULL in SQL (use IS NULL, not = NULL)
- Infinite loops in BFS/DFS (mark visited!)
- Integer overflow (use long in Java/C++)
- SQL: `HAVING` vs `WHERE` — HAVING filters after GROUP BY, WHERE before

---

## PART 6 — QUICK RECALL CARDS

### Formulas you MUST know cold

```
Bayes:          P(A|B) = P(B|A)·P(A) / P(B)
Cross-entropy:  L = -Σ yᵢ·log(ŷᵢ)
Attention:      softmax(QKᵀ/√d)V
Gradient desc:  θ ← θ - α·∇L
Adam:           m = β₁m+(1-β₁)g,  v = β₂v+(1-β₂)g²,  θ -= α·m̂/√(v̂+ε)
Ridge (L2):     w* = (XᵀX + λI)⁻¹Xᵀy
Sigmoid:        σ(x) = 1/(1+e⁻ˣ),  σ'(x) = σ(x)(1-σ(x))
Softmax:        sᵢ = e^zᵢ / Σe^zⱼ
Entropy:        H(X) = -Σ p·log₂(p)
KL divergence:  KL(p||q) = Σ p·log(p/q) ≥ 0
PCA:            Eigenvectors of cov matrix Σ = (1/n)XᵀX
F1 score:       2PR/(P+R)
MSE:            (1/n)Σ(yᵢ-ŷᵢ)²
Gini impurity:  1 - Σpᵢ²
```

### Key inequalities / properties
- KL divergence ≥ 0 (equality iff p = q)
- Variance ≥ 0 always
- AUC = 0.5 means random classifier
- Entropy is max when distribution is uniform
- L1 norm ≥ L2 norm ≥ L∞ norm (for a given vector, generally)
- For positive definite matrix: all eigenvalues > 0

---

## PART 7 — NIGHT-BEFORE CHECKLIST

### Study Priority Order (if time is short)
1. **[30 min]** Scan MCQ sections 2.1–2.10 (ML fundamentals)
2. **[20 min]** Review Part 6 formulas — say them out loud
3. **[20 min]** Practice 2-3 SQL window function queries mentally
4. **[20 min]** Review DP patterns + 1-2 DSA patterns
5. **[10 min]** Scan math sections 1.1–1.4

### Day-of Logistics
- [ ] Test webcam at webcamtests.com BEFORE 1:00 PM
- [ ] Disable OBS / virtual camera software
- [ ] Download SmartHire app beforehand (don't wait until 2 PM)
- [ ] Close all notifications (Windows: Focus Assist → Alarms only)
- [ ] Plug in laptop + stable WiFi (or phone hotspot ready as backup)
- [ ] Keep water + snacks — no bathroom break mid-test
- [ ] Sit in a well-lit room alone (face detection violations)
- [ ] Have your Unstop login credentials ready
- [ ] Start the test by 2:30 PM at the latest (gives buffer before 6 PM window closes)

### Mental approach
- The test is designed to be passable with solid fundamentals — not tricks.
- For MCQs: if two options seem right, pick the more specific/quantitative one.
- For coding: working brute-force > elegant partial solution.
- Breathe. You built half this curriculum already. You know this stuff.

---

*Good luck, Nikhil. You've got this.* 🎯
