# Technical Knowledge Guide — What You Need to Know

> The tech stack, tools, libraries, and concepts behind the three ML problem statements. Use this to
> level up *before* you defend, run, or judge them. Companion to
> [ml_problem_statements.md](ml_problem_statements.md) and
> [interview_pitch_guide.md](interview_pitch_guide.md).
>
> Each section marks depth: **🟢 must-know** (you'll be asked), **🟡 should-know** (shows depth),
> **🔵 nice-to-know** (bonus points).

---

## 0. Shared foundations (applies to all three)

### Languages & runtime
- **Python 3.10+** — the lingua franca for all three. Know `venv`/`conda`, `pip install -r
  requirements.txt`, and how to pin versions for reproducibility. 🟢
- **Git & GitHub** — public repo, `README.md` with a one-command run, fixed random seed. 🟢

### Concepts you must hold in your head
- **Train / validation / test split** — and *why* the split strategy matters (random vs. temporal vs.
  hidden). This is the single most-tested idea across all three. 🟢
- **Data leakage** — any signal in training that won't exist at inference time. The silent killer of
  ML projects. 🟢
- **Metric vs. target (Goodhart's Law)** — optimising a proxy too hard breaks it. The spine of
  problem 3, but it shows up everywhere. 🟡
- **Reproducibility** — fixed seeds, pinned deps, deterministic runs. Judges reward it; interviewers
  expect it. 🟢

### Tooling baseline
- `numpy`, `pandas` — array + dataframe manipulation. 🟢
- `matplotlib` — plots (reliability diagrams, ROC curves). 🟢
- Jupyter / VS Code notebooks — for EDA and demos. 🟡

---

## 1. Problem 1 stack — Tabular ML 🟢

### Libraries
| Library | Use | Depth |
|---------|-----|-------|
| `pandas` | load CSV, feature engineering, datetime math | 🟢 |
| `numpy` | numeric ops | 🟢 |
| `scikit-learn` | models, metrics, calibration, splits | 🟢 |
| `matplotlib` | reliability diagram, ROC/PR curves | 🟢 |
| `river` *(optional)* | online learning + drift detectors (ADWIN, DDM) | 🔵 |

> **Rule:** no deep-learning frameworks here. This track is fundamentals done well.

### Models to know
- **Logistic Regression** — the baseline; understand the sigmoid, log-loss, and that its raw outputs
  are *probabilities* (relatively well-calibrated out of the box). 🟢
- **Gradient-Boosted Trees** (`sklearn.ensemble.HistGradientBoostingClassifier`, or XGBoost/LightGBM
  if allowed) — the stronger model; understand it's an ensemble of weak learners fit on residuals,
  and that its scores are often *mis*-calibrated. 🟡
- **Random Forest** — bagging baseline, for contrast. 🔵

### Feature engineering (the real work)
- **Lead time** = `AppointmentDay − ScheduledDay` (days). The single strongest feature. 🟢
- **Datetime features** — day-of-week, month, is-weekend, from `pandas.to_datetime`. 🟢
- **Per-entity history** — prior no-show rate per patient (computed *only* from past rows — leakage
  risk!). 🟡
- **Categorical encoding** — one-hot or target encoding for `Neighbourhood`, with care to avoid
  leakage in target encoding. 🟡
- **Data cleaning** — negative `Age`, non-binary `Handcap` (known quirks). 🟢

### Imbalance handling
- **Class weights** (`class_weight='balanced'`) — cheapest, often best. 🟢
- **Resampling** — **SMOTE** (synthetic minority oversampling, `imbalanced-learn`), random
  over/under-sampling. Know SMOTE interpolates between minority neighbours. 🟡
- **Threshold moving** — don't use 0.5; pick the threshold from the PR curve / cost analysis. 🟡

### Evaluation (where the grade lives)
- **ROC-AUC** — ranking quality, threshold-independent. 🟢
- **PR-AUC (Average Precision)** — the *right* metric for rare positives; know why it beats ROC-AUC
  under imbalance. 🟢
- **Reliability diagram** — predicted probability (x) vs. observed frequency (y); the diagonal is
  perfect calibration. 🟢
- **Expected Calibration Error (ECE)** — weighted average gap between confidence and accuracy across
  bins. Lower is better. 🟢
- **Brier score** — mean squared error of probabilities; a single-number calibration+accuracy proxy.
  🟡

### Calibration methods
- **Platt scaling** — fit a logistic regression on the model's scores. `CalibratedClassifierCV(method='sigmoid')`. 🟢
- **Isotonic regression** — non-parametric, monotonic step function; needs more data.
  `CalibratedClassifierCV(method='isotonic')`. 🟡

### Splits & drift
- **Temporal split** — sort by `AppointmentDay`, train early / test late. Use
  `sklearn.model_selection.TimeSeriesSplit` for CV. 🟢
- **Drift detection** — PSI (Population Stability Index) by hand, or ADWIN/DDM via `river`. 🔵
- **Conformal prediction** — distribution-free prediction intervals for honest uncertainty. 🔵

### Knowledge to revise
Sigmoid & log-loss · bias-variance tradeoff · ROC vs. PR under imbalance · what "calibrated" means ·
why random k-fold leaks time-ordered data.

---

## 2. Problem 2 stack — RAG + Verification 🟠

### Libraries
| Library | Use | Depth |
|---------|-----|-------|
| `sentence-transformers` | dense embeddings | 🟢 |
| FAISS / `chromadb` / `numpy` | vector index + similarity search | 🟢 |
| `rank-bm25` / Elasticsearch | lexical (BM25) retrieval | 🟡 |
| `transformers` (HuggingFace) | NLI model (e.g. DeBERTa-MNLI) | 🟢 |
| an LLM (open-weight or API) | generation / verdict reasoning | 🟡 |
| `ragas` | faithfulness / RAG metrics | 🟡 |

### The RAG pipeline (know each stage cold)
1. **Chunking** — split docs into passages/sentences. Chunk size is a real tradeoff (too big = noisy,
   too small = lost context). 🟢
2. **Embedding** — encode chunks + query into vectors with a sentence encoder. Know cosine similarity
   is the distance metric. 🟢
3. **Retrieval** — top-k nearest chunks.
   - **Dense** — semantic, embedding-based. 🟢
   - **Sparse / BM25** — lexical, term-frequency based. 🟡
   - **Hybrid** — combine both (reciprocal rank fusion). 🟡
4. **Re-ranking** *(bonus)* — a cross-encoder re-scores the top-k for precision. 🔵
5. **Verdict** — entailment classification or LLM reasoning over retrieved evidence. 🟢

### Verification / NLI
- **Natural Language Inference (NLI)** — classify (premise, hypothesis) as **entailment /
  contradiction / neutral** → maps to **SUPPORTS / REFUTES / NOT ENOUGH INFO**. 🟢
- **DeBERTa-MNLI / RoBERTa-MNLI** — off-the-shelf entailment classifiers. 🟡
- **FEVER format** — claim + evidence sentences + 3-way label. Evidence F1 means a verdict is only
  "right" if the *cited sentences* are also right. 🟢

### Measuring hallucination
- **RAGAS faithfulness** = (claims in the answer entailed by retrieved context) / (total claims).
  Know the decomposition: split answer → atomic claims → check each. 🟢
- **Context precision / recall** — did retrieval pull the right evidence? 🟡
- **Answer relevancy** — does the answer address the claim? 🔵

### Security — the adversarial layer
- **Indirect prompt injection** — malicious instructions hidden in *retrieved* documents, not the
  user prompt. The Greshake et al. threat model. 🟢
- **Defences:**
  - **Spotlighting / delimiting** — wrap retrieved content in clear markers so the model treats it as
    data, not instructions. 🟡
  - **Instruction–data separation** — system prompt asserts retrieved text is never a command. 🟡
  - **Injection classifier** — a filter that flags payloads before they reach the LLM. 🔵
  - **Ablation** — measure attack-success-rate with defence on vs. off (this is the *evidence* it
    works). 🟡

### Knowledge to revise
Cosine similarity & embeddings · vector databases · BM25/TF-IDF · NLI/entailment · what
"grounded vs. parametric knowledge" means · the prompt-injection threat model.

---

## 3. Problem 3 stack — Autonomous Agents 🔴

### Libraries / building blocks
| Component | Options | Depth |
|-----------|---------|-------|
| LLM backend | open-weight (Llama/Qwen) or API (function-calling capable) | 🟢 |
| Agent framework | LangGraph, AutoGen, CrewAI, OpenAI Agents SDK — *or hand-rolled* | 🟡 |
| Sandbox | Docker container, subprocess with denylist, restricted FS | 🟢 |
| Tool schemas | JSON Schema / Pydantic for validated tool calls | 🟢 |
| Tracing | OpenTelemetry (GenAI semantic conventions) | 🔵 |

> A **hand-rolled loop** is often the best answer in an interview — it proves you understand the
> mechanics frameworks hide.

### The agent loop (know it cold)
**Plan → Act (tool call) → Observe → Reflect → repeat → Submit.**
- **Function calling / tool use** — the LLM emits a structured call (name + JSON args); your harness
  executes it and feeds back the result. Validate args against a schema. 🟢
- **Verification gate** — agent runs the *visible* tests itself; only submits when they pass (or
  budget is exhausted). 🟢
- **Observation / step budget** — hard cap on tool calls / tokens; must terminate. 🟢

### Reflexion (the key technique)
- **Verbal reinforcement** — on failure, the agent writes a self-critique and stores it in **episodic
  memory**, feeding it into the next attempt. No weight updates. 🟢
- **The ablation** — reflection-on vs. reflection-off resolve rate is the experiment that proves it
  works. 🟡

### Search beyond brute retry 🔵
- **Tree-of-Thoughts (ToT)** — branch over candidate reasoning paths, evaluate, backtrack.
- **LATS** — language-agent tree search (MCTS-style over actions).
- **Critic model** — a separate pass that gates submissions.

### The adversarial layer
- **Hidden held-out tests** — visible tests guide, hidden tests judge. Overfitting the visible suite
  earns nothing (the SWE-bench-Verified design). 🟢
- **Reward-hacking detection** — static + runtime checks for: editing/deleting test files,
  hard-coded return values, grader-input special-casing, network exfiltration, out-of-workspace
  writes. All *subtract* from score. 🟢
- **Prompt injection in tool outputs** — planted strings like "task complete, stop now" in file
  contents/command output. The agent must not obey them. 🟡
- **Sandboxing** — command denylist, no network unless allowed, filesystem jail; escaping = DQ. 🟢

### Evaluation
- **% Resolved** — fraction of tasks where hidden tests pass (SWE-bench metric). 🟢
- **Resolve-rate per token** — cost-aware ranking; cheap-and-correct beats expensive-and-slightly-
  better. 🟡
- **Trace replay** — OpenTelemetry spans so a judge can see exactly what the agent did. 🔵

### Knowledge to revise
The plan-act-observe loop · function calling / tool schemas · sandboxing & least privilege ·
Goodhart's Law / reward hacking · Reflexion vs. fine-tuning · held-out vs. visible test design ·
SWE-bench & GAIA at a high level.

---

## 4. Cross-cutting skills (worth real points)

| Skill | Why it matters | Depth |
|-------|----------------|-------|
| **Ablation studies** | "I added X and the metric moved by Y" beats "I added X." All three reward this. | 🟢 |
| **Reproducibility** | Fixed seeds, pinned deps, one-command run. | 🟢 |
| **Reading a paper** | You cite FEVER/SWE-bench/Reflexion — be able to summarise each in 2 sentences. | 🟡 |
| **Error analysis** | "Where did it fail and why" shows maturity. | 🟡 |
| **Cost/latency awareness** | Tokens, inference time, budget. Production thinking. | 🟡 |
| **Threat modelling** | Who's the attacker, what can they touch, how do you measure the defence. | 🟡 |

---

## 5. Fast self-check — can you explain these in one sentence each?

If you can't, revise before the interview.

- [ ] Why accuracy is the wrong metric for an imbalanced problem
- [ ] The difference between ROC-AUC and PR-AUC
- [ ] What "calibrated probabilities" means and how to measure it (ECE)
- [ ] Platt scaling vs. isotonic regression
- [ ] Why a random split leaks the future on time-ordered data
- [ ] What data leakage is, with an example
- [ ] How dense retrieval works (embeddings + cosine similarity)
- [ ] Dense vs. BM25 retrieval, and why hybrid helps
- [ ] NLI and how entailment maps to SUPPORTS/REFUTES/NOT ENOUGH INFO
- [ ] RAGAS faithfulness as a hallucination metric (the formula)
- [ ] Indirect prompt injection and one defence against it
- [ ] The plan-act-observe-reflect agent loop
- [ ] Function calling / structured tool use
- [ ] Goodhart's Law and reward hacking with a concrete example
- [ ] Why Reflexion improves an agent without training
- [ ] Why hidden held-out tests prevent gaming

---

## 6. Tooling install cheat-sheet

```bash
# Problem 1 — tabular
pip install numpy pandas scikit-learn matplotlib imbalanced-learn river

# Problem 2 — RAG + verification
pip install sentence-transformers faiss-cpu rank-bm25 transformers torch ragas

# Problem 3 — agents
pip install openai pydantic   # + Docker for the sandbox; a framework (langgraph/autogen) is optional
```

> Exact versions depend on your environment; always pin them in `requirements.txt` for
> reproducibility. Check the curriculum's dependency allowlist in `AGENTS.md` if you're contributing
> back to this repo.

---

## 7. Curriculum lessons that teach this (in-repo)

- **Problem 1:** [Logistic Regression](phases/02-ml-fundamentals/03-logistic-regression-and-classification/),
  [Handling Imbalanced Data](phases/02-ml-fundamentals/17-handling-imbalanced-data/),
  [Model Evaluation](phases/02-ml-fundamentals/09-model-evaluation-metrics-cross-validation/),
  [Bias, Variance & the Learning Curve](phases/02-ml-fundamentals/10-bias-variance-and-the-learning-curve/).
- **Problem 2:** [RAG](phases/11-llm-engineering/06-rag/),
  [Advanced RAG](phases/11-llm-engineering/07-advanced-rag/),
  [Guardrails](phases/11-llm-engineering/12-guardrails/),
  [NLI / Textual Entailment](phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment/),
  [Embedding Models](phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive/),
  [Indirect Prompt Injection](phases/18-ethics-safety-alignment/15-indirect-prompt-injection/).
- **Problem 3:** [The Agent Loop](phases/14-agent-engineering/01-the-agent-loop/),
  [Reflexion](phases/14-agent-engineering/03-reflexion-verbal-rl/),
  [Tool Use & Function Calling](phases/14-agent-engineering/06-tool-use-and-function-calling/),
  [Verification Gates](phases/14-agent-engineering/38-verification-gates/),
  [Reward Hacking & Goodhart](phases/18-ethics-safety-alignment/02-reward-hacking-goodhart/).
