# Session 2 — Project Deep-Dives: Systems & Evaluation

| | |
|---|---|
| **Projects defended** | RAG failure-mode analysis · Calibration under distribution shift |
| **Prerequisites** | Session 1 assessment passed; your run logs and eval artifacts for both projects |
| **Session length** | 3 lessons, ~4–6 hours including drills and assessment |
| **Assessment** | Systems Bar Raiser (lesson 3) — passing it unlocks Session 3 |

---

## 🟢 Why These Two Projects Share a Session

Session 1 defended things you *built*; this session defends things you *measured*. Both projects here are measurement studies — one diagnoses a pipeline, the other diagnoses a model's self-knowledge — and the interrogation style changes accordingly. The questions stop being "derive the mechanism" and become "**defend the methodology**": how you labeled failures, what your estimator hides, what would falsify your conclusion. For an Applied Scientist role this is the highest-signal round there is, because experimental judgment is the job.

The two projects also share a failure pattern to defend against: both make a *causal-sounding claim* ("generation, not retrieval, is the dominant error source"; "temperature scaling fails to transfer") from *observational sweeps*. Every hard follow-up in this session attacks the gap between what you measured and what you concluded.

---

## 🟢 Scope Brief: Everything an Interviewer Could Raise

### 🔷 From the RAG failure-mode bullet

| Resume phrase | Interrogation surface |
|---|---|
| "bge-small embeddings + FAISS" | Model choice at that size, the bge query-instruction prefix, normalization and metric, index type, corpus scale |
| "labeled QA corpus" | Which corpus, what the labels license you to measure, gold-passage availability, leakage between corpus and generator pretraining |
| "trace every wrong answer to its root cause" | The operational definitions: how instrumentation decides retrieval-miss vs. retrieved-but-ignored vs. hallucination; boundary cases; inter-annotator agreement |
| "controlled sweeps over chunk size and retrieval depth" | Sweep design, what was held fixed, the mechanism by which chunk size moves each failure mode, context-budget confounds |
| "isolating generation … as the dominant error source" | The oracle-context experiment, falsifiability, judge validity, whether the conclusion survives a different generator |
| (implicit) | Answer-correctness measurement itself: exact match vs. LLM-judge vs. human, and its error rate |

### 🔷 From the calibration bullet

| Resume phrase | Interrogation surface |
|---|---|
| "Expected Calibration Error" | Precise definition, binning-estimator pathologies, top-label vs. classwise ECE, what ECE can't see |
| "reliability diagrams" | How to read them, bin counts and confidence mass, the gap-vs-accuracy visual |
| "graded corruptions (blur, noise, brightness, JPEG)" | The corruption-benchmark methodology, severity scaling, which shifts hurt calibration most and why |
| "confidence remained high while accuracy collapsed" | The mechanism of overconfidence under covariate shift, quantifying the decoupling |
| "temperature scaling fails to transfer" | Derivation of TS, why one scalar can't serve a family of shifted distributions, oracle per-severity temperature, alternatives (ensembles, abstention) |
| (implicit) | What you'd actually ship: selective prediction, shift detection, monitoring ⚠️ JD-DEPENDENT |

⚠️ JD-DEPENDENT: production RAG system design (rerankers, caching, latency budgets) is out of scope for a default fresher loop but flips in for teams shipping retrieval products — Session 5 covers the design space; here you need one honest paragraph, not an architecture.

---

## 🟢 Session Structure

1. [Lesson 1 — Defending the RAG Failure-Mode Analysis](01_defending_rag_failure_modes.md)
2. [Lesson 2 — Defending Calibration Under Distribution Shift](02_defending_calibration_under_shift.md)
3. [Lesson 3 — Tiered Challenges & Assessment](03_tiered_challenges_and_boss_fight.md)

Before lesson 3, fill the Session 2 rows of the **Metric Vault** in [PROGRESS.md](../../PROGRESS.md). Same law as Session 1: every number from run logs, `[FILL]` or silence.

---

## 🔷 The Armory (Interactive Notebooks)

Same rules as Session 1: notebooks produce evidence, and defense in drills remains the only measure of mastery.

| Notebook | Launch | Pairs with | What it produces |
|---|---|---|---|
| RAG Failure Triage Lab | [▶ Open in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/02_systems_and_evaluation/01_rag_failure_triage_lab.ipynb) | Lesson 1 | A working miniature of your instrumentation: chunk×k sweep, the three-way failure classification, and the oracle-context condition — the exact machinery your conclusion depends on (CPU-only) |
| Calibration & Temperature Lab | [▶ Open in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/02_systems_and_evaluation/02_calibration_and_temperature_lab.ipynb) | Lesson 2 | ECE from scratch, reliability diagrams, temperature fitting on clean validation, and the transfer-failure curve under graded corruption (CPU-only) |

> [!NOTE]
> Both labs run on synthetic/public data so they're executable anywhere — they demonstrate the *machinery*, not your results. Numbers for the Metric Vault come from your own project artifacts only.

---

## 🟢 Session 2 Core Topics (Gold required to pass the assessment)

- Failure-mode operational definitions
- Sweep design & the oracle-context test
- Chunking mechanics & position effects
- Judge validity & measurement error
- ECE & its estimator's pathologies
- Temperature scaling: derivation and limits
- Why calibration breaks under shift
- Acting under miscalibration (abstention, risk–coverage)
