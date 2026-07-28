# Course Progress

> Update this file after every drill session. Every mark here must trace back to an actual attempt you made out loud — there is no credit for a made-up number, including in your own tracker.

---

## 🟢 Mastery Tracker

Legend: `[ ]` untouched · `[k]` knows it · `[d]` can derive it · `[D]` **can defend it** (the hiring bar)

```text
SESSION 1 — FINE-TUNING & ARCHITECTURE          SESSION 2 — SYSTEMS & EVALUATION
├─ [ ] NF4 / quantile quantization              ├─ [ ] Failure-mode operational definitions
├─ [ ] LoRA math (rank, alpha, placement)       ├─ [ ] Sweep design & oracle-context test
├─ [ ] GPU memory accounting (16GB budget)      ├─ [ ] Chunking mechanics & position effects
├─ [ ] Causal-LM-as-classifier design           ├─ [ ] Judge validity & measurement error
├─ [ ] Macro-F1 & ESCI imbalance                ├─ [ ] ECE & estimator pathologies
├─ [ ] RoPE derivation                          ├─ [ ] Temperature scaling: derivation & limits
├─ [ ] Pre-LN vs Post-LN                        ├─ [ ] Why calibration breaks under shift
├─ [ ] Contrastive loss & negatives             ├─ [ ] Acting under miscalibration (abstention)
└─ [ ] MOCK ROUND: Architecture                 └─ [ ] MOCK ROUND: Systems

SESSION 3 — CORE THEORY                         SESSION 4 — DL & TRANSFORMERS
├─ [ ] Bias–variance decomposition              ├─ [ ] Backprop mechanics
├─ [ ] Regularization as priors (L2/L1)         ├─ [ ] Normalization family
├─ [ ] MLE / MAP / cross-entropy                ├─ [ ] Attention variants
├─ [ ] SGD → momentum → Adam → AdamW            ├─ [ ] Training pathologies
├─ [ ] Metrics & validation design              └─ [ ] MOCK ROUND: DL
└─ [ ] MOCK ROUND: ML breadth
                                                SESSION 6 — DSA
SESSION 5 — LLMs & RETRIEVAL                    ├─ [ ] Pattern fluency
├─ [ ] PEFT landscape                           ├─ [ ] Interview protocol
├─ [ ] Decoding & sampling                      └─ [ ] MOCK ROUND: Coding
├─ [ ] RAG design space
├─ [ ] LLM evaluation                           SESSION 8 — LP & STAR
└─ [ ] MOCK ROUND: LLMs                         ├─ [ ] Story bank (12 stories)
                                                ├─ [ ] STAR with quantified results
SESSION 7 — ML FROM SCRATCH                     ├─ [ ] Failure/conflict sourcing
├─ [ ] Attention from blank editor              └─ [ ] MOCK ROUND: Behavioral
├─ [ ] Classic ML from blank editor
└─ [ ] MOCK ROUND: Whiteboard

SESSION 9 — MOCK LOOP
└─ [ ] FINAL MOCK LOOP: all four rounds
```

---

## 🟢 Practice Log

| Date | Drill / chain | Depth survived | Outcome | Notes |
|---|---|---|---|---|
| | | | | |

Recording reminder: log depth survived per chain, note where you run out of depth, and mark whether hand-waving or unbacked numbers caused a level drop.

---

## 🟢 Consistency Tracker

| Week | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| W1 | | | | | | | |
| W2 | | | | | | | |
| W3 | | | | | | | |
| W4 | | | | | | | |

Mark ✅ for any day with at least one drill attempt. Even an unsuccessful attempt counts; a zero-activity day is the real obstacle.

---

## 🟢 Gap Log

Log every point where you ran out of depth. This is next session's study map, not a failure record.

| Date | Session | Question chain | Depth reached (1–5) | Exact point your depth ran out | Follow-up action |
|---|---|---|---|---|---|
| | | | | | |

---

## 🟢 [FILL] Metric Vault

Before Session 1's mock round, pull these from your actual run logs. Any slot you cannot fill becomes a "qualitative-only" topic — you may describe direction and method, never a number.

| Slot | Value from run logs | Source (log/notebook path) |
|---|---|---|
| QLoRA: macro-F1 (test) | [FILL] | |
| QLoRA: per-class F1 (E/S/C/I) | [FILL] | |
| DeBERTa-v3 baseline: macro-F1 | [FILL] | |
| QLoRA: trainable-param % and count | [FILL] | |
| QLoRA: LoRA rank, alpha, target modules | [FILL] | |
| QLoRA: peak GPU memory observed | [FILL] | |
| ROPE model: layers/heads/d_model/params | [FILL] | |
| ROPE model: QQP eval metric + value | [FILL] | |
| ROPE model: baseline compared against (if any) | [FILL] | |
| Contrastive loss: exact formulation + temperature/margin | [FILL] | |

### Session 2 — Systems & Evaluation

Fill before the Systems mock round. Same law: run logs or `QUALITATIVE-ONLY`. The armory notebooks show the *machinery*; only your own artifacts fill these rows.

| Slot | Value from run logs | Source (log/notebook path) |
|---|---|---|
| **RAG — project setup** | | |
| Corpus + split (name, size, gold-label source) | [FILL] | |
| Embedder + query instruction prefix used? (y/n) | [FILL] | |
| FAISS index type + corpus scale | [FILL] | |
| Generator model used | [FILL] | |
| **RAG — headline evidence** | | |
| Failure distribution (miss / ignored / hallucination %) | [FILL] | |
| n (eval questions) | [FILL] | |
| recall@k curve points (k → recall) | [FILL] | |
| End-to-end accuracy at operating point | [FILL] | |
| Oracle-context accuracy (generation floor) | [FILL] | |
| Closed-book baseline (leakage control) | [FILL] | |
| Chunk × k sweep grid + what was held fixed | [FILL] | |
| **RAG — instrument validity** | | |
| Judge type + known bias direction | [FILL] | |
| Judge audit: sample size + human agreement | [FILL] | |
| One real example per failure bucket | [FILL] | |
| **Calibration — project setup** | | |
| Architecture (timm id) + dataset | [FILL] | |
| Corruption set + severity range | [FILL] | |
| ECE config (bins, scheme, n) | [FILL] | |
| **Calibration — headline evidence** | | |
| Clean accuracy / mean confidence / ECE | [FILL] | |
| Per-severity accuracy | [FILL] | |
| Per-severity ECE (fixed-width) | [FILL] | |
| Per-severity ECE (equal-mass cross-check) | [FILL] | |
| Fitted T on clean validation | [FILL] | |
| Oracle T per severity (the mechanism exhibit) | [FILL] | |
| ECE under clean-fitted T, per severity | [FILL] | |
| Worst corruption for calibration + hypothesis why | [FILL] | |
| NLL / Brier reported alongside ECE? | [FILL] | |
