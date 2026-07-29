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
├─ [ ] Bias–variance decomposition              ├─ [ ] Backprop derivation & shapes
├─ [ ] Regularization as priors (L2/L1)         ├─ [ ] Vanishing/exploding & residuals
├─ [ ] MLE / MAP / cross-entropy                ├─ [ ] LayerNorm vs BatchNorm vs RMSNorm
├─ [ ] SGD → momentum → Adam → AdamW            ├─ [ ] Attention O(n²) & the KV cache
├─ [ ] Metrics & validation design              ├─ [ ] MHA / MQA / GQA / FlashAttention
└─ [ ] MOCK ROUND: ML breadth                   ├─ [ ] Training pathologies & debugging
                                                └─ [ ] MOCK ROUND: DL breadth
SESSION 5 — LLMs & RETRIEVAL
├─ [ ] PEFT memory arithmetic & landscape       SESSION 6 — DSA & ALGORITHMS
├─ [ ] Decoding, sampling & speculative         ├─ [ ] The five beats, under a clock
├─ [ ] Scaling laws & compute-optimal           ├─ [ ] Constraint → complexity class
├─ [ ] SFT / RLHF / DPO stack                   ├─ [ ] Pattern + invariant, together
├─ [ ] RAG design space & hybrid retrieval      ├─ [ ] Hidden container costs
├─ [ ] LLM evaluation & paired statistics       ├─ [ ] The three proof shapes
└─ [ ] MOCK ROUND: LLMs                         ├─ [ ] Heap / sort / quickselect
                                                ├─ [ ] Binary search: both hazards
                                                ├─ [ ] Four test classes
                                                ├─ [ ] Streams: heap, reservoir, union–find
                                                └─ [ ] MOCK ROUND: Coding

                                                SESSION 8 — LP & STAR
SESSION 7 — ML FROM SCRATCH                     ├─ [ ] Story bank (12 stories)
├─ [ ] Attention + shapes, from blank editor    ├─ [ ] STAR with quantified results
├─ [ ] The √d_k variance argument               ├─ [ ] Failure/conflict sourcing
├─ [ ] Masking: −∞ before the softmax           └─ [ ] MOCK ROUND: Behavioral
├─ [ ] RoPE + the relative-position proof
├─ [ ] Cross-entropy from logits (log-sum-exp)
├─ [ ] ∇ = p − y, derived
├─ [ ] Finite-difference gradient check
├─ [ ] Label smoothing: gain and cost
├─ [ ] k-means + monotone-inertia proof
├─ [ ] Convex vs non-convex (logreg vs k-means)
├─ [ ] Beam search in log space
├─ [ ] Decoder identity tests
└─ [ ] MOCK ROUND: Implementation

SESSION 9 — MOCK LOOP
└─ [ ] FINAL MOCK LOOP: all four rounds
```

---

## 🟢 Practice Log

| Date | Drill / chain | Depth survived | Outcome | Notes |
|---|---|---|---|---|
| | | | | |

Recording reminder: log the depth you survived per chain, note exactly where your depth ran out, and flag any answer where you hand-waved or quoted a number you could not back.

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

### Session 6 — Coding Protocol

These are **your own rehearsal measurements**, not lab outputs. Take them from a recording of yourself solving a problem out loud on a timer (Reflection prompt 1 in the Session 6 quiz). Nothing from `dsa_lab.ipynb` belongs here — those timings are the notebook's machine, not your performance.

| Slot | Value from your recording | Source (date + problem) |
|---|---|---|
| Seconds from problem statement to first keystroke | [FILL] | |
| Approach statement: all four parts present? (y/n) | [FILL] | |
| Approach statement ended in a question? (y/n) | [FILL] | |
| Longest continuous silence (seconds) | [FILL] | |
| Said "done" or "let me test it"? | [FILL] | |
| Bugs found by you vs. found for you | [FILL] | |
| Beats completed out of 5 | [FILL] | |
| Verified LeetCode solved count (from your profile) | [FILL] | |
| ⚠️ Permitted coding language, per recruiter | [FILL] | |

### Session 7 — Implementation Self-Audit

Claim Vault #8 and #9 are the only résumé claims an interviewer can verify **in the room**. Fill these from your own repository, not from memory — Reflection prompt 1 in the Session 7 quiz walks the audit.

| Slot | Value from your code | Source (file + line) |
|---|---|---|
| ROPE model: layers / heads / d_model / d_head | [FILL] | |
| RoPE layout used: interleaved pairs or split-half | [FILL] | |
| Did you remember the layout correctly before checking? (y/n) | [FILL] | |
| Is RoPE applied anywhere to V? (y/n) | [FILL] | |
| Pre-LN or Post-LN, and where exactly the norm sits | [FILL] | |
| Softmax axis in your attention (key axis?) | [FILL] | |
| Any assertion in the file that would fail if it were wrong | [FILL] | |
| Loss computed from logits or from probabilities | [FILL] | |
| Blank-editor time: attention + RoPE, unaided | [FILL] | |
| Were the clustered embeddings L2-normalised? (Claim Vault #10) | [FILL] | |
