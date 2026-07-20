# Chapter 3 — Your Own Tech Stack, Defined

| | |
|---|---|
| **Source** | Every term below appears **on your resume**. Nothing else is here |
| **Ordering** | By **interrogation risk**, not alphabetically |
| **Reading time** | ~30 min first pass; then it becomes a reference |
| **Output** | A completed self-check that feeds Chapter 2's Depth Signals score and Chapter 5's plan |

> 📍 **What this chapter is for.** This is a mirror, not a textbook. Every term here is something *you already claimed in writing*. The intended reaction to some entries is productive alarm: *"I wrote that, and I could not define it cleanly right now."* That reaction is the chapter working. Finding it here costs nothing; finding it in the loop costs the offer.

## 🟢 How To Use It

For each term, cover the definition and answer aloud:

1. **Define it in one sentence.**
2. **Survive one follow-up:** *why this and not the obvious alternative?*

Then mark yourself honestly:

| Mark | Meaning | What it implies |
|---|---|---|
| ✅ **Yes** | Defined it cleanly and answered the follow-up | Safe to keep on the resume |
| ⚠️ **Partly** | Got the gist; fumbled the follow-up | Study target — it stays, but it needs work |
| ❌ **No** | Could not define it | **Either learn it or remove it from your resume** |

Every ❌ is a decision, not a failure. A term you cannot defend is not an asset — it is an interrogation opening you handed over voluntarily.

> [!IMPORTANT]
> Do this **before** reading Sessions 1–2's answers. Reading the model answer first and then rating yourself ✅ produces a comfortable, useless self-assessment.

---

## 🔴 Tier 1 — Highest Interrogation Risk

These sit on your headline project, name specific mechanisms, and are the most likely opening moves in a depth round. `risk = P(asked) × depth reachable × centrality`.

| # | Term | One-line meaning | Self-check |
|---|---|---|---|
| 1 | <abbr title="Quantized Low-Rank Adaptation: fine-tuning where the frozen base model is stored in 4-bit and only small added matrices are trained.">**QLoRA**</abbr> | Fine-tune a 4-bit-stored frozen model by training only small added matrices | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 2 | <abbr title="NormalFloat4: a 4-bit storage format whose 16 levels sit at standard-normal quantiles, matching how pretrained weights are distributed.">**NF4**</abbr> | 4-bit format with levels placed at normal-distribution quantiles | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 3 | <abbr title="Low-Rank Adaptation. Freezes pretrained weights and learns each update as the product of two thin matrices.">**LoRA adapters**</abbr> | The trainable thin-matrix pair added to frozen layers | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 4 | <abbr title="The unweighted mean of per-class F1 scores. Every class counts equally however rare, so a collapsed minority class drags the average down.">**macro-F1**</abbr> | Unweighted mean of per-class F1 — rare classes count fully | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 5 | <abbr title="Rotary Position Embedding: rotates query and key vectors by angles proportional to position, so attention scores depend only on relative distance.">**RoPE**</abbr> | Position encoding by rotation; attention depends only on offset | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 6 | <abbr title="Normalization applied inside each sublayer branch rather than on the residual path, leaving gradients an unobstructed route through the network.">**Pre-LayerNorm**</abbr> | Norm inside the branch, clean residual path, trains without warmup | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 7 | <abbr title="A loss that pulls matching pairs together and pushes non-matching pairs apart in embedding space.">**Contrastive Loss**</abbr> | Pulls positives together, pushes negatives apart | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 8 | <abbr title="Expected Calibration Error: the average gap between confidence and accuracy, computed in confidence bins and weighted by bin population.">**ECE**</abbr> | Average gap between stated confidence and actual accuracy | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 9 | <abbr title="Dividing logits by one learned scalar before softmax to fix overconfidence, without changing which class is predicted.">**Temperature Scaling**</abbr> | One scalar on the logits; fixes confidence, never accuracy | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 10 | <abbr title="Retrieval-Augmented Generation: retrieve relevant documents, then condition a generator on them.">**RAG**</abbr> | Retrieve documents, then generate an answer conditioned on them | ☐ ✅ ☐ ⚠️ ☐ ❌ |

**Why Tier 1 is Tier 1:** #4 and #8 are metrics you *named without reporting a value* (Chapter 2, Finding A). Naming a metric guarantees the question. Be able to define it, state your number, and explain what it hides.

---

## 🟠 Tier 2 — High Risk

Named mechanisms an interviewer can drill two or three levels into.

| # | Term | One-line meaning | Self-check |
|---|---|---|---|
| 11 | <abbr title="Multi-Head Attention: several attention functions computed in parallel over different learned projections, then concatenated.">**Multi-Head Attention**</abbr> | Parallel attention heads over different projections, then combined | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 12 | <abbr title="Reducing numeric precision of weights, here from 16-bit floats to 4-bit codes, to cut memory.">**4-bit Quantization**</abbr> | Storing weights in 4 bits instead of 16 to fit memory | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 13 | <abbr title="Parameter-Efficient Fine-Tuning: the family of methods that adapt a model by training a small fraction of its parameters.">**PEFT**</abbr> | Family of methods training a small fraction of parameters | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 14 | <abbr title="A model that takes a query and a document together and scores their relevance jointly, rather than embedding each separately.">**Cross-encoder**</abbr> | Scores a pair jointly rather than embedding each separately | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 15 | <abbr title="A library for fast similarity search over dense vectors, supporting exact and approximate indexes.">**FAISS**</abbr> | Fast similarity search over dense vectors | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 16 | <abbr title="A plot of per-bin accuracy against per-bin confidence; the diagonal is perfect calibration.">**Reliability diagram**</abbr> | Accuracy vs. confidence per bin; diagonal = calibrated | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 17 | <abbr title="A model output that is fluent and confident but unsupported by the retrieved evidence.">**Hallucination**</abbr> | Confident output unsupported by the retrieved context | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 18 | <abbr title="A change in the input distribution between training and deployment while the label relationship stays the same.">**Distribution Shift**</abbr> | Inputs move away from the training distribution | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 19 | <abbr title="A small BAAI general embedding model used to turn text into vectors for retrieval.">**bge-small**</abbr> | Small embedding model turning text into retrieval vectors | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 20 | <abbr title="A transformer encoder trained to map sentences to embeddings whose distances reflect semantic similarity.">**Sentence-Transformers**</abbr> | Encoders producing semantically meaningful sentence vectors | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 21 | <abbr title="An encoder model using disentangled attention over content and position, with ELECTRA-style pretraining in v3.">**DeBERTa-v3**</abbr> | Strong encoder; your stated baseline | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 22 | <abbr title="A 1.5-billion-parameter decoder-only language model from the Qwen2.5 family.">**Qwen2.5-1.5B**</abbr> | The decoder you fine-tuned | ☐ ✅ ☐ ⚠️ ☐ ❌ |

**Trap in this tier:** #14 and #21. You say you *benchmarked against* a DeBERTa-v3 cross-encoder. Expect *"why would a cross-encoder be structurally advantaged here?"* If you cannot answer that, the comparison looks unconsidered.

---

## 🟡 Tier 3 — Medium Risk

Real terms you used, drillable but less likely to be the opening move.

| # | Term | One-line meaning | Self-check |
|---|---|---|---|
| 23 | <abbr title="Amazon's Shopping Queries dataset labelling query-product pairs as Exact, Substitute, Complement, or Irrelevant.">**ESCI dataset**</abbr> | Amazon query–product relevance data, 4 labels | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 24 | <abbr title="When some classes have far more examples than others, so accuracy is dominated by the majority class.">**Class imbalance**</abbr> | Skewed label distribution; why you chose macro-F1 | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 25 | <abbr title="A dataset of question pairs labelled as duplicates or not, used for paraphrase detection.">**Quora Question Pairs**</abbr> | Duplicate-question dataset | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 26 | <abbr title="Learning vector representations of data where geometric closeness encodes semantic similarity.">**Representation Learning**</abbr> | Learning embeddings where distance means similarity | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 27 | <abbr title="The mechanism letting a model weight different parts of its input when producing each output.">**Attention Mechanisms**</abbr> | Weighting input positions when producing each output | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 28 | <abbr title="Searching by meaning using vector similarity rather than keyword overlap.">**Semantic Search**</abbr> | Retrieval by vector similarity, not keyword match | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 29 | <abbr title="A database that indexes dense vectors for nearest-neighbour retrieval.">**Vector Databases**</abbr> | Storage and indexing for embedding retrieval | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 30 | <abbr title="Systematically categorising the ways a system produces wrong outputs, rather than only measuring how often.">**Failure-Mode Analysis**</abbr> | Categorising *how* a system fails, not just how often | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 31 | <abbr title="How well a model maintains accuracy when inputs differ from its training distribution.">**Distribution-Shift Robustness**</abbr> | Holding up when inputs drift | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 32 | <abbr title="A PyTorch library of image models and pretrained weights.">**timm**</abbr> | Image-model library (your calibration project) | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 33 | <abbr title="Adapting a pretrained model to a specific task by continuing training on task data.">**Fine-Tuning**</abbr> | Continuing training on task data | ☐ ✅ ☐ ⚠️ ☐ ❌ |
| 34 | <abbr title="Neural architectures built on self-attention rather than recurrence or convolution.">**Transformer Architectures**</abbr> | Self-attention-based architectures | ☐ ✅ ☐ ⚠️ ☐ ❌ |

---

## ⚫ Tier 4 — Listed But Undemonstrated (Chapter 2, Finding D)

**These are the dangerous ones.** They appear in your Skills section but **no project bullet demonstrates them.** An interviewer reading your skills list may open here — and you would be defending a claim with no supporting work.

| Term | Where it appears | The problem | Decision |
|---|---|---|---|
| **CUDA** | Tools & Systems; ROPE project tech tag | No bullet describes a CUDA-level contribution. Did you write kernels, or use PyTorch on a GPU? | ☐ Defend ☐ **Remove** |
| **SQL** | Programming Languages | No project uses it. Expect "describe a complex query you wrote" | ☐ Defend ☐ **Remove** |
| **Computer Vision** | ML & Deep Learning | Only the calibration project touches images, and there you *consume* a pretrained classifier | ☐ Defend ☐ **Reframe** |
| **C++** | Programming Languages | Genuinely strong (700+ LeetCode) but attached to no project | ☐ **Keep** — back it with LeetCode, not a project |
| **Linux / Git / Jupyter** | Tools & Systems | Table stakes; low risk but zero signal | ☐ Keep (harmless) |

### How to decide

Ask one question per row: **"If an interviewer spends five minutes here, do I come out looking stronger or weaker?"**

- *Stronger* → keep it.
- *Weaker* → remove it. Removing an undemonstrated skill costs you nothing; almost nobody is hired for a keyword, and plenty are damaged by one.

For **CUDA** specifically: if you wrote no custom kernels, the honest options are to drop it, or move it to a phrasing you can defend (e.g. GPU-based training with PyTorch). "CUDA" as a bare skill implies kernel-level work.

---

## 🟢 Score Your Surface

Count your marks across Tiers 1–3 (34 terms):

| Count | What it means | Action |
|---|---|---|
| ✅ **28+** | Strong command of your own claims | Move to Sessions 1–2 depth work |
| ✅ **20–27** | Solid, with real gaps | Study every ⚠️ before mock rounds |
| ✅ **< 20** | Your resume outruns your current recall | **Highest-priority fix.** Study first, interview later |

**Every ❌ in Tier 4 should become a resume edit this week**, not a study task.

### Feed the results back

1. Record your counts in [PROGRESS.md](../PROGRESS.md).
2. **Chapter 2's Depth Signals sub-score (currently a provisional 2/6) is recomputed from this.** Mostly ✅ → it rises toward 6 and your intrinsic score improves. Mostly ⚠️/❌ → it stays low and honestly reflects the risk.
3. Every ⚠️ and ❌ in Tiers 1–2 enters the Chapter 5 study plan, ordered by tier.

> The point of this chapter is not to feel bad about gaps. It is that **you get to choose which of these conversations you have.** Every term you keep is a conversation you have chosen; every term you cut is one you declined.

**Next:** [Chapter 4 — The Introduction](04_the_introduction.md)
