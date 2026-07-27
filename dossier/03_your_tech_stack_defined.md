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

Rehearse one term at a time out loud — pick it, get asked a real interrogation question, record and review your answer, then reveal a detailed model answer to compare against. Do this *before* you self-mark below.

```rehearsal-drill
[[QLoRA]]
T: 1
Q: What is QLoRA, and why use it instead of full fine-tuning or plain LoRA?
A: QLoRA fine-tunes a model whose frozen base weights are stored in 4-bit NF4, while training only small LoRA adapter matrices; gradients flow through the dequantized base but only the adapters update. **Why not full fine-tuning:** full FT of a 1.5B model needs weights + gradients + optimizer state + activations in 16-bit — tens of GB — whereas QLoRA fits it in ~`[FILL]` GB on one GPU. **Why not plain LoRA:** plain LoRA still stores the base in 16-bit; NF4 quantization is what makes it fit. The cost is a small quantization error plus a dequant step each forward pass.

[[NF4]]
T: 1
Q: What is NF4, and why NF4 rather than int4 or FP4?
A: NormalFloat4 is a 4-bit datatype whose 16 levels sit at the quantiles of a standard normal, so each level carries equal probability mass. **Why not int4:** int4 spaces levels uniformly and wastes resolution in the tails where pretrained weights — empirically ~N(0,1) — rarely land. NF4 is information-theoretically optimal *given* the normality assumption — say that caveat aloud, it turns a memorized phrase into an understood one. Levels are normalized to [−1, 1] and applied per block with a separate scale.

[[LoRA adapters]]
T: 1
Q: What exactly is a LoRA adapter, and why does training under 2% of parameters work?
A: For a frozen weight W, LoRA learns ΔW = BA where B (d×r) and A (r×k) are thin matrices of rank r ≪ d; only A and B train, scaled by α/r. **Why so few params suffice:** the task-specific update to a pretrained weight is empirically low-rank — most adaptation lives in a small subspace — so a rank-`[FILL]` update captures it. At inference you can merge BA back into W for zero added latency.

[[macro-F1]]
T: 1
Q: Define macro-F1 as you computed it, and why macro-F1 rather than accuracy or micro-F1 on ESCI?
A: Macro-F1 is the unweighted mean of per-class F1 across the four ESCI classes, so every class counts equally regardless of frequency. **Why not accuracy / micro-F1:** ESCI is dominated by 'Exact', so both reward the majority class and hide collapse on rare classes like Complement; macro-F1 penalizes that collapse. My value was `[FILL: macro-F1]`. What it hides: one weak class drags the mean far below the per-example hit rate.

[[RoPE]]
T: 1
Q: Derive RoPE from what you want the attention score to satisfy, and why RoPE over learned absolute embeddings?
A: I want ⟨f(q,m), f(k,n)⟩ to depend only on the offset m − n. Rotating each 2-D pair of the query and key by an angle proportional to position does exactly that: rotation matrices are orthogonal, so R(mθ)ᵀR(nθ) = R((n−m)θ) and the absolute positions cancel. Each dimension pair gets frequency θᵢ = 10000^(−2i/d), a multi-scale clock. **Why not learned absolute embeddings:** those fix a maximum length and encode absolute position; RoPE encodes relative position, extrapolates further, and is applied to q and k only, not v.

[[Pre-LayerNorm]]
T: 1
Q: What is Pre-LN, and why Pre-LN over Post-LN?
A: Pre-LN puts the LayerNorm inside each sublayer branch (norm → attention/FFN → add), so the residual path is a clean sum. **Why not Post-LN:** Post-LN normalizes after the residual add, so every layer's gradient passes through a LayerNorm; at depth that compounds and needs a warmup schedule to train stably. Pre-LN gives well-behaved gradients at initialization and trains without warmup. The cost: a slightly lower performance ceiling and a residual stream that grows across depth, which the final norm must absorb.

[[Contrastive Loss]]
T: 1
Q: Write your contrastive loss and explain the temperature; what breaks if it's set wrong?
A: For matched (anchor, positive) pairs I pull embeddings together and push mismatches apart, e.g. InfoNCE: −log[ exp(sim(zᵢ,zⱼ)/τ) / Σ exp(sim(zᵢ,z_k)/τ) ]. Temperature τ sharpens the softmax over similarities. **Wrong τ:** too low over-weights the hardest (often false) negatives and can collapse the space to a point; too high flattens gradients so fine distinctions never form. Sweet spot ≈ 0.05–0.1 for sentence tasks; mine was `[FILL]`. Watch false negatives — on QQP many in-batch "negatives" are true duplicates.

[[ECE]]
T: 1
Q: Define ECE exactly as you computed it, then name three ways the number lies.
A: Bin predictions by confidence; ECE is the population-weighted average of |accuracy − mean confidence| across bins. **Three ways it lies:** (1) binning — too few bins hide miscalibration, too many make bins noisy; (2) it's an average, so over- and under-confidence in different regions cancel; (3) it's top-label only, so a model can look calibrated overall yet be badly miscalibrated per class. My value was `[FILL]`; adaptive-bin ECE or Brier score address some of these.

[[Temperature Scaling]]
T: 1
Q: Derive temperature scaling and its objective; why can one scalar never change accuracy?
A: Divide every logit by a single learned T > 1 before softmax, and fit T by minimizing NLL on a held-out set. **Why NLL:** it's a proper scoring rule minimized at the true probabilities, and with logit *directions* fixed it has a clean 1-D optimum. **Why accuracy can't change:** dividing by a positive scalar is monotonic — it never moves the argmax, only the confidence. That's also its limit under shift: one global scalar can't track a miscalibration that varies across the input space.

[[RAG]]
T: 1
Q: What is RAG, and why retrieve-then-generate instead of fine-tuning the knowledge into the model?
A: Retrieve relevant documents by dense embedding search, then condition the generator on them so answers are grounded in retrieved text. **Why not bake it into the weights:** retrieval updates instantly (swap the index) with no retraining, can cite sources, and scales to corpora too large to memorize; fine-tuning bakes in stale facts and hallucinates beyond them. The cost is a retrieval stage that can itself fail — which is exactly the failure-mode analysis I did.

[[Multi-Head Attention]]
T: 2
Q: Walk the shapes of multi-head attention; why multiple heads, and why the √d scaling?
A: Each of h heads projects X to Qᵢ, Kᵢ, Vᵢ ∈ ℝ^(seq×d_head), computes softmax(QᵢKᵢᵀ / √d_head)·Vᵢ, and the heads concatenate back to d_model through W_O. **Why multiple heads:** each can attend to a different relation — syntax, position, rare tokens — in its own subspace; a single head must average them. **Why √d_head:** without it, dot products grow with dimension and push softmax into saturation where gradients vanish; dividing by √d_head restores unit variance.

[[4-bit Quantization]]
T: 2
Q: What does 4-bit quantization actually store, and why doesn't it destroy the model?
A: Instead of a 16-bit float per weight, store a 4-bit code indexing one of 16 levels, plus a shared (often per-block) scale to reconstruct approximate values. **Why it survives:** inference is dominated by matrix products that are robust to small per-weight error, per-block scales stop outliers from dominating, and the base is frozen so errors don't compound through training. It's lossy — you trade a small accuracy hit for roughly a 4× memory cut.

[[PEFT]]
T: 2
Q: What is PEFT, and when would you NOT use it?
A: Parameter-Efficient Fine-Tuning is the family (LoRA, adapters, prefix/prompt tuning) that adapts a model by training a small fraction of parameters while freezing the rest. **When not to:** with abundant data and compute where you need maximum task performance, full fine-tuning still edges it; and when the task needs to reshape low-level representations broadly, a tiny low-rank update can underfit. PEFT wins on memory, on storage (swap tiny per-task adapters), and on reduced overfitting with small data.

[[Cross-encoder]]
T: 2
Q: What is a cross-encoder, and why is it structurally advantaged over your bi-encoder baseline?
A: A cross-encoder feeds the query and document *together* through the model, so every query token attends to every document token, then outputs one relevance score. **Why advantaged:** that full cross-attention captures fine-grained interactions a bi-encoder — which embeds each side independently and then dots them — structurally cannot. The trade-off is cost: it must re-run per candidate and can't pre-index, so it's a reranker, not a first-stage retriever. That's the honest reason it beats my setup on accuracy but not on scalability.

[[FAISS]]
T: 2
Q: What is FAISS doing, and exact vs approximate search — which did you use and why?
A: FAISS indexes dense vectors for fast nearest-neighbor search under a chosen metric. **Exact (flat)** scans every vector — correct but O(N) per query. **Approximate (IVF / HNSW)** trades a little recall for large speedups by only searching likely regions. For `[FILL: corpus size]` I used `[FILL: flat/IVF]`; at my scale exact was fine, and I'd move to IVF/HNSW only when latency at N demanded it — knowing the recall@k I'd give up.

[[Reliability diagram]]
T: 2
Q: What does a reliability diagram show, and how do you read miscalibration off it?
A: It plots, per confidence bin, the model's mean confidence (x) against its actual accuracy (y); the 45° diagonal is perfect calibration. **Reading it:** points below the diagonal mean overconfidence (says 0.9, right 0.7 of the time), above means underconfidence. It's the visual companion to ECE — the diagram shows *where* on the confidence range the model is miscalibrated, which the single ECE number hides.

[[Hallucination]]
T: 2
Q: Define hallucination precisely in a RAG system, and how do you attribute it?
A: A hallucination is a fluent, confident output not supported by the retrieved evidence — distinct from a retrieval miss, where the right document was never fetched. **Attribution:** hand the generator the gold/oracle context; if it still fabricates, the fault is generation, if it's now correct, the fault was retrieval. That oracle-context test is how I separated generation-dominant from retrieval-dominant errors instead of lumping all wrong answers together.

[[Distribution Shift]]
T: 2
Q: What is distribution shift, and which kind did your calibration project study?
A: The input distribution P(x) changes between training and deployment while the task P(y|x) is assumed stable — that's covariate shift; the other kinds are label shift and concept drift. **Mine:** synthetic corruption severity (ImageNet-C-style) as a controlled covariate shift, so I could measure calibration as inputs move progressively off-distribution. Naming which shift you mean matters — the fix differs by type.

[[bge-small]]
T: 2
Q: Why bge-small for retrieval, and what did you leave on the table?
A: bge-small is a compact BAAI general-embedding model mapping text to retrieval vectors — cheap to run and index. **Trade-off:** a larger embedder (bge-base/large or an instruction-tuned model) would raise recall, and a cross-encoder reranker on top would raise precision; I chose small for speed and cost at `[FILL]` scale. The retrieval quality I gave up is exactly what a reranking stage would recover.

[[Sentence-Transformers]]
T: 2
Q: What do Sentence-Transformers do that a raw BERT [CLS] embedding doesn't?
A: They fine-tune an encoder — usually with a siamese/contrastive objective and mean-pooling — so cosine distance between sentence embeddings reflects semantic similarity. **Why not raw BERT:** off-the-shelf BERT embeddings are anisotropic and not trained for cosine comparison, so raw [CLS] similarity is weak. The contrastive training is what makes the embedding geometry meaningful.

[[DeBERTa-v3]]
T: 2
Q: What makes DeBERTa-v3 strong, and is it a fair baseline for your decoder?
A: DeBERTa uses disentangled attention — separate content and relative-position vectors — and v3 adds ELECTRA-style replaced-token-detection pretraining, making it very strong per parameter. **Fairness:** as a cross-encoder it's structurally advantaged for pairwise relevance and is far smaller than my 1.5B decoder — so if it wins it may be the architecture, not the size, and if my decoder wins that's a real result. State which won (`[FILL]`) and attribute it to architecture vs scale, not just the number.

[[Qwen2.5-1.5B]]
T: 2
Q: Why a 1.5B decoder-only model for a classification task instead of an encoder?
A: Qwen2.5-1.5B is a decoder-only LM; I adapt it to 4-class relevance by reading logits over the label tokens (or a head on the last hidden state). **Why a decoder for classification:** to demonstrate PEFT on a generative LM, and because the same setup extends to generative relevance and explanations — but I'll own that an encoder like DeBERTa is the more natural, cheaper choice for pure classification. That trade-off is exactly what the project compares.

[[ESCI dataset]]
T: 3
Q: What is the ESCI dataset, and what makes its labels tricky?
A: Amazon's Shopping Queries dataset labels query–product pairs as Exact, Substitute, Complement, or Irrelevant. **Tricky part:** the classes are heavily imbalanced toward Exact, and Substitute vs Complement is genuinely ambiguous even for humans — so label noise bounds achievable accuracy, and macro-F1 is needed to see minority-class performance.

[[Class imbalance]]
T: 3
Q: How did class imbalance affect your ESCI work, and what did you do beyond picking macro-F1?
A: Exact dominates, so a naive model maximizes accuracy by under-predicting rare classes. **Beyond the metric:** the training-side options are class-weighted loss, resampling/oversampling the minority classes, or focal loss; I `[FILL: what you did / would do]`. The honest answer names the metric choice *and* a training-side mitigation, since the metric only measures the problem — it doesn't fix it.

[[Quora Question Pairs]]
T: 3
Q: What is QQP, and what is its main evaluation pitfall?
A: QQP is a paraphrase dataset of question pairs labeled duplicate / not-duplicate, used to train and evaluate sentence similarity. **Pitfall:** transitivity leakage — if (A,B) and (B,C) are duplicates, a random pair-level split can put A–B in train and A–C in test, letting the model memorize the cluster; proper evaluation splits by question cluster. Labels are also crowd-noisy on borderline paraphrases.

[[Representation Learning]]
T: 3
Q: What is representation learning, and how do you know a representation is good?
A: Learning vector encodings where geometric structure — distance, direction — carries semantics, rather than hand-engineering features. **How you know it's good:** it transfers — the embeddings support downstream tasks with little extra training, cluster by meaning, and stay useful off the training distribution. For my contrastive model, "good" meant positives close and negatives far under cosine, verified on held-out pairs.

[[Attention Mechanisms]]
T: 3
Q: Explain attention as a mechanism, not a formula; why did it replace recurrence?
A: Attention lets each output position pull a weighted combination of all input positions, where the weights come from query–key similarity — so the model routes information directly between any two tokens. **Why over recurrence:** RNNs pass information step by step, so long-range dependencies decay and computation is sequential; attention connects distant tokens in one hop and parallelizes across the sequence, which is what let transformers scale.

[[Semantic Search]]
T: 3
Q: How is semantic search different from keyword search, and when does it lose?
A: Semantic search embeds query and documents into a vector space and retrieves by similarity, matching meaning even without shared words. **When it loses:** exact-match needs — part numbers, names, rare tokens — where lexical/BM25 search is stronger, and out-of-domain terms the embedder never learned. The robust answer is hybrid retrieval (dense + sparse), not treating semantic as strictly better.

[[Vector Databases]]
T: 3
Q: What does a vector database add over just calling FAISS?
A: It indexes dense vectors for approximate nearest-neighbor retrieval like FAISS, but adds production concerns: persistence, metadata filtering, CRUD/updates, sharding, and concurrency. **Why it matters:** FAISS is an in-process index; a vector DB is the serving layer around it. At my project scale FAISS alone sufficed — I'd reach for a vector DB when the corpus changes continuously or must be queried by a service.

[[Failure-Mode Analysis]]
T: 3
Q: What is failure-mode analysis, and why is it more useful than an accuracy number?
A: Systematically categorizing *how* a system produces wrong outputs — retrieval miss vs. generation hallucination vs. chunking artifact — not just how often. **Why more useful:** an aggregate error rate tells you nothing about what to fix; a taxonomy with per-mode frequencies tells you where a week of effort buys the most error reduction. It turns "12% wrong" into a prioritized action list.

[[Distribution-Shift Robustness]]
T: 3
Q: What does robustness to distribution shift mean, and is calibration part of it?
A: Robustness is maintaining performance — and honest uncertainty — as inputs drift from training. **Calibration angle:** accuracy robustness and calibration robustness are different; my project showed a model can keep predicting while its confidence becomes a lie under shift. So "robust" must specify which — a model that fails but *knows* it's unsure is safer than one that's confidently wrong.

[[timm]]
T: 3
Q: What is timm, and what did you use it for?
A: timm (PyTorch Image Models) is a library of image architectures and pretrained weights. **Use:** in the calibration project I *consumed* a pretrained classifier from timm as the fixed model whose confidence I measured under shift — I didn't train it, I instrumented it. Being precise that I consumed rather than trained it keeps the claim honest.

[[Fine-Tuning]]
T: 3
Q: What is fine-tuning, and how do you decide full vs parameter-efficient?
A: Continuing training of a pretrained model on task data so its weights specialize. **Full vs PEFT:** full fine-tuning updates all weights — best raw performance, but heavy on memory/storage and prone to overfitting on small data; PEFT (LoRA/adapters) trains a small fraction, fits one GPU, and stores tiny per-task deltas. Decide by data size, compute budget, and whether you need many swappable task variants.

[[Transformer Architectures]]
T: 3
Q: What defines a transformer, and what's the difference between encoder-only, decoder-only, and encoder–decoder?
A: A transformer stacks self-attention and feed-forward blocks with residual connections and no recurrence. **The three forms:** encoder-only (bidirectional attention — BERT/DeBERTa, good for understanding/classification); decoder-only (causal masked attention — Qwen, good for generation); encoder–decoder (T5 — the encoder reads, the decoder generates conditioned on it). Both my projects live here: a decoder I fine-tuned and an encoder I built from scratch.
```

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
