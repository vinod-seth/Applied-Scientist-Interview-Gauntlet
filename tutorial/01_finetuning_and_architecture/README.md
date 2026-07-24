# Session 1 — Project Deep-Dives: Fine-Tuning & Architecture

| | |
|---|---|
| **Projects defended** | QLoRA on ESCI · From-scratch RoPE transformer |
| **Prerequisites** | Course README (course structure), your own run logs for both projects |
| **Session length** | 3 lessons, ~4–6 hours including drills and assessment |
| **Assessment** | Architecture Bar Raiser (lesson 3) — passing it unlocks Session 2 |

---

## 🟢 Why This Session Is First

These two projects are your strongest interview assets: fine-tuning efficiency and transformer internals are exactly the NLP/LLM specialization the role wants, and both projects are deep enough to sustain a full 50-minute science round. That also makes them the highest-risk surface on your resume — every line in those two bullets is a potential five-why chain. You defend your best material first, while motivation is highest.

---

## 🟢 Scope Brief: Everything an Interviewer Could Raise

If it’s on this list, it’s fair game in the assessment. Nothing outside this list appears in Session 1.

### 🔷 From the QLoRA bullet

| Resume phrase | Interrogation surface |
|---|---|
| "QLoRA (4-bit NF4 + LoRA adapters)" | Quantile quantization, why NF4 over int4/FP4, block size, double quantization, dequantization in the forward pass, gradient flow through frozen quantized weights |
| "training <2% of parameters" | LoRA math (rank, alpha/scaling, target modules), why low-rank adaptation works, what you lose vs. full fine-tuning |
| "single 16GB GPU" | Complete memory accounting: weights, gradients, optimizer states, activations; paged optimizers; gradient checkpointing |
| "4 classes, 20K+ pairs" | ESCI label semantics (Exact/Substitute/Complement/Irrelevant), imbalance, ordinal structure, locale/split hygiene, leakage risks |
| "macro-F1 as primary metric" | Why macro over micro/accuracy, what macro-F1 hides, per-class analysis, threshold/loss alternatives for imbalance |
| "benchmarked against DeBERTa-v3 cross-encoder" | Encoder vs. decoder for pair classification, fairness of the comparison, capacity vs. adaptation trade-off |
| (implicit) | How a causal LM produces a 4-class prediction at all: head vs. verbalizer, pooling token, padding side, prompt template |

### 🔷 From the ROPE transformer bullet

| Resume phrase | Interrogation surface |
|---|---|
| "Rotary Position Embeddings" | Full derivation from the relative-position requirement, why rotation, frequency spectrum, why applied to q/k but not v, extrapolation behavior |
| "Pre-LayerNorm Transformer" | Pre-LN vs. Post-LN gradient behavior, warmup dependence, residual-stream growth, final LayerNorm |
| "Multi-Head Attention" | 1/√d_k derivation, head-dimension trade-offs, what heads specialize in, projection matrix shapes |
| "custom Contrastive Loss" | Exact formulation you used, temperature/margin role, in-batch negatives, false negatives on QQP, hard-negative mining, embedding collapse |
| "Quora Question Pairs" | Dataset pitfalls: label noise, transitivity leakage across splits, sentence-length artifacts |
| "FAISS" / "semantic vector clustering" | Index choice (Flat/IVF/HNSW), metric choice, normalization, recall/latency trade-off |
| (implicit) | Why build from scratch at all; what a pretrained sentence-transformer baseline would have scored |

⚠️ JD-DEPENDENT: deploying either model (serving latency, batching, monitoring) is out of scope for a default fresher loop; it flips in if the team runs production search/retrieval. Prepare a one-paragraph answer, not a deep one — flagged where relevant in the lessons.

---

## 🟢 Session Structure

1. [Lesson 1 — Defending QLoRA on ESCI](01_defending_qlora_on_esci.md): concept refreshers at peer level + the 6 hardest follow-ups with model answers.
2. [Lesson 2 — Defending the RoPE Transformer](02_defending_the_rope_transformer.md): same format.
3. [Lesson 3 — Tiered Challenges & Assessment](03_tiered_challenges_and_boss_fight.md): Bronze/Silver/Gold drills per core topic, then the Architecture Bar Raiser.

Before lesson 3, fill the **Metric Vault** in [PROGRESS.md](../../PROGRESS.md) from your run logs. The assessment assumes it’s filled; every unfilled slot becomes a "qualitative-only" topic in your answers.

---

## 🔷 The Armory (Interactive Notebooks)

Three Colab-ready notebooks accompany this session. They are the **armory, not a track**: they produce the evidence you defend in drills. Defense in drills remains the only measure of mastery.

| Notebook | Launch | Pairs with | What it produces |
|---|---|---|---|
| Memory Budget Verifier | [▶ Open in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/01_finetuning_and_architecture/01_memory_budget_verifier.ipynb) | Lesson 1 | Your paper memory budget confronted with real `torch.cuda` allocator numbers for Qwen2.5-1.5B in NF4; your true trainable-param count (GPU runtime: Colab T4) |
| Architecture Unit Tests | [▶ Open in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/01_finetuning_and_architecture/02_architecture_unit_tests.ipynb) | Lesson 2 | Five executable claims: RoPE shift invariance, why V is never rotated, padding-mask equivalence, the last-token indexing bug live, and the 1/√d_k variance measurement (CPU-fine) |
| Metric Vault Extractor | [▶ Open in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/01_finetuning_and_architecture/03_metric_vault_extractor.ipynb) | Lesson 3 | Per-class F1, confusion structure, and alignment/uniformity diagnostics from **your** artifacts — the fill pipeline for the Metric Vault (runs in clearly-labeled demo mode until you point it at real prediction dumps) |

> [!NOTE]
> These launch links open GitHub-hosted notebooks in Colab. Open them in a real browser signed into a Google account. The notebook files' internal "Open In Colab" image badges are for GitHub's web view; a plain text link is used here because some markdown renderers mishandle image-wrapped links.

> [!IMPORTANT]
> Notebook 3's demo mode exists so you can see the machinery — demo numbers never enter the vault, the pitch, or the resume. Only REAL-mode outputs, with their source paths recorded, count.

---

## 🟢 Session 1 Core Topics (Gold required to pass the assessment)

- NF4 / quantile quantization
- LoRA math (rank, alpha, placement)
- GPU memory accounting (the 16GB budget)
- Causal-LM-as-classifier design
- Macro-F1 & ESCI imbalance
- RoPE derivation
- Pre-LN vs. Post-LN
- Contrastive loss & negatives
