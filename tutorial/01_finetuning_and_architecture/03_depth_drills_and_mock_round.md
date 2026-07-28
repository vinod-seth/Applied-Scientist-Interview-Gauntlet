# Lesson 3 — Depth Drills & Mock Round: Architecture Bar Raiser

| | |
|---|---|
| **Prerequisites** | Lessons 1–2 complete; Metric Vault in PROGRESS.md filled from run logs |
| **Session time** | ~90 min (drills + mock round) |
| **Domain tag** | Interview Simulation |
| **Objective** | Reach Level 3 (can defend it) on all 8 core topics |

🔬 **Interactive companion** (run this to fill the vault before the mock round): [▶ Open the Metric Vault Extractor notebook in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/01_finetuning_and_architecture/03_metric_vault_extractor.ipynb)

> 📍 **Roadmap:** Depth drills first (self-score honestly), then the mock round. The mock round is a scripted Bar Raiser interrogation — five levels deep on the session's core topics. It shows you exactly where your depth runs out.

## 🟢 Before You Start

1. Open [PROGRESS.md](../../PROGRESS.md) in a second window.
2. Confirm every `[FILL]` slot in the Metric Vault is either filled from your run logs or explicitly marked `QUALITATIVE-ONLY`. The mock round will probe any slot you filled — an unsupported number ends the attempt.
3. Have your two pitch scripts (QLoRA, RoPE transformer) ready. You'll deliver them under time pressure.

---

## 🟢 Depth Drills

For each of the 8 core topics, attempt all three levels in order. Record your honest level in PROGRESS.md. Skip to the next topic once you run out of depth — that's the gap log doing its job.

### Topic 1: NF4 / Quantile Quantization

**Level 1 — State it correctly** (no notes)

Explain NF4 in ≤ 3 sentences. What is it, why does it beat uniform int4, and what assumption does it rely on?

<details>
<summary>🔑 Level 1 check</summary>

Must include: quantile-based levels, normal-distribution assumption, blockwise absmax scaling. If you said "it compresses weights" without the mechanism, you have not reached Level 1.
</details>

**Level 2 — Derive it**

Starting from a standard normal distribution, explain why equal-probability-mass bins minimize expected quantization error. Then calculate the storage overhead of double quantization: block size 64, fp32 → 8-bit constants, in blocks of 256.

<details>
<summary>🔑 Level 2 check</summary>

Equal-mass bins: each level represents equal probability density, so the expected distance from any sample to its nearest level is minimized (minimax over the distribution). Double quantization: 64-block needs one fp32 (32 bits) per 64 params = 0.5 bits/param. Quantizing that fp32 to 8-bit in groups of 256 saves 24 bits per constant but adds one fp32 per 256 constants. Net: ~0.127 bits/param overhead. If you got the direction right but the numbers wrong, that is Level 1, not Level 2.
</details>

**Level 3 — Defend it** (5 follow-ups)

Drill chain — answer each before reading the next:

1. When does the normality assumption break, and what happens to NF4 quality?
2. Why block size 64 and not 32 or 128? What's the trade-off curve?
3. How does NF4 interact with outlier channels — when does int8 decomposition (LLM.int8()) win?
4. If you dequantize to bf16 for compute, where does the quantization error enter the final output?
5. Design an experiment to measure the accuracy cost of NF4 vs. fp16 on your specific ESCI task.

<details>
<summary>🔑 Level 3 check</summary>

1. Heavy-tailed weight distributions (e.g., from poorly trained models or models with activation outliers that propagate to weights) shift mass away from the assumed quantiles; NF4 levels misallocate, increasing error in the tails. The block absmax partially mitigates this per-block, but systematic non-normality (e.g., bimodal weights) defeats the scheme.
2. Smaller blocks = more scaling constants = more memory overhead but better outlier isolation. Larger blocks amortize the constant but let one outlier distort 128 values. 64 is the empirical sweet spot from Dettmers et al. — but the right answer includes *why* it's a trade-off, not just the number.
3. LLM.int8() decomposes weight matrices into outlier channels (kept in fp16) and non-outlier channels (quantized to int8). When a small fraction of channels have very large magnitudes (common in larger LLMs, 6B+), this is better than NF4's per-block approach because it preserves the outlier channels exactly. At 1.5B parameters, outlier structure is typically less severe, making NF4 competitive.
4. The quantization error is a constant additive perturbation to each weight: W_actual = W_NF4 + ε. Forward output = (W_NF4 + ε)x = W_NF4·x + ε·x. The error term ε·x scales with input magnitude. Over many layers this can accumulate, but for inference and fine-tuning (where only LoRA params update), the perturbation is fixed — it shifts the function but LoRA can compensate to some degree.
5. Run three conditions: (a) full-precision inference with the same LoRA adapters merged into fp16 weights, (b) NF4 inference, (c) NF4 training + NF4 inference. Compare macro-F1 and per-class F1 across all three. The (a)−(b) gap isolates inference-time quantization cost; the (a)−(c) gap captures accumulated training-time effects.

All five must be directionally correct and mechanistic, not hand-waved. If you hit the floor at #3 or #4, that's your study target — log it.
</details>

---

### Topic 2: LoRA Math (Rank, Alpha, Placement)

**Level 1 — State it:** State the LoRA decomposition for a weight matrix W, including the scaling factor. What do A and B initialize to?

**Level 2 — Derive it:** Derive the trainable parameter count for your specific QLoRA setup: `[FILL: rank, target modules, model size]`. Then explain why α/r decouples the learning rate from the rank.

**Level 3 — Defend it (5 follow-ups):**

1. Why does LoRA work despite the low-rank constraint? (cite intrinsic dimensionality)
2. What's the evidence that LoRA *doesn't* fully match full fine-tuning? (cite Biderman et al.)
3. "Would you rather double rank or add target modules?" — answer with the QLoRA paper's finding.
4. At inference, you merge ΔW into the base. What happens if the base is NF4-quantized?
5. Design a rank-sweep experiment for your ESCI task. What would a flat curve at r=16 tell you?

<details>
<summary>🔑 Level 3 check</summary>

1. Aghajanyan et al. 2020: fine-tuning objectives have low intrinsic dimensionality; pretrained features already span the needed subspace; adaptation is reweighting, not learning new features.
2. Biderman et al. 2024: LoRA underperforms full FT when the target task demands genuinely new capabilities; it forgets less of the base model (useful for multi-task).
3. QLoRA paper finding: applying LoRA to all linear layers (q, k, v, o, gate, up, down) at rank 16 outperformed applying to q/v only at rank 64. Coverage beats rank.
4. You cannot merge a low-rank fp16 update into an NF4 weight without dequantizing first. Options: dequantize → merge → re-quantize (loses some quality), or keep the adapter separate at inference (extra computation per forward pass).
5. Sweep r ∈ {4, 8, 16, 32, 64}, fix all else. Plot macro-F1 vs. r. Flat curve at r=16 → the task's adaptation lives in ≤16 dimensions; the low-rank constraint isn't the bottleneck. Monotone improvement → the constraint is binding and you'd benefit from higher rank or full FT.
</details>

---

### Topic 3: GPU Memory Accounting (16GB Budget)

**Level 1 — State it:** Name the four main consumers of GPU memory during full fine-tuning with AdamW.

**Level 2 — Derive it:** Derive the byte counts for each consumer for Qwen2.5-1.5B in mixed-precision (bf16 weights, fp32 optimizer states). Show the total exceeds 16GB. Then show the QLoRA budget.

**Level 3 — Defend it (5 follow-ups):**

1. Why are optimizer moments kept in fp32 even in mixed-precision training?
2. What does gradient checkpointing actually store, and what does it recompute?
3. When do paged optimizers trigger, and what do they actually do?
4. Activations scale with batch × seq_len × hidden × layers. For your config `[FILL]`, estimate the activation memory.
5. "Could you have fully fine-tuned a 0.5B model on 16GB instead? Would it have been better?" (no memorized answer — reason it live.)

<details>
<summary>🔑 Level 3 check</summary>

1. The second-moment EMA (v_t) involves squaring gradients and dividing by their root. In fp16/bf16, the limited mantissa causes precision loss in the running average, leading to unstable updates — especially for parameters with small gradients. fp32 moments keep the optimization numerically stable.
2. Stores only the inputs to each checkpoint segment (every N transformer blocks, typically N=1). Discards intermediate activations. On the backward pass, recomputes the forward pass for each segment to regenerate the intermediates needed for gradient computation. Trade: ~30% extra compute for ~O(√L) memory (L = layers).
3. Paged optimizers (BitsAndBytes) use CUDA unified memory: when GPU memory is full, optimizer states are automatically paged to CPU RAM. They trigger on memory-pressure events (allocation failures), not on a fixed schedule. This prevents OOM during batch-size spikes or long-sequence steps, but CPU-paged updates are slower. They don't reduce steady-state memory — they handle *spikes*.
4. Rough estimate: per transformer block, the attention score matrix is seq_len² × n_heads × batch × 2 bytes (bf16); the hidden activations are batch × seq_len × d_model × 2 bytes; there are ~4 such tensors per block (input, post-attention, post-FFN, FFN intermediate at 4×d_model). Multiply by number of blocks. Without checkpointing, this often exceeds the weight memory.
5. A 0.5B model has ~3× fewer parameters, so full FT budget: ~2 GB weights + 2 GB gradients + 8 GB optimizer = ~12 GB, leaving ~4 GB for activations. Feasible. Whether it's *better* depends on whether the 0.5B model's representations are as rich as 1.5B's for this task. The QLoRA approach bets that a larger base model's representations are worth the adapter constraint. This is an empirical question — and proposing it as an ablation is the right move.
</details>

---

### Topic 4: Causal-LM-as-Classifier Design

**Level 1 — State it:** Explain why you use the *last* non-padding token's hidden state for classification in a causal LM.

**Level 2 — Derive it:** Compare the classification-head approach vs. the verbalizer approach. When does each win? Describe the padding-side bug and how to prevent it.

**Level 3 — Defend it (5 follow-ups):**

1. You set pad_token = eos_token. When is that *not* safe?
2. Walk through the exact failure mode: right-padding, naive last-position indexing, variable-length batch.
3. "Why not mean-pool over all tokens?" (causal masking makes early tokens partial)
4. "Would a bidirectional encoder be strictly better for this task?"
5. Design the verbalizer alternative for ESCI's 4 classes. What label tokens do you pick and why?

---

### Topic 5: Macro-F1 & ESCI Imbalance

**Level 1 — State it:** Define macro-F1 and explain why it's preferred over accuracy for ESCI.

**Level 2 — Derive it:** State what macro-F1 hides (which class is failing, asymmetric error costs, ordinal structure). Produce the argument for and against focal loss on your task.

**Level 3 — Defend it (5 follow-ups):**

1. Your per-class F1 is `[FILL]`. Which class drags the macro average, and why?
2. "Macro-F1 treats E→I and E→S errors equally. Design a metric that doesn't."
3. "When would micro-F1 be the right choice?"
4. "Is there a fundamental label-noise ceiling on ESCI? How would you estimate it?"
5. "Your model improved on Complement by 5 points but dropped on Exact by 1. Is that a win?"

---

### Topic 6: RoPE Derivation

**Level 1 — State it:** State what RoPE does and why it's applied to Q/K but not V.

**Level 2 — Derive it:** Derive the 2D case from the relative-position inner-product requirement. Extend to d dimensions with the frequency schedule.

**Level 3 — Defend it (5 follow-ups):**

1. Why θᵢ = 10000^(−2i/d)? What does the base 10000 control?
2. What happens at inference positions beyond the training length?
3. Compare RoPE to ALiBi (Press et al. 2021). When does ALiBi win?
4. "Can you make the RoPE frequencies learnable? What would that buy?"
5. "You applied RoPE in an encoder. Most RoPE literature is on decoders. Does anything change?"

---

### Topic 7: Pre-LN vs. Post-LN

**Level 1 — State it:** State the difference in one sentence. Which did you use and why?

**Level 2 — Derive it:** Explain the gradient-flow argument for why Pre-LN trains more stably. What does the residual-stream growth trade-off look like?

**Level 3 — Defend it (5 follow-ups):**

1. "At what depth does Post-LN typically fail without warmup?"
2. "What does the final LayerNorm in a Pre-LN model have to absorb?"
3. "RMSNorm vs. LayerNorm — what's dropped and does it matter?"
4. "DeepNorm (Wang et al. 2022) claims to fix Post-LN for 1000+ layers. How?"
5. "Design an experiment to compare Pre-LN and Post-LN on your QQP task."

---

### Topic 8: Contrastive Loss & Negatives

**Level 1 — State it:** State your loss function from memory. What is the temperature parameter?

**Level 2 — Derive it:** Derive the gradient of InfoNCE w.r.t. the anchor embedding. Show why low temperature concentrates gradient on the hardest negative.

**Level 3 — Defend it (5 follow-ups):**

1. "QQP has ~37% duplicates. What's the false-negative contamination rate in a batch of 64?"
2. "How would you mine hard negatives for QQP specifically?"
3. "What's embedding collapse and how do you detect it during training?"
4. "Compare your contrastive approach to cross-entropy on the binary QQP labels. When does each win?"
5. "If you switched from cosine similarity to dot product in your loss, what changes?"

---

## 🟢 Self-Scoring Protocol

After completing the drills, update PROGRESS.md:
- Mark each topic as `[k]` (knows it), `[d]` (can derive it), or `[D]` (can defend it) based on where you run out of depth.
- Log every gap in the Gap Log: the exact question chain, the depth you reached, and the point where depth ran out.
- There is no score to collect here — the only question is whether you can defend each topic. Aim to reach Level 3 on all 8 before the mock round; nothing is locked if you do not.

Not at Level 3 on all 8 yet? Study your gap log, revisit the refresher at the exact point you stalled, and re-drill. The mock round is there whenever you want it — nothing is gated on it.

---

## 🟢 Mock Round: Architecture Bar Raiser

> ⚔️ **Format:** Simulated 50-minute science-depth round. The "interviewer" drills five levels deep on topics drawn from all 8 core areas. You answer out loud (or written, but out loud is better practice). Score yourself honestly per the rubric below.

### Round Setup

1. Set a 50-minute timer.
2. Deliver your QLoRA pitch first (~4 min), then your RoPE pitch (~4 min). Total ≤ 8 min.
3. The interviewer questions below fire in sequence. Answer each fully before reading the next.
4. After the questions, score yourself.

Record both pitches back-to-back under time pressure, then review the delivery before you face the interrogation:

<RehearsalStudio prompt="Deliver both Session 1 pitches back-to-back under time pressure — QLoRA on ESCI first, then the from-scratch RoPE transformer. ≤ 8 minutes total. Result-first, decision tour, invite the drill; no unfilled or invented numbers." minSeconds="360" maxSeconds="480" rubric="project-pitch" />

### The Interrogation

**Opening — QLoRA project:**

> "Good. You said NF4 is 'information-theoretically optimal.' Optimal in what sense, and when does it stop being optimal?"

*(Expecting: normality assumption, quantile bins, failure under non-normal weight distributions. If you said "it's optimal" without the caveat, the round is already downhill.)*

> **Follow-up 1:** "Walk me through your 16GB memory budget. I want byte counts."

*(Expecting: live derivation — bf16 weights, fp32 optimizer states, gradients, activations. Then the QLoRA budget with NF4 + adapter-only optimizer. End with your observed peak memory `[FILL]` or an honest 'I didn't log it.')*

> **Follow-up 2:** "Your macro-F1 is `[FILL]`. Show me the per-class F1. Which class is failing, and what does that tell you about the model vs. the task?"

*(Expecting: `[FILL: per-class F1]` from your logs. Identification of the weakest class — likely Complement or Substitute. The distinction between model failure (learnable confusion boundary) and task ambiguity (label noise on the S/C boundary). If the slot is QUALITATIVE-ONLY, say so and explain what you'd expect and why.)*

> **Follow-up 3:** "You compared a 1.5B decoder with adapters to a 184M encoder with full fine-tuning. Walk me through every confound."

*(Expecting: architecture (causal vs. bidirectional), parameter count, adaptation method (LoRA vs. full), pretraining data/objective, sequence length, hyperparameter tuning budget. Then: how to design the controlled version. Frame yours as an engineering comparison, not a scientific ablation.)*

> **Follow-up 4:** "Fair. Now switch to the other project. Derive RoPE from the requirement."

*(Expecting: the full 2D derivation → d-dimensional extension → frequency schedule → why q/k only. If you stall on the orthogonality step, that is a real gap.)*

**Transition — RoPE project:**

> **Follow-up 5:** "You used Pre-LN. If I gave you 10× the compute budget, would you switch to Post-LN?"

*(Expecting: acknowledgment of the Post-LN performance ceiling advantage, discussion of when the warmup cost is worth paying, the depth at which Post-LN becomes unstable. The honest answer depends on your model's depth — `[FILL: layers]`.)*

> **Follow-up 6:** "Write your contrastive loss on the board. What's τ doing, and what happens when you halve it?"

*(Expecting: exact formulation from memory, τ = `[FILL]`, explanation that halving τ sharpens the softmax → concentrates gradient on hardest negatives → amplifies false-negative damage → potential collapse if taken too far.)*

> **Follow-up 7:** "QQP has ~37% duplicates. In a batch of 64, how many of your 'negatives' are actually positives? And what does that do to your gradients?"

*(Expecting: quantitative estimate of false-negative rate, explanation that false negatives produce repulsive gradients on embeddings that should attract, degradation of the embedding space. Mitigation options: label-aware negative filtering, accepting the noise with an honesty caveat.)*

> **Follow-up 8:** "Last one. Why build from scratch at all? A pretrained sentence-transformer would beat your model. Justify the project's existence."

*(Expecting: the mastery-demonstration frame, not a SOTA claim. The ability to defend every component in this conversation *is* the project's value. Acknowledge the performance gap honestly and state what the project proves.)*

### Scoring Rubric

| Question | Bar-clearing answer | Floor indicators |
|---|---|---|
| NF4 optimality caveat | States assumption, explains failure modes | Says "optimal" without qualification |
| Memory derivation | Live arithmetic, reaches observed peak | Can't separate optimizer states from weights |
| Per-class F1 analysis | Quotes numbers (or honest qualitative), identifies weakest class | Can't say which class is failing |
| Confound analysis | Names ≥ 4 confounds, proposes controlled design | Lists only "different model sizes" |
| RoPE derivation | Complete 2D → d-dim, orthogonality step clean | Stalls at "it rotates the vectors" |
| Pre-LN trade-off | Acknowledges ceiling, conditions on depth and budget | "Pre-LN is just better" |
| Contrastive loss from memory | Writes exact formula, explains τ sensitivity | Can't reproduce the formula |
| False negatives | Quantitative estimate, gradient explanation | "I don't think false negatives matter" |
| From-scratch justification | Mastery frame, honest gap, proposes comparison | "Mine is better because it's from scratch" |

**What a clear round looks like:** all 9 rubric items answered at the bar-clearing standard, with zero hand-waves and zero unsupported numbers.

### After the Mock Round

1. **Update PROGRESS.md:**
   - Mark all 8 core topics with their level: `[k]` (knows it), `[d]` (can derive it), or `[D]` (can defend it).
   - Log each drill: depth survived, where you run out of depth, and the exact question.
   - Record gaps in the Gap Log with exact question and depth.

2. **If it went well (Level 3 on all topics):** update the Consistency Tracker and move to Session 2 when you're ready.

3. **If it didn't:** study the gap log, re-read the refresher where your depth ran out, re-drill that topic's Level 3 chain, and run the mock round again whenever you like — each attempt restarts from question one.

---

## 🟢 Summary

- The depth drills are self-scored: Level 1 = states it correctly, Level 2 = derives it, Level 3 = defends five follow-ups. Level 3 is the hiring bar.
- The mock round simulates a 50-minute science-depth round across both projects. It probes all 8 core topics at five-plus levels.
- Every number must trace to a run log. Every gap goes in the log. Every hand-wave is a gap worth logging.
- Next: Session 2 — Project Deep-Dives: Systems & Evaluation (RAG + Calibration).

**The enemy is not the assessment. The enemy is an empty gap log.**
