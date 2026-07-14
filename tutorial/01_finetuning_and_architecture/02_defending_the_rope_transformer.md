# Lesson 2 — Defending the From-Scratch RoPE Transformer

| | |
|---|---|
| **Prerequisites** | Session 1 README; your ROPE transformer run logs open in another window |
| **Reading time** | ~25 min + drills |
| **Domain tag** | Transformers / Embeddings / Contrastive Learning |
| **Resume line under fire** | "Built a Transformer encoder from scratch with Rotary Position Embeddings, Pre-LayerNorm, and Multi-Head Attention; trained with a custom contrastive loss on Quora Question Pairs" |

**🔬 Interactive companion — *Architecture Unit Tests* (CPU-only, runs instantly).** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/01_finetuning_and_architecture/02_architecture_unit_tests.ipynb)

> [!NOTE]
> The badge above is an image VS Code's preview can't load or click. **Copy this URL into a real browser (Chrome/Edge) signed into Google:**
> `https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/01_finetuning_and_architecture/02_architecture_unit_tests.ipynb`

> 📍 **Roadmap:** 🟢 sections rebuild the defense-level intuition; 🔷 sections carry the derivations and the model answers. Same rule as Lesson 1: every number comes from your logs. `[FILL]` or silence — never an invented figure.

## 🟢 Learning Objectives

- Derive Rotary Position Embeddings from the relative-position inner-product requirement, without notes.
- Explain every architectural choice in your from-scratch transformer (Pre-LN, head count, d_model) and what each costs you.
- Answer the six hardest follow-ups on this project at Applied Scientist depth.
- Identify which claims in your pitch are backed by run logs and which must stay qualitative.

---

## 🟢 The Interrogation Logic of This Bullet

This bullet invites a different chain than QLoRA. There, the interrogation is "did you understand the tool you used?" Here, it is "did you understand the thing you built from nothing?" Building from scratch means you chose *every* component — and every choice is attackable because you had alternatives. The interviewer's opening move is almost always: "Why build from scratch instead of fine-tuning a pretrained model?" Your answer sets the frame for the rest of the round.

The honest frame: this is a learning project that demonstrates mastery of transformer internals, not an engineering claim that from-scratch beats pretrained. If you present it as a scientific contribution, the follow-ups will be brutal and the comparison unfavorable. If you present it as demonstrated understanding with a working system, the follow-ups become a tour of that understanding — which is exactly your strength.

---

## 🔷 Peer-Level Refresher 1: RoPE — From Requirement to Rotation

Skip the "RoPE encodes position" framing. Start from the requirement and derive the mechanism.

**The requirement.** We want the inner product between query q at position m and key k at position n to be a function of their *relative* distance (m − n), not their absolute positions. Formally: ⟨f(q, m), f(k, n)⟩ = g(q, k, m − n) for some encoding function f and some resulting function g.

**Why rotations satisfy this.** Consider 2D vectors. If f(x, m) = R(mθ)·x, where R(mθ) is a 2×2 rotation matrix by angle mθ, then:

⟨f(q, m), f(k, n)⟩ = qᵀR(mθ)ᵀR(nθ)k = qᵀR((n − m)θ)k

The inner product depends only on (m − n) — the relative distance. Rotation matrices are orthogonal, so R(a)ᵀR(b) = R(b − a). The requirement is satisfied exactly.

**Extension to d dimensions.** Pair the d_head dimensions into d/2 pairs, each pair receiving its own rotation frequency θᵢ. The standard frequency schedule is θᵢ = 10000^(−2i/d), placing lower frequencies on later dimensions. This creates a spectrum: early pairs rotate fast (high-resolution local position), late pairs rotate slowly (coarse long-range position). The concatenation of all rotated pairs is the RoPE encoding.

**Applied to q and k only, not v.** Values carry content, not positional relevance; applying rotation to v would distort the content signal that gets aggregated. Position should influence *which* values to attend to (via the attention score), not *what* gets read once attended.

**Extrapolation behavior.** RoPE has no trainable position embedding table — positions are generated on the fly from the rotation formula. In principle this allows extrapolation beyond training length. In practice, the model has never seen the attention patterns that arise at unseen lengths, so quality degrades. NTK-aware scaling and YaRN (Peng et al. 2023, *YaRN: Efficient Context Window Extension of Large Language Models*, https://arxiv.org/abs/2309.00071) address this by rescaling the frequency spectrum, not by adding parameters.

Citation: Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* (https://arxiv.org/abs/2104.09864).

---

## 🔷 Peer-Level Refresher 2: Pre-LayerNorm vs. Post-LayerNorm

The difference is one line of code and a large gradient-flow consequence.

**Post-LN (original Vaswani).** The residual path is: x → sublayer(x) → x + sublayer(x) → LayerNorm. The normalization sits *after* the residual addition. Problem: gradients flowing through the residual stream pass through every LayerNorm, each of which rescales, so the effective gradient magnitude can shrink or explode across depth. This makes deep Post-LN models require careful warmup and sometimes fail to train at all beyond ~12 layers.

**Pre-LN.** The residual path is: x → LayerNorm(x) → sublayer(LayerNorm(x)) → x + sublayer(LayerNorm(x)). The normalization sits *inside* the sublayer branch. The residual stream itself is a clean addition — gradients from the loss flow back through the sum without passing through any normalization. Xiong et al. (2020), *On Layer Normalization in the Transformer Architecture* (https://arxiv.org/abs/2002.04745) show this produces well-behaved gradients at initialization, eliminating the need for learning-rate warmup in most cases.

**The trade-off nobody mentions.** Pre-LN makes training easier but can produce slightly worse final performance than a well-tuned Post-LN model of the same depth. The residual stream grows in magnitude across layers (each sublayer *adds* to it without renormalization), which a final LayerNorm before the output head must absorb. For your from-scratch model at `[FILL: number of layers]` layers, this trade-off is mild — Pre-LN was the pragmatic choice for trainability, and owning that reasoning is the right answer.

**Know your own architecture cold.** `[FILL: layers / heads / d_model / d_head / total params]`. If you cannot reproduce these from memory, the round stalls here.

---

## 🔷 Peer-Level Refresher 3: Multi-Head Attention — The Shapes and Why

You built this from a blank editor. Every matrix dimension should be recitable.

**Projection shapes.** For input X ∈ ℝ^{seq × d_model}, each head i has W_Q^i, W_K^i, W_V^i ∈ ℝ^{d_model × d_head} where d_head = d_model / n_heads. Queries Q_i = X·W_Q^i, keys K_i = X·W_K^i, values V_i = X·W_V^i.

**Scaled dot-product attention.** Attention(Q, K, V) = softmax(QKᵀ / √d_head)·V. The 1/√d_head scaling prevents the dot products from growing with dimension, which would push softmax into saturation (near-one-hot), producing near-zero gradients. Derivation: if q and k entries are iid with variance σ², their dot product has variance d_head·σ⁴ (or d_head·σ² if σ²=1). Dividing by √d_head restores unit variance, keeping softmax in its informative range.

**Output projection.** The concatenated heads [head_1; ...; head_h] ∈ ℝ^{seq × d_model} pass through W_O ∈ ℝ^{d_model × d_model}. This projection lets the model mix information across heads — without it, each head's subspace is isolated.

**What heads learn.** Voita et al. (2019), *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, and the Rest Can Be Pruned* (https://arxiv.org/abs/1905.09418) show that a small fraction of heads carry most of the function: positional heads, syntactic heads, rare-word heads. Most heads are prunable. For your `[FILL: n_heads]`-head model on QQP, you likely see a similar pattern — but you'd need attention-weight visualization to prove it, so present it as a hypothesis, not a claim, unless you ran that analysis.

---

## 🔷 Peer-Level Refresher 4: Your Contrastive Loss — Formulation, Temperature, and Failure Modes

This is *your* custom loss. The interviewer expects you to write it on the board from memory.

**General form.** For a batch of N (anchor, positive) pairs, each pair produces embeddings (z_a, z_p). The 2N embeddings create 2N(2N−1)/2 possible pairs. The contrastive objective pulls positives together and pushes negatives apart. Specific formulations differ:

- **InfoNCE / NT-Xent** (symmetric): For anchor i with positive j, loss = −log( exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ). Temperature τ controls sharpness: low τ makes the loss focus on hard negatives, high τ spreads the gradient across all negatives.
- **Triplet loss**: max(0, sim(a, neg) − sim(a, pos) + margin). Simpler but only uses one negative per anchor.

**Your formulation.** `[FILL: exact loss function you implemented, including temperature/margin values and whether you used cosine similarity or dot product]`. State this from memory — it is your code.

**In-batch negatives and false negatives on QQP.** QQP has ~37% positive pairs. In a batch of B pairs, the 2B−2 negatives per anchor include *all other questions* — but some of those "negatives" are actually duplicates (different pair, same meaning). False negatives push apart embeddings that should be close. At QQP's positive rate, a batch of 64 has ~23 true positive pairs on average, so the false-negative contamination is material. Mitigation: filter known duplicates from the negative set (requires the label matrix), or accept the noise and acknowledge it degrades the embedding space — that honesty is worth more than pretending you handled it.

**Embedding collapse.** The degenerate solution where the encoder maps everything to the same point (or a narrow cone) achieves low loss trivially — all similarities equal, no useful discrimination. Symptoms: loss plateaus early, downstream task performance is random, cosine similarities cluster near 1.0. Prevention: large batch sizes (more diverse negatives), batch normalization on embeddings, temperature tuning (too-low τ accelerates collapse), and hard-negative mining (forces the model to discriminate similar pairs).

---

## 🟢 Concept Check

What property of rotation matrices makes RoPE encode *relative* position in the attention score?

* [ ] Rotation matrices are symmetric
* [x] Rotation matrices are orthogonal, so R(a)ᵀR(b) = R(b − a), making the inner product depend only on the position difference
* [ ] Rotation matrices have unit determinant, preserving vector norms
* [ ] Rotation matrices commute with the softmax function

Why is Pre-LayerNorm easier to train than Post-LayerNorm for deep transformers?

* [ ] Pre-LN uses fewer parameters per layer
* [ ] Pre-LN normalizes gradients during backpropagation
* [x] Pre-LN keeps the residual stream as a clean addition — gradients flow through the sum without passing through normalization layers, avoiding the rescaling that causes gradient pathologies in Post-LN
* [ ] Pre-LN eliminates the need for the output projection matrix

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Q1: option 2.** Orthogonality is the key property. When both q and k are rotated by their respective position angles, the transpose-product of the rotation matrices collapses to a single rotation by the *difference* angle. Unit determinant (option 3) preserves norms, which is nice but not the relative-position mechanism. Symmetry (option 1) is false for rotation matrices in general.

**Q2: option 3.** The residual stream in Pre-LN accumulates sublayer outputs via plain addition. In Post-LN, every layer's residual passes through LayerNorm, which rescales the gradient. At 24+ layers, this repeated rescaling makes the gradient signal degrade — not vanish in the classical sense, but become poorly conditioned. Pre-LN's clean residual path avoids this entirely.
</details>

---

## 🔷 Peer-Level Refresher 5: Quora Question Pairs — Dataset Pitfalls

**Label noise.** QQP labels are crowd-sourced binary (duplicate / not-duplicate). Agreement rates are imperfect: borderline pairs ("How do I learn Python?" vs. "What's the best way to learn Python?") have genuine ambiguity. Your model's error floor is bounded by annotator disagreement — if inter-annotator agreement is ~90%, no model should claim 95%+ without an explanation.

**Transitivity leakage.** If (A, B) is a duplicate and (B, C) is a duplicate, then (A, C) should be too. Random train/test splits can leak this: A–B in train, A–C in test, and the model memorizes the equivalence through B. Proper evaluation requires split-by-question-cluster, not split-by-pair. `[FILL: did you check this / which split strategy you used]`.

**Length artifacts.** Short question pairs are easier to classify (less ambiguity). A model that learns a length heuristic scores well on average but fails on long, nuanced pairs — which are the ones that matter for a real application. If you have per-length-bucket accuracy, bring it; if not, flag it as an analysis you'd add.

---

## 🔷 The Six Hardest Follow-Ups (With Model Answers)

### 🔷 Follow-up 1: "Why build a transformer from scratch instead of fine-tuning a pretrained sentence-transformer? What did you gain that a BERT baseline wouldn't give you?"

**Why they ask:** tests whether the project has a defensible purpose beyond "I wanted to learn."

**Model answer:** "The purpose is demonstrated mastery, not a performance claim. A pretrained sentence-transformer — say all-MiniLM-L6-v2 — would almost certainly beat my from-scratch model on QQP, because it starts with 6 layers of distilled knowledge on 1B+ sentence pairs. What I gained is the ability to defend every component in this conversation: I can derive RoPE from the relative-position requirement, I can explain why Pre-LN over Post-LN, I can write the contrastive loss from memory with its temperature sensitivity. A fine-tuned model gives better numbers; a from-scratch model gives better *understanding*, which is what this round is testing. My model's QQP performance was `[FILL: metric + value]` — I'd frame the project as 'built and trained a working transformer, then used it to study contrastive learning dynamics,' not as a SOTA claim."

**The chain behind it:** "What baseline did you compare against?" → "How many parameters in yours vs. MiniLM?" → "So you spent GPU hours to underperform — what's the ROI argument for the team?" (answer: this is a research/learning project; the ROI is the engineer's depth, which is what you're evaluating right now) → "If you had to make it production-quality, what's the first change?" (pretrain on a large corpus before contrastive fine-tuning — self-supervised pretraining is the entire gap).

### 🔷 Follow-up 2: "Derive RoPE for me. Start from what you want the attention score to satisfy."

**Why they ask:** the derivation is the single cleanest test of whether you understand the math or memorized the formula.

**Model answer:** "I want ⟨f(q, m), f(k, n)⟩ = g(q, k, m−n) — the dot product depends on relative position only. Consider 2D first: let f(x, m) = R(mθ)x, a rotation by mθ. Then ⟨R(mθ)q, R(nθ)k⟩ = qᵀR(mθ)ᵀR(nθ)k = qᵀR((n−m)θ)k, since rotation matrices are orthogonal. The inner product is now a function of (n−m) — done. For d-dimensional heads, I pair dimensions into d/2 two-dimensional subspaces, each with its own frequency θᵢ = 10000^(−2i/d). Low-index pairs get high frequencies for fine-grained local position; high-index pairs get low frequencies for coarse long-range position. The encoding applies element-wise as: [q₂ᵢ, q₂ᵢ₊₁] → R(m·θᵢ)[q₂ᵢ, q₂ᵢ₊₁] for each pair i. Applied to q and k only — values carry content, and rotating them would distort what gets read."

**The chain:** why θᵢ = 10000^(−2i/d) specifically (empirical default from the RoFormer paper; the base 10000 sets the wavelength range; larger bases extend the range for longer contexts) → "what happens at positions beyond training length?" (unseen rotation angles; attention patterns the model never trained on; NTK-aware scaling adjusts the base to compress the frequency spectrum) → "how does this compare to learned absolute position embeddings?" (learned embeddings have a fixed vocabulary of positions; RoPE generates any position, but the model still hasn't *learned* the attention patterns for unseen positions — the advantage is smoothness, not magic).

### 🔷 Follow-up 3: "Write your contrastive loss on the board. What's the temperature doing, and what happens if you set it wrong?"

**Why they ask:** custom loss is the strongest claim in this bullet — and the easiest to expose if it's copy-pasted.

**Model answer:** `[FILL: write your exact loss here — the formulation, the similarity function (cosine vs. dot), and the temperature/margin value]`. "Temperature τ controls the sharpness of the softmax over similarities. Low τ (say 0.05) concentrates gradient on the hardest negatives — the pairs closest to the anchor that aren't positives. This accelerates learning but amplifies the impact of false negatives: a mislabeled negative that's actually a duplicate gets a huge repulsive gradient. High τ (say 1.0) spreads gradient uniformly across all negatives — safer but slower, and the model may never learn fine-grained distinctions. I used τ = `[FILL: your value]`. The optimal range for sentence-level tasks is empirically 0.05–0.1 (Chen et al. 2020, *A Simple Framework for Contrastive Learning of Visual Representations*, https://arxiv.org/abs/2002.05709 — the finding transfers to NLP). If τ is too low, you risk embedding collapse; if too high, the loss landscape is too flat for SGD to navigate efficiently."

**The chain:** "How did you choose τ?" → "Did you sweep it? What was the sensitivity?" `[FILL: if you did a sweep]` → "What are your false negatives doing to the gradient?" → "How would you mine hard negatives for QQP specifically?" (by question-cluster overlap; or use the current model's top-k nearest neighbors as candidate hard negatives, re-mine every N epochs).

### 🔷 Follow-up 4: "Pre-LN makes training easy. What does it cost you, and how do you know?"

**Why they ask:** choosing the easier option is fine; not knowing the trade-off is not.

**Model answer:** "Two costs. First, the residual stream grows in magnitude across depth — each sublayer adds to it without renormalization. At `[FILL: your layer count]` layers this is manageable; at 48+ layers, the final LayerNorm before the output head has to compress a wide dynamic range, which can lose precision. Second, some studies (Xiong et al. 2020) show that a *well-tuned* Post-LN model can slightly outperform Pre-LN of the same depth — the repeated normalization acts as an implicit regularizer that Pre-LN lacks. For my scale and training budget, the trade-off is clear: Pre-LN trained stably without warmup, and the potential performance gap is within noise at my parameter count. I'd switch to Post-LN only with a larger model and a training budget that could absorb warmup sweeps."

**The chain:** "Did you use a final LayerNorm before the output head?" `[FILL]` → "What about RMSNorm instead of LayerNorm?" (RMSNorm drops the mean-centering, keeping only the variance normalization; ~5-10% faster per layer; functionally similar when mean is near zero, which it usually is for residual streams) → "What training pathology would you see if you accidentally used Post-LN without warmup?" (loss spikes or NaN in early steps; the gradient norms for early layers are much larger than for late layers, so a single learning rate either under-updates late layers or over-updates early ones).

### 🔷 Follow-up 5: "You used Multi-Head Attention. Why not Multi-Query or Grouped-Query Attention?"

**Why they ask:** tests awareness of the design space beyond the choice you made — and whether you chose deliberately.

**Model answer:** "MQA (Shazeer 2019, *Fast Transformer Decoding: One Write-Head is All You Need*, https://arxiv.org/abs/1911.02150) shares a single K and V head across all query heads; GQA (Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, https://arxiv.org/abs/2305.13245) groups query heads into g groups, each sharing one K/V head. Both reduce the KV-cache size at inference — MQA by n_heads×, GQA by n_heads/g ×. The quality cost is modest: GQA at g=2 or g=4 matches MHA on most benchmarks while halving or quartering the KV cache. I used standard MHA because (a) my model is small enough that KV-cache size is irrelevant — I'm not serving it — and (b) for a learning project, implementing the full MHA makes every head independently inspectable. If I were building this for production inference, GQA would be the default choice."

**The chain:** "What's the KV-cache bottleneck you'd hit at inference?" → "Can you convert an MHA model to GQA post-training?" (yes — Ainslie et al. show mean-pooling the K/V heads within a group, then brief continued training, recovers most of the quality; you'd need the checkpoint and a few thousand steps) → "How does the parameter count change?" (K/V projections shrink by the grouping factor; Q and O projections unchanged).

### 🔷 Follow-up 6: "Walk me through a training step end to end — forward, loss, backward, update — for one batch of QQP pairs through your model."

**Why they ask:** this is the "do you actually own this code?" question. It's not about theory — it's about the specific operations in the specific order in your implementation.

**Model answer:** "A batch of `[FILL: batch size]` (question₁, question₂) pairs, each with a binary label. Both questions are tokenized to max length `[FILL: max_len]` and padded. Each question passes through the same encoder: token embedding → `[FILL: positional embedding mechanism]` → `[FILL: N]` transformer blocks (each: Pre-LayerNorm → MHA with RoPE on Q/K → residual add → Pre-LayerNorm → FFN → residual add) → pooling to get a fixed-size embedding. Pooling strategy: `[FILL: CLS token / mean pooling / last token]`. Both embeddings are L2-normalized (for cosine similarity). The contrastive loss is computed over the batch: `[FILL: your loss function]` with in-batch negatives. Backward: PyTorch autograd computes gradients through the loss, the similarity computation, both encoder passes (shared weights), through every transformer block. Optimizer: `[FILL: Adam/AdamW]` with learning rate `[FILL: LR]` and schedule `[FILL: schedule]`. Gradient clipping: `[FILL: if used, max norm]`."

**The chain:** "What's your pooling strategy and why?" → "Mean pool over what — all tokens including padding?" (mask out padding before mean) → "What happens if you forget the mask?" (padding tokens contribute noise; the mean is pulled toward the pad embedding; short sequences are disproportionately diluted) → "Show me the gradient path from the loss to a single RoPE frequency parameter" (trick question — RoPE frequencies are not learned parameters in the standard formulation; they're fixed by the formula; if your implementation makes them learnable, say so and explain why).

---

## 🔷 Hands-On Lab: Build Your RoPE Transformer Pitch Menu

30 minutes, produces the artifact you'll use in the boss fight and the real loop.

1. From your run logs, fill every ROPE row of the Metric Vault in [PROGRESS.md](../../PROGRESS.md). Empty slots get marked `QUALITATIVE-ONLY`.
2. Write your 4-minute project pitch (see [round playbook](../../playbooks/round_playbook.md), beat structure: result → framing → decision tour → invite the drill).
   - **Result first**: "I built a transformer encoder from scratch with RoPE and Pre-LN, trained with contrastive learning on QQP, achieving `[FILL: metric]` on `[FILL: eval metric name]`."
   - **Frame as demonstrated mastery, not SOTA claim**: "The goal was to own every component — position encoding, normalization, loss function — not to outperform pretrained models."
   - **Decision tour**: RoPE over learned embeddings (relative position, extrapolation), Pre-LN over Post-LN (trainability), contrastive loss over cross-entropy (embedding space quality).
   - **Invite**: stop at ~4 minutes.
3. Underline every technical term in the pitch. For each, self-assess Bronze/Silver/Gold honestly. **Delete or downgrade every hook that isn't Silver+.**
4. Say the pitch out loud, timed. Twice.

Expected output: a pitch of ≤ 4 minutes containing zero unfilled numbers and zero Bronze hooks.

## 🟢 Concept Check

In a contrastive learning batch of 64 QQP pairs, approximately how many in-batch negatives does each anchor have, and what fraction are likely false negatives?

* [ ] 64 negatives, ~0% false negatives
* [ ] 127 negatives, ~5% false negatives
* [x] 126 negatives (2×64 − 2, excluding self and positive), and with QQP's ~37% positive rate, a meaningful fraction of those 126 are semantically duplicate questions mislabeled as negatives
* [ ] 63 negatives, ~50% false negatives

You notice your contrastive model's cosine similarities are all clustered near 0.98 for both positive and negative pairs after 5 epochs. What is happening?

* [ ] The temperature is too high, spreading gradients too thin
* [ ] The learning rate is too low to move the embeddings
* [x] Embedding collapse — the encoder is mapping all inputs to near-identical points, producing uniformly high cosine similarity; the loss is low but the representations are useless
* [ ] The model has converged to the optimal embedding space

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Q1: option 3.** With N=64 pairs producing 128 total embeddings, each anchor has 126 potential negatives (all embeddings except itself and its known positive). QQP's ~37% duplicate rate means a substantial number of those 126 "negatives" are actually semantically equivalent questions paired with different partners in the batch. This false-negative contamination pushes apart embeddings that should be close, degrading the embedding space. The exact count depends on cluster structure within the batch.

**Q2: option 3.** Embedding collapse is the degenerate solution to contrastive objectives: when all embeddings converge to a single point (or narrow cone), all cosine similarities approach 1.0, and the softmax in the loss produces near-uniform attention — low loss, zero discriminative power. Diagnosis: check embedding variance across the batch. Fixes: increase batch size (more diverse negatives), lower temperature (harder negative pressure, but watch false negatives), add batch normalization on embeddings, or use asymmetric architectures (BYOL-style, though that's a different approach).
</details>

---

## 🟢 Summary & Resources

- RoPE = rotation matrices applied to q/k pairs, derived from the requirement that attention scores depend on relative position; each dimension pair gets a different frequency, creating a multi-resolution position spectrum.
- Pre-LN trades a small potential performance ceiling for stable, warmup-free training via a clean residual path; the right choice at your model's scale.
- Your contrastive loss formulation, temperature, and false-negative handling are *your code* — reproduce them from memory or the round stalls.
- The from-scratch project is a mastery demonstration, not a SOTA claim. Frame it that way and own the comparison gap honestly.
- QQP's label noise and transitivity leakage bound achievable performance — and knowing that bound is Gold-tier.

**References:** Su et al. 2021 (RoPE/RoFormer, arXiv:2104.09864) · Xiong et al. 2020 (Pre-LN, arXiv:2002.04745) · Voita et al. 2019 (MHA head analysis, arXiv:1905.09418) · Chen et al. 2020 (SimCLR contrastive framework, arXiv:2002.05709) · Shazeer 2019 (MQA, arXiv:1911.02150) · Ainslie et al. 2023 (GQA, arXiv:2305.13245) · Peng et al. 2023 (YaRN, arXiv:2309.00071)

**Next:** [Lesson 3 — Tiered Challenges & Boss Fight](03_tiered_challenges_and_boss_fight.md)
