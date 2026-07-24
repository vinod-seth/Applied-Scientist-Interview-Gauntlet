# Session 1 — Chapter Quiz Bank

| | |
|---|---|
| **Prerequisites** | Lessons 1–3 |
| **Time** | ~40 min |
| **Rules** | Closed notes. Answer out loud where the question asks for reasoning — the interview is spoken, not written. |

This bank brings Session 1 to interview-drill volume: 12 quiz questions plus 2 reflection prompts, on top of the 9 concept-check questions inside lessons 1–2 and the assessment chains. No mastery credit for completing the quiz — mastery comes from the drills and assessment. Use this bank to find floors cheaply.

---

## 📝 Chapter Quiz

**Q1.** In NF4, what does the per-block absmax constant accomplish?

* [ ] It stores the block's mean for de-biasing
* [x] It rescales each 64-value block into the quantizer's input range, so one outlier only distorts its own block
* [ ] It replaces the need for double quantization
* [ ] It selects between int4 and FP4 per block

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** NF4 levels are fixed quantiles of N(0,1) normalized to [−1, 1]; real blocks are scaled by their absolute maximum to fit that range. Small blocks localize outlier damage at the cost of one constant per block — which is exactly the overhead double quantization then compresses.
</details>

**Q2.** Rank the memory consumers of full mixed-precision AdamW fine-tuning of a 1.5B model, largest first.

* [x] fp32 optimizer moments > fp32 master weights > bf16 weights ≈ bf16 gradients
* [ ] bf16 weights > gradients > optimizer moments > master weights
* [ ] Activations are always larger than all parameter states combined
* [ ] All four are equal by construction

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 1.** Two fp32 moments = 8 bytes/param (~12.3 GB), master weights = 4 bytes/param (~6.2 GB), bf16 weights and gradients = 2 bytes/param each (~3.1 GB each). Activations vary with batch/sequence config, so option 3 is not "always" true — and knowing it *depends* is the interview-grade detail.
</details>

**Q3.** *Debugging.* Your QLoRA run trains, but every checkpoint predicts the majority class at eval while train loss decreases. Which two checks do you run first?

* [x] Verify the classification head indexes the last *non-padding* token, and compare train-time vs. eval-time padding/batching behavior
* [ ] Lower the LoRA rank and retrain
* [ ] Switch NF4 to int4 to rule out quantization
* [ ] Increase epochs — the model needs more time

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 1.** A head reading pad positions sees near-constant input → constant (majority) predictions, while the training loss can still fall if train batches have more homogeneous lengths. Rank, quantization format, and epochs all change *capacity or precision*, not the mechanism that produces a constant prediction.
</details>

**Q4.** Why does the α/r factor exist in LoRA's ΔW = (α/r)·BA?

* [ ] It normalizes B's zero initialization
* [ ] It keeps ΔW orthogonal to W
* [x] It makes the update's effective scale roughly invariant to rank, so learning rate doesn't need retuning when r changes
* [ ] It enforces the <2% trainable-parameter budget

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** As r grows, the sum BA accumulates more rank-1 contributions; dividing by r compensates, so sweeping rank doesn't silently sweep the update magnitude too. Without it, every rank experiment confounds two variables — a clean experimental-design point worth saying in the room.
</details>

**Q5.** *Scenario.* Recruiter says the team runs production product-search ranking. Which change to your QLoRA pitch is correct?

* [ ] Remove the DeBERTa baseline — encoders are outdated
* [x] Prepare the ⚠️ JD-DEPENDENT layer: latency/cost of serving a 1.5B adapter model vs. a distilled cross-encoder, and where you'd measure online vs. offline metric gaps
* [ ] Claim your model is production-ready as-is
* [ ] Swap macro-F1 for accuracy since production cares about traffic-weighted quality

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** A production-search team flips serving considerations into scope. Option 4 is half-clever bait: traffic-weighted metrics matter online, but that argues for *adding* micro/weighted views and online metrics (CTR, purchase rate), not abandoning the imbalance-aware offline metric.
</details>

**Q6.** Which statement about gradient flow in QLoRA is exactly right?

* [ ] Gradients are quantized to 4-bit to match the weights
* [ ] A straight-through estimator approximates the quantizer's derivative
* [x] Dequantized bf16 base weights act as constants in backprop; gradients exist only for LoRA parameters (and flow through frozen layers via ∂L/∂x)
* [ ] The base model receives gradients but skips optimizer updates

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** No parameter gradient is ever computed for the frozen base — that's also why option 4 is wrong (skipping updates would still waste memory materializing those gradients; QLoRA never creates them).
</details>

**Q7.** *Code output prediction.* In your RoPE unit test you shift all token positions by s = 100 and recompute attention logits on the same sequence. What must happen?

* [x] Logits are numerically unchanged — RoPE scores depend only on position offsets
* [ ] Logits shift by a constant additive factor
* [ ] Logits are unchanged only for the first attention layer
* [ ] Logits change, because absolute position enters through the value vectors

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 1.** ⟨R((m+s)θ)q, R((n+s)θ)k⟩ = qᵀR((n−m)θ)k — the shift cancels in every layer. This invariance is a five-line test that catches most RoPE implementation bugs (wrong pairing of dimensions, applying rotation to v, off-by-one in position ids).
</details>

**Q8.** Why does Post-LN require learning-rate warmup while Pre-LN typically doesn't?

* [ ] Post-LN has more parameters to warm up
* [x] With LN on the residual path, early-layer gradients at init are badly scaled relative to late layers; warmup keeps updates small until the layers equilibrate
* [ ] Pre-LN normalizes gradients directly
* [ ] Warmup is only needed with Adam, and Post-LN models historically used SGD

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Xiong et al. (2020) showed the expected gradient magnitude in Post-LN depends strongly on layer index at initialization; a full-size learning rate immediately destabilizes the mis-scaled layers. Pre-LN's untouched identity path keeps gradient scale roughly uniform across depth, removing the need for that protective phase.
</details>

**Q9.** Halving the temperature τ in InfoNCE does what to the gradient distribution over negatives?

* [ ] Spreads it more uniformly across all negatives
* [ ] Nothing — τ only scales the loss value
* [x] Concentrates it on the highest-similarity (hardest) negatives, which also amplifies damage from false negatives
* [ ] Transfers gradient from negatives to the positive term only

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** Each negative's repulsive gradient is weighted by its softmax probability exp(sim/τ)/Z; lower τ sharpens that distribution toward the top-similarity items. On QQP, the "hardest negatives" are disproportionately *unlabeled true duplicates* — so aggressive τ actively corrupts the embedding space. That coupling is the Gold-tier observation.
</details>

**Q10.** *Debugging.* Contrastive loss drops smoothly, but retrieval of held-out duplicates is at chance. The single most diagnostic quick check is:

* [ ] Re-run with a bigger batch
* [ ] Plot the loss on a log scale
* [x] Compute the variance (or effective rank) of a batch of embeddings — near-zero spread means collapse, which produces low loss and useless representations
* [ ] Rebuild the FAISS index with HNSW

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** Collapse is the one failure that *rewards* the training objective while destroying downstream utility, so it exactly matches the symptom pattern. Batch size and index type change quality at the margin; neither explains chance-level retrieval with a healthy loss curve.
</details>

**Q11.** Your QQP split was random over *pairs*. The most serious resulting claim-invalidator is:

* [ ] Class imbalance between splits
* [x] The same question appears in train and test pairs, so the encoder can score well by memorizing question identity rather than learning paraphrase structure
* [ ] Test pairs are shorter on average
* [ ] FAISS recall differs between splits

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** QQP questions form a duplicate graph; pair-level splitting leaks nodes (and, via transitivity, even labels) across the boundary. The fix is assigning whole connected components to one split. Volunteering this *against your own project* is exactly the self-skepticism the science round scores.
</details>

**Q12.** Which comparison claim about your two baselines is defensible as stated?

* [ ] "QLoRA matches full fine-tuning in general"
* [ ] "My from-scratch encoder shows contrastive learning beats cross-entropy on QQP"
* [x] "Under a fixed 16GB budget, QLoRA on a 1.5B decoder reached `[FILL]` vs. `[FILL]` for a fully fine-tuned DeBERTa-v3 — an engineering comparison with acknowledged confounds, not a controlled ablation"
* [ ] "RoPE improves accuracy over learned positional embeddings on QQP" (untested)

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** It scopes the claim to the actual experimental conditions, quotes only logged numbers, and names its own limits. Options 1 and 2 overgeneralize from one setup; option 4 asserts an ablation that was never run — all three are the kind of sentence a Bar Raiser dismantles in one follow-up.
</details>

---

## 🟢 Reflection Prompts (open-ended)

**R1.** Write the *single most honest weakness* of each project in one sentence each — the sentence you would say if asked "what's wrong with this work?" Then check: does your pitch already surface it, or would an interviewer have to extract it?

<details>
<summary>🔑 How to think about it</summary>

Strong candidates volunteer the weakness at the pitch's close (Insist on the Highest Standards) rather than defending territory until it falls. For QLoRA, the likely candidates: uncontrolled baseline comparison, or no rank ablation. For the RoPE project: missing pretrained/lexical baselines, or pair-level split. Whichever you pick, the follow-up you must survive is "so why didn't you fix it?" — time-boxed scope with a named next step is an acceptable answer; ignorance of the flaw is not.
</details>

**R2.** An interviewer asks a question you've never considered about your own project. Script your first 30 seconds — the exact sentences — for the case where you *can* reason it out live, and the case where you genuinely can't.

<details>
<summary>🔑 How to think about it</summary>

Case 1 pattern: restate the question in your own terms → name the relevant mechanism you *do* know → reason forward out loud, flagging each assumption. Case 2 pattern: "I don't know" within ten seconds → what you'd measure or read to find out → connect to the nearest thing you do know. Both patterns score; a five-minute bluff scores negatively. Rehearse them until the honest version is your reflex under pressure.
</details>

---

## 🟢 Summary

- 12 quiz questions + 2 reflections here, 9 concept checks in lessons 1–2, plus 8 five-deep Gold chains and a 9-item assessment: Session 1's full question volume.
- Miss a quiz question → it’s a gap. Log it in [PROGRESS.md](../../PROGRESS.md) like any drill failure.
- Cleared everything? Deliver the two pitches once more, cold, tomorrow morning. If they survive a day of forgetting, say "next" for Session 2.
