# Session 5 — Chapter Quiz Bank

| | |
|---|---|
| **Prerequisites** | Lessons 1–6 |
| **Time** | ~40 min |
| **Rules** | Closed notes, paper to hand. Say the trade-off out loud before selecting — this session's questions are about *choices*, and recognizing the right one is not the same as being able to defend it. |

12 quiz questions plus 2 reflection prompts. Nothing here is scored; the bank exists to find the judgments you *think* you have before an interviewer finds them.

---

## 📝 Chapter Quiz

**Q1.** Mixed-precision AdamW fine-tuning costs roughly 16 bytes per parameter. Freezing the base model removes:

* [ ] The weight storage, which is the largest term
* [x] The gradient, optimizer-state, and master-weight terms — 14 of the 16 bytes, which exist only for *trainable* parameters
* [ ] The activation memory
* [ ] Nothing; it only saves compute

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The breakdown is 2 (fp16 weights) + 2 (gradients) + 4 + 4 (Adam moments in fp32) + 4 (fp32 master weights). Only the first 2 bytes belong to the frozen model. Option 3 is the trap worth noticing: activations are *unaffected* by freezing the base, which is exactly why gradient checkpointing still appears alongside QLoRA.
</details>

**Q2.** For LoRA on a $4096 \times 4096$ matrix at rank 8, the trainable parameter count and initialization are:

* [ ] $r^2 = 64$ parameters, both matrices random
* [x] $r(d+k) = 8 \times 8192 = 65{,}536$ parameters, with $A$ Gaussian and $B$ zero so the update is exactly zero at step 0
* [ ] $dk = 16.8$M parameters, both matrices zero
* [ ] $r \times d = 32{,}768$ parameters, both matrices Gaussian

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Two separate facts, both commonly fumbled. The count is $r(d+k)$ because there are *two* matrices — 65,536 against 16.8M is 0.4%. The zero-initialization of $B$ is what makes training start exactly at the pretrained model; initializing both randomly (option 4) would perturb the model with a random rank-$r$ matrix before the first step.
</details>

**Q3.** QLoRA's memory saving relative to plain LoRA comes from:

* [ ] Training fewer parameters than LoRA does
* [x] Storing the *frozen base* in 4-bit NF4 — a different term from LoRA's saving, which was the optimizer/gradient state; the two compose
* [ ] Removing the KV cache during training
* [ ] Using a smaller batch size

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** This is the discriminating question in the PEFT area. LoRA deletes the ~14 bytes/param of optimizer and gradient state; QLoRA additionally shrinks the frozen weights from ~14 GB to ~3.5 GB at 7B. Candidates who describe them as "the same saving" lose the follow-up immediately. The cost of QLoRA is slower steps, from dequantizing on every forward pass.
</details>

**Q4.** You must serve 50 customer-specific LoRA adapters. The right design is:

* [ ] Merge each into its own model copy for zero latency overhead
* [x] Keep them unmerged — one resident base serves a mixed batch and each request adds its own low-rank term through a batched kernel
* [ ] Average the adapters into a single general adapter
* [ ] Convert them to prefix tuning, which needs no weights

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Merging is right for one adapter and wrong for fifty: 50 full checkpoints, and no cross-customer batching. Keeping adapters separate is the S-LoRA/Punica design — the cost is one small extra matmul per adapted layer and adapter residency competing with the KV cache. Option 3 destroys the per-customer behavior; option 4 answers a training question with a serving problem.
</details>

**Q5.** Sampling is preferred over greedy or beam decoding for open-ended generation because:

* [ ] Sampling is computationally cheaper
* [x] Likelihood-maximizing text is degenerate — human text does not occupy the high-probability region, and greedy decoding drifts into self-reinforcing repetition
* [ ] Greedy decoding cannot produce long outputs
* [ ] Sampling produces higher-likelihood sequences

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Holtzman et al. 2020 is the reference. The mechanism worth adding: once a phrase repeats, the context contains evidence for it, so its probability rises — repetition is self-reinforcing. The complementary point is *when the opposite holds*: for translation, extraction, and classification the output distribution is low-entropy, the mode is a good answer, and deterministic decoding is correct.
</details>

**Q6.** Top-p (nucleus) sampling improves on top-k because:

* [ ] It always keeps more tokens
* [x] The candidate set adapts to the distribution's entropy — tiny when the model is confident, large when many continuations are valid — whereas a fixed $k$ is wrong for both cases
* [ ] It removes the need to set temperature
* [ ] It guarantees the argmax token is included

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The two-example contrast is the answer to give aloud: after "The capital of France is" the distribution is near one-hot and $k=50$ admits 49 wrong tokens; after "She opened the door and" thousands of continuations are valid and $k=50$ truncates good ones. Option 3 is wrong in a specific way worth knowing — temperature is applied *first*, so it changes which tokens survive the same $p$ cut.
</details>

**Q7.** Speculative decoding:

* [ ] Approximates the target model's distribution for a 2–3× speedup
* [x] Produces exactly the target model's distribution, using a rejection-sampling step; the speedup comes from verifying $k$ drafted tokens in one weight-reading pass on bandwidth-bound hardware
* [ ] Only works at temperature 0
* [ ] Mainly improves throughput under heavy batching

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Exactness is the property that matters, and it parallels FlashAttention in Session 4 — both are systems optimizations that leave the math untouched. Option 4 is the well-informed distractor: speculative decoding buys *single-stream latency*, and its benefit shrinks under heavy batching where the GPU is already saturated.
</details>

**Q8.** Chinchilla's compute-optimal finding, given $C \approx 6ND$, is:

* [ ] Parameters should absorb most additional compute
* [x] Parameters and tokens should scale in roughly equal proportion — about 20 tokens per parameter — demonstrated by 70B/1.4T outperforming 280B/300B at equal compute
* [ ] Training data quality dominates quantity
* [ ] Loss decreases without bound as compute grows

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The Gopher comparison is the evidence to quote — 4× smaller, 4.7× more data, and it won. Option 4 is worth rejecting explicitly: the fitted form $L = E + A/N^\alpha + B/D^\beta$ has an **irreducible term** $E$, the entropy of text itself, so loss approaches a floor.
</details>

**Q9.** Llama-class 8B models are trained on trillions of tokens, far past 20:1. This is because:

* [ ] The Chinchilla result was later shown to be wrong
* [x] Chinchilla optimizes *training* compute only; when a model is served at scale, inference cost scales with parameters, so overtraining a smaller model is the cheaper lifetime trade
* [ ] More tokens always reduce loss proportionally
* [ ] Smaller models are easier to quantize

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Chinchilla answers a different question, correctly. Option 4 inverts a real finding — heavily overtrained models can be *more* sensitive to post-training quantization, not less, which is a good detail to raise unprompted.
</details>

**Q10.** DPO's central simplification relative to RLHF is:

* [ ] It needs no preference data
* [x] The KL-regularized objective's optimal policy has a closed form that can be inverted to express reward in terms of the policy, turning preference learning into a direct classification loss — no reward model, no PPO loop
* [ ] It removes the KL constraint entirely
* [ ] It replaces human preferences with model-generated ones

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** DPO still needs preference pairs (option 1) and still carries a KL strength $\beta$ inside its implicit reward $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ (option 3); option 4 describes RLAIF. The trade to volunteer: DPO is **off-policy** — it sees only a fixed dataset, while RLHF keeps scoring fresh samples from the current policy, which is why online RLHF can still reach a higher ceiling.
</details>

**Q11.** A cross-encoder reranker cannot serve as the first-stage retriever because:

* [ ] It is less accurate than a bi-encoder
* [x] Its score depends on the query–document pair jointly, so nothing can be precomputed — it costs one forward pass per pair, which is impossible across a corpus
* [ ] It cannot handle documents longer than a paragraph
* [ ] It requires the query at index time

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The impossibility is structural, and it is *why* retrieval stacks are two-stage: cheap recall over everything (hybrid, $k\approx100$), expensive precision over a shortlist ($k\approx5$). The corresponding strength is that full attention across query and document catches negation and qualifiers that two independent summaries lose.
</details>

**Q12.** Two prompts are compared on the same 200 examples: 71% vs 74%. The correct analysis is:

* [ ] A two-sample proportion test on the two accuracies
* [x] A paired test — McNemar's on the discordant examples only, since both systems ran on the same items; a 12-vs-6 fixed/broken split gives $p \approx 0.24$, so the improvement is not established
* [ ] No test needed; 74 > 71
* [ ] A 3-point difference is always noise regardless of $n$

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Option 1 discards the pairing and is needlessly weak. Option 4 overcorrects — whether 3 points is noise depends on $n$ and the discordance rate, which is exactly what the test computes. The non-statistical move that adds the most value: **read the ~18 examples that flipped**, because their pattern says more than the aggregate.
</details>

---

## 🪞 Reflection Prompts

Reflection prompt 1 — *Your QLoRA run, placed in the landscape.* Without notes, write down: (a) the memory arithmetic for your own model at your own batch size, showing which term QLoRA removed and which term LoRA removed; (b) two PEFT methods you did **not** use and the specific reason each was worse for your task; (c) what you would do differently with 8× the GPU memory, and what evidence would tell you it was worth it.

<details>
<summary>🔑 Evaluation Criteria</summary>

The interviewer's real question behind this is "did you choose QLoRA, or did you copy a tutorial?" A strong answer separates the two savings cleanly, names alternatives with task-specific reasons — not generic ones — and answers (c) with a *test* rather than an ambition: train a LoRA and a full fine-tune on a subsample, compare target-task gain against a held-out general benchmark, and switch only if the gap exceeds the cost. Weak answers say "I'd use a bigger model." Note the honest position for a narrow classification task on an in-domain corpus: the constrained update is expected to match full fine-tuning, so more memory may buy nothing — and saying *that*, with the arithmetic, is stronger than claiming an upgrade.
</details>

Reflection prompt 2 — *Re-audit your own RAG study with Lesson 5's methodology.* Take the headline claim from your RAG failure-mode project (`[FILL: your claim]`). Then answer: (a) how many evaluation questions did it rest on, and what is the standard error at that $n$? (b) was any comparison in it paired, and did you analyze it as paired? (c) what was your judge's measured agreement with human labels, and if you don't have that number, what does its absence do to the claim? (d) which of the four evaluation layers did your study actually cover?

<details>
<summary>🔑 Evaluation Criteria</summary>

This prompt is deliberately uncomfortable, and the discomfort is the lesson: most student projects rest on an $n$ small enough that the headline split has a confidence interval nobody computed. A strong answer states the numbers where they exist, marks `[FILL]` or "not measured" where they don't, and — this is the part that scores in a real interview — says what it would take to make the claim defensible: a specific $n$, a paired analysis, a judge audit with $\kappa$, and the oracle-context run as the attribution instrument. An interviewer who hears a candidate audit their own prior work this way learns more about them than any correct answer in the quiz above. Never respond to this prompt by inventing a number.
</details>

---

**Next:** [Mock Round — The LLM Round](06_mock_round.md) if you have not run it, then Session 6.
