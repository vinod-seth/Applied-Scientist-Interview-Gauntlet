# Session 2 — Chapter Quiz Bank

| | |
|---|---|
| **Prerequisites** | Lessons 1–3 |
| **Time** | ~40 min |
| **Rules** | Closed notes. Reason out loud where asked — the interview is spoken, not written. |

12 quiz questions plus 2 reflection prompts, on top of the concept checks in lessons 1–2 and the assessment chains. No mastery credit for completing this bank — it exists to find floors cheaply, before a Bar Raiser finds them expensively.

---

## 📝 Chapter Quiz

**Q1.** Your RAG eval shows recall@k = 0.94 but end-to-end accuracy of 0.51. Which single experiment best isolates whether the generator is the bottleneck?

* [ ] Sweep chunk size at fixed k
* [x] The oracle-context condition — supply the gold passage alone and measure residual error, leaving no retrieval explanation available
* [ ] Swap FAISS for an exact index
* [ ] Increase k to 50

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Recall@0.94 means evidence is usually present, so the loss is downstream — but "downstream" still splits into *the generator is weak* and *the context we built is hard to use*. Oracle context (gold only, minimal length) collapses that ambiguity. Option 4 makes it worse: more context lengthens the prompt and invites position effects, adding a confound rather than removing one.
</details>

**Q2.** *Debugging.* Your bge-small retrieval underperforms a BM25 baseline on the same corpus. Which check comes first?

* [x] Whether the query-side instruction prefix was applied and embeddings were L2-normalized before an inner-product index
* [ ] Retrain the embedder on your corpus
* [ ] Switch to IVF for speed
* [ ] Increase chunk size to 1024

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 1.** Both are silent failures that produce plausible-but-degraded rankings: BGE v1 expects the query instruction prefix, and unnormalized vectors make inner product ≠ cosine. Suspect the plumbing before the model. Option 3 trades recall for latency (wrong direction); option 2 is weeks of work for a bug you haven't ruled out.
</details>

**Q3.** A wrong answer contains a claim supported by *no* retrieved chunk and *no* gold passage. Correct bucket?

* [ ] Retrieved-but-ignored
* [ ] Retrieval-miss
* [x] Hallucination — the claim is unsupported by anything in context, which is precisely the operational definition
* [ ] Unclassifiable

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** The support test is what separates hallucination from the other two: retrieved-but-ignored requires the claim to be traceable to *some* retrieved text (the model grounded on the wrong evidence); hallucination requires support from nothing. If the gold passage was also absent, note the case is *jointly* a retrieval-miss and a hallucination — ordering your decision tree (miss first, then support) is what makes the buckets mutually exclusive.
</details>

**Q4.** Why does a strict exact-match judge bias your conclusion *toward* blaming the generator?

* [ ] It inflates retrieval-miss counts
* [x] It scores correct paraphrases as wrong; those cases usually have gold in context, so they land in retrieved-but-ignored — the bucket used to indict generation
* [ ] It has no systematic direction
* [ ] It deflates hallucination only

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** This is the uncomfortable one: the judge's bias points the same way as your headline claim, so it's self-serving. Recognizing that a measurement error *supports your own conclusion* — and volunteering it — is exactly the honesty an AS round scores. The fix is a stratified per-bucket audit, not a better vibe.
</details>

**Q5.** *Scenario.* Failure distribution is 20% retrieval-miss, 55% retrieved-but-ignored, 25% hallucination. You have one week. Best first move?

* [ ] Add a cross-encoder reranker
* [ ] Switch to a larger embedding model
* [x] Attack the generation side: grounding-hardened prompting with quoted evidence spans plus permitted abstention, and a claim-support gate
* [ ] Increase k from 5 to 25

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** 80% of the mass is generation-side. Rerankers and bigger embedders (options 1–2) attack the 20% bucket — the fashionable fix aimed at the wrong target. The discipline being tested: let the taxonomy allocate the budget, and be able to say out loud which popular upgrade you're *declining* and why.
</details>

**Q6.** Chunk size 128 → 512 at fixed k=10. Most defensible prediction?

* [ ] Retrieval-miss falls monotonically
* [ ] Hallucination falls
* [x] Two effects compete — dilution pushes misses up while fragmentation falls — and the 10×512 context now invites position effects; the net depends on whether answers span sentence boundaries
* [ ] Nothing changes; only k matters

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** A confident single-direction answer is the trap. Larger chunks dilute embeddings (misses up) but keep evidence intact (fragmentation down), and chunk×k jointly set context length, so the two sweep axes aren't independent conditions. Naming the tension plus the interaction is the Gold answer; picking a direction without it is Silver at best.
</details>

**Q7.** ECE with 15 fixed-width bins = 2.9%; with 15 equal-mass bins = 6.4%. What do you report?

* [ ] Fixed-width, since it's standard
* [ ] Equal-mass, since it's higher and therefore conservative
* [x] Both, with the scheme stated — the discrepancy reveals that prediction mass concentrates near confidence 1.0, which is itself a finding
* [ ] Neither; use accuracy

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** There is no canonical "the" ECE — it's an estimator with knobs. The gap between schemes tells you fixed-width bins are mostly empty and smoothing over the region where the data lives. Reporting one number without its scheme is unreproducible; reporting both, with the interpretation, is what a scientist does.
</details>

**Q8.** *Code output prediction.* You apply temperature scaling with T = 2.2 and re-evaluate. What must happen to top-1 accuracy?

* [x] Exactly unchanged — dividing logits by a positive scalar is strictly monotone per example, so every argmax is preserved
* [ ] It improves slightly
* [ ] It degrades slightly
* [ ] It depends on the dataset

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 1.** Bit-for-bit unchanged (up to floating-point ties). If your accuracy moved after applying TS, you have a bug — most likely you refit the head, applied T inconsistently between fit and eval, or shuffled the eval set. This makes accuracy a free integrity check on any TS implementation.
</details>

**Q9.** Why is NLL, not ECE, minimized when fitting T?

* [ ] NLL is faster to compute
* [x] ECE is piecewise-constant in T because bin memberships jump at discrete points, giving no usable gradient; NLL is smooth and convex in T for fixed logits, and is a proper scoring rule
* [ ] They have identical minimizers, so it doesn't matter
* [ ] ECE isn't differentiable because of the absolute value only

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** As T varies, examples cross bin boundaries only at discrete points — the objective is a step function, flat almost everywhere. NLL in β = 1/T is a sum of log-sum-exp minus a linear term: convex, smooth, one-dimensional. Option 4 misdiagnoses the cause (the binning, not the absolute value, is what kills it).
</details>

**Q10.** *Scenario.* Under severity-5 corruption, accuracy = 0.31 while mean confidence = 0.88. Which mechanism explains the persistence of confidence?

* [ ] The softmax is broken at low accuracy
* [x] Corrupted inputs produce off-manifold features; the linear classifier head extrapolates them to large-margin logits, and softmax saturates those margins into near-certainty — the model has no mechanism to detect it left familiar territory
* [ ] The model memorized the corruption
* [ ] ECE binning artifacts

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The chain is: off-manifold features → linear head extrapolates without bound → large margins → saturated softmax. Nothing in a standard discriminative classifier represents "this input is unlike training data," which is exactly why shift detection is a *separate* system rather than something you can read off the confidence.
</details>

**Q11.** Which result is the decisive exhibit that a clean-fitted T cannot serve shifted data?

* [ ] ECE rises with severity under fixed T
* [x] The oracle temperature refit at each severity increases monotonically — the *needed* correction moves, so a scalar fitted on clean data is the wrong member of a family
* [ ] Accuracy falls with severity
* [ ] Ensembles outperform single models

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Option 1 shows TS *failed* but not *why* — it's consistent with "TS is just weak." The oracle-T sweep proves the correction itself is severity-dependent, violating temperature scaling's uniform-inflation assumption. That distinction — demonstrating the mechanism rather than the symptom — is what makes a reproduction your own work.
</details>

**Q12.** A team asks you to make confidences trustworthy under expected shift. Which statement is defensible?

* [ ] "Temperature scaling on clean validation will fix it"
* [ ] "Ensembles make calibration shift-invariant"
* [x] "Selective prediction plus shift detection can control risk and make abstention visible — but none of it restores the accuracy lost to shift; if severe shift is routine, the fix is training-time"
* [ ] "Report accuracy instead; calibration is academic"

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 3.** It layers the interventions and — critically — states the non-promise. Option 1 is the exact failure your project documented; option 2 overclaims (ensembles degrade too, just more gracefully). The Bar Raiser is listening for the refusal as much as the recommendation: a candidate who promises more than the method delivers fails Earn Trust regardless of technical depth.
</details>

---

## 🟢 Reflection Prompts (open-ended)

**R1.** Both Session 2 projects make causal-sounding claims from observational sweeps. Write the one-sentence scoped version of each headline — the version you'd actually say to a Bar Raiser. Then ask: does your current pitch already say it that way, or does it overclaim until challenged?

<details>
<summary>🔑 How to think about it</summary>

Scoped versions read like: "At my operating points, for this generator, on this corpus, under this judge, generation-side failures carried most of the residual error"; and "For this architecture and this corruption family, a temperature fitted on clean validation degraded monotonically with severity." Overclaiming until challenged is a losing trade — the interviewer discovers the caveat instead of you supplying it, and every later answer is then audited rather than heard. Pre-scoping costs eight words and buys the round.
</details>

**R2.** Your calibration project reproduces a known result (Ovadia et al. 2019). Script the 30 seconds you'd use when a Bar Raiser says "that's already known — what did *you* add?" Then stress-test it: does your answer survive "so why should we be impressed?"

<details>
<summary>🔑 How to think about it</summary>

The strong shape: no defensiveness, no invented novelty. "The result is known; what's mine is a validated instrument on a specific model and corruption family — cross-checked binning, bootstrapped CIs, and an oracle-T sweep that shows the *mechanism*, not just the symptom. Reproducing carefully is how I learned to trust my own measurements, and it's why I'd catch it if the instrument were lying." The second question is answered by the same honesty: the impressive part isn't the finding, it's that you can attack your own numbers. A candidate who inflates a reproduction into a discovery loses the round on Earn Trust — and Bar Raisers probe exactly here because it's where freshers most often bluff.
</details>

---

## 🟢 Summary

- 12 quiz questions + 2 reflections here, plus the concept checks in lessons 1–2 and 8 five-deep Gold chains: Session 2's full question volume.
- Miss a question → it’s a gap. Log it in [PROGRESS.md](../../PROGRESS.md) like any drill failure.
- Cleared everything? Deliver both pitches cold tomorrow morning. If the scoped claims survive a night of forgetting, take the assessment.
