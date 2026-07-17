# Lesson 3 — Tiered Challenges & Boss Fight: Systems Bar Raiser

| | |
|---|---|
| **Prerequisites** | Lessons 1–2 complete; Session 2 Metric Vault rows filled from run logs |
| **Session time** | ~90 min (drills + boss fight; retries add time) |
| **Domain tag** | Assessment / Interview Simulation |
| **Objective** | Reach Gold on all 8 core topics to unlock Session 3 |

🔬 **Interactive companion:** [▶ Open the Calibration & Temperature Lab in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/02_systems_and_evaluation/02_calibration_and_temperature_lab.ipynb)

> 📍 **Roadmap:** Tiered drills first (self-score honestly), then the Boss Fight. Session 1's boss tested whether you understood what you *built*. This one tests whether you can defend what you *concluded* — a harder standard, because conclusions have confounds and instruments have error bars.

## 🟢 Before You Start

1. Open [PROGRESS.md](../../PROGRESS.md) in a second window.
2. Every Session 2 Metric Vault slot must be filled from run logs or explicitly marked `QUALITATIVE-ONLY`. The boss probes what you filled.
3. Have both project pitches ready (4 min each). Remember the menu rule: only surface hooks you can defend five deep.

---

## 🟢 Tiered Drills

Attempt all three tiers per topic, in order. Record your honest tier. Stop at your floor and log it — that's the map working.

### Topic 1: Failure-Mode Operational Definitions

**🥉 Bronze:** Name the three failure modes and state, in one sentence each, the instrumented test that identifies them.

**🥈 Silver:** Write the classifier as an ordered decision tree, including what labels it requires. Then name the two boundary cases where the buckets overlap.

**🥇 Gold — 5 follow-ups:**

1. A wrong answer, gold passage in context, content traceable to a *different* retrieved chunk. Which bucket, and why not the other two?
2. Your gold labels are passage-level; retrieval is chunk-level. Define "gold in context" precisely and defend the threshold.
3. A multi-hop question needs two passages; one is retrieved. Classify it, and say what your taxonomy is missing.
4. What is your taxonomy's own error rate, and how did you (or would you) estimate it?
5. Give me one real example from your logs for each bucket.

<details>
<summary>🔑 Gold check</summary>

1. Retrieved-but-ignored — specifically a *selection/grounding* failure. The distinguishing test is support: the claim is traceable to retrieved text, so it isn't hallucination (which requires support from nothing retrieved); the gold was present, so it isn't a miss. The practical consequence matters too: selection failures respond to reranking and source-discipline prompting; hallucination doesn't.
2. Any defensible rule works if stated and defended: "any token overlap with the gold span" is permissive (inflates retrieved-but-ignored by counting fragments as sufficient); "full answer span present in one chunk" is strict (inflates retrieval-miss). The Gold move is naming the direction each rule biases your headline number.
3. It's a *retrieved-but-insufficient* case, which a three-bucket taxonomy has no home for — it gets forced into "miss" or "ignored" and corrupts both. Admitting the taxonomy is incomplete for multi-hop is stronger than defending three buckets as exhaustive.
4. Requires a stratified human audit: sample per bucket, relabel, report agreement. "Unaudited, and here's the audit I'd run" scores; "I didn't think about it" doesn't.
5. Non-negotiable at Gold. If you can't produce real examples, you inspected aggregates, not answers — and "traced every wrong answer" is then an overclaim.
</details>

---

### Topic 2: Sweep Design & The Oracle-Context Test

**🥉 Bronze:** What did you sweep, over what range, and what was held fixed?

**🥈 Silver:** Explain the recall-up/error-flat dissociation and what it licenses you to conclude. Then describe the oracle-context experiment and what it isolates.

**🥇 Gold — 5 follow-ups:**

1. Higher k raises recall *and* lengthens context. How do you separate "generation is weak" from "context got harder"?
2. You tuned retrieval and diagnosed failures on the same eval set. What's the threat, and the fix?
3. What would falsify "generation dominates"? Name three falsifiers.
4. With n = `[FILL]` eval questions, is your bucket-share difference outside noise? How would you show it?
5. Your conclusion holds for your generator. What's the scope of the claim you're actually entitled to?

<details>
<summary>🔑 Gold check</summary>

1. The oracle-context condition breaks the tie: supply the gold passage alone (minimal context). High oracle accuracy ⇒ the problem is context construction (length/position), not the generator; low oracle accuracy ⇒ the generator itself. Without this, the two explanations are entangled and the claim is unscoped.
2. Adaptive overfitting: sweeping and concluding on one set means the operating point is chosen to look good on the very data used to attribute blame. Fix: a held-out slice scored once at the end, or a fresh split for the final attribution.
3. (a) A stronger generator closes the gap ⇒ finding was generator-specific, not architectural. (b) A judge audit shows "ignored" was largely judge error ⇒ the dominant bucket was a measurement artifact. (c) Gold-label noise inflated the miss rate ⇒ the retrieval side was understated.
4. Bucket shares are proportions; a binomial/bootstrap CI on each share is the check. At small n the CIs overlap and "dominant" is not supported — knowing that your n may not support the word "dominant" is exactly the honesty the round scores.
5. "At my operating points, for my generator, on this corpus, under my judge." Volunteering the scope *before* being forced to is the difference between a scientist and an advocate.
</details>

---

### Topic 3: Chunking Mechanics & Position Effects

**🥉 Bronze:** Name two mechanisms by which chunk size changes answer quality.

**🥈 Silver:** Give all three (dilution, fragmentation, position) and explain how chunk size and k interact through the context budget.

**🥇 Gold — 5 follow-ups:**

1. Chunk 128 → 512 at fixed k=10: predict which bucket grows and why.
2. Does chunk overlap fix fragmentation? What does it cost?
3. Cite the position effect and state what it predicts for large-k contexts.
4. Why would sentence-boundary chunking beat fixed-token chunking — and when would it not?
5. Your sweep grid was `[FILL]`. What does the grid's coarseness hide?

<details>
<summary>🔑 Gold check</summary>

1. Two competing effects, and saying so is the answer: dilution rises (each embedding averages more topics → retrieval-miss grows), while fragmentation falls (more evidence per chunk). Net direction depends on whether your answers span sentence boundaries. Also, 10×512 tokens of context invites position effects. A confident single-direction answer without the tension is Silver at best.
2. Partly — overlap re-joins evidence split across boundaries, at the cost of duplicated content in the index (larger index, redundant retrieved slots consuming the k budget) and inflated apparent recall.
3. Liu et al. (2023), *Lost in the Middle* (arXiv:2307.03172): retrieval accuracy degrades for evidence in the middle of long contexts. Prediction: at large k, gold passages ranked mid-list are used worse than the same passage ranked first — so ranking matters even when recall is unchanged.
4. Sentence boundaries preserve semantic units, so embeddings represent coherent propositions rather than truncated fragments. It fails when answers require cross-sentence context (co-reference) or when sentences are wildly uneven in length, producing unstable chunk sizes.
5. Coarse grids hide non-monotonicity — the optimum may sit between your points, and a two-point "trend" can be noise. Naming the interaction (chunk×k both feed context length, so grid corners are not comparable conditions) is the Gold detail.
</details>

---

### Topic 4: Judge Validity & Measurement Error

**🥉 Bronze:** Name your judge and what it decides.

**🥈 Silver:** State each judge type's bias direction and which failure bucket that bias inflates or deflates.

**🥇 Gold — 5 follow-ups:**

1. Exact-match judge: which bucket does its bias inflate, and by what mechanism?
2. If judge precision on "wrong" is 85%, what happens to your headline distribution?
3. Would you trust an LLM judging output from its own model family? Why is that a validity threat?
4. Design the cheapest audit that could actually change your conclusion.
5. You changed the prompt to require quoted evidence. Does your judge still work?

<details>
<summary>🔑 Gold check</summary>

1. Retrieved-but-ignored. Mechanism: correct paraphrases are scored wrong; those examples usually *do* have gold in context, so they land in "ignored" — inflating exactly the bucket used to blame the generator. This is a self-serving bias in the direction of your own conclusion, which is why it must be volunteered.
2. Roughly 15% of "wrong" answers were actually correct; they're concentrated where paraphrase is likely, so bucket shares shift non-uniformly — you can't just scale everything by 0.85. Correct answer: re-derive shares after removing audited false-wrongs, and state that CIs widen.
3. No — self-preference/style bias: judges favor phrasings resembling their own generations, so a judge from the generator's family systematically under-flags its errors. Validity threat because judge error correlates with the thing being measured rather than being random noise.
4. Stratified sample (e.g. 50/bucket), two annotators, report per-bucket agreement + Cohen's κ. Cheap because it's hundreds of items, decisive because it bounds the bucket that carries the conclusion.
5. Probably not — a style change (quoted spans) breaks lexical judges and shifts LLM-judge behavior. Judges must be re-validated whenever answer style changes; otherwise pre/post numbers aren't comparable. Spotting that a *fix* invalidates the *measurement* is Gold-tier.
</details>

---

### Topic 5: ECE & Its Estimator's Pathologies

**🥉 Bronze:** Define calibration and write the ECE formula with your configuration.

**🥈 Silver:** Name three ways your ECE number lies, and what you'd report alongside it.

**🥇 Gold — 5 follow-ups:**

1. Fixed-width gives 3.1%, equal-mass gives 5.8%. Explain, and say which you'd report.
2. Construct a useless model with near-zero ECE.
3. What does ECE not see at all?
4. Is your severity-5 vs. severity-1 ECE difference outside estimator noise? Show how you'd know.
5. When is ECE the wrong metric entirely for a stakeholder?

<details>
<summary>🔑 Gold check</summary>

1. Prediction mass concentrates near confidence 1.0; fixed-width bins are mostly empty and smooth over the region where the data actually is, while equal-mass bins subdivide it. Report *both* with the scheme stated — the discrepancy is information about the confidence distribution, not a bug to hide.
2. Predict the marginal class distribution always, with confidence equal to the base rate. Accuracy equals confidence in aggregate ⇒ ECE ≈ 0, skill = 0. Proper scoring rules (NLL, Brier) catch it because they reward sharpness too.
3. The non-top-label probability vector (classwise calibration); within-bin cancellation of over/under-confidence; and anything about *which* examples are miscalibrated (a subgroup can be badly miscalibrated while aggregate ECE looks fine).
4. Bootstrap the eval set per severity and compare CIs; also check per-severity n. Without this, "ECE rises with severity" is a claim about the estimator as much as the model.
5. Under asymmetric error costs. A calibrated 90% means something very different when false negatives are catastrophic; expected-cost or risk–coverage curves serve the decision, ECE doesn't.
</details>

---

### Topic 6: Temperature Scaling — Derivation and Limits

**🥉 Bronze:** What is TS, and what's your fitted T?

**🥈 Silver:** Prove accuracy invariance; justify NLL as the objective; explain why one parameter suffices in-distribution.

**🥇 Gold — 5 follow-ups:**

1. Prove NLL is convex in T for fixed logits (sketch it).
2. Why fit on validation rather than training data?
3. When does vector scaling beat temperature scaling, and what does it risk?
4. Your T = `[FILL]`. What does that value tell you about the model?
5. Would label smoothing during training have removed the need for TS?

<details>
<summary>🔑 Gold check</summary>

1. One-dimensional in β = 1/T: NLL = Σ [log-sum-exp(βz) − βz_y], a sum of log-sum-exp (convex in β, being a composition with a linear map) minus a linear term ⇒ convex. Sketching the structure is enough; a full proof isn't expected, but "it's convex, here's why" is.
2. Training logits are the overfit ones — the network already drove NLL down on them by sharpening confidence, so the fitted T would come out near 1 and correct nothing. Validation logits exhibit the true generalization miscalibration.
3. When miscalibration is class-heterogeneous (imbalanced or fine-grained classes), a per-class parameterization can fit better. Risks: it can change accuracy (no longer a single monotone per-example rescale) and overfits small validation sets.
4. T > 1 ⇒ the raw model was overconfident, and the magnitude indicates how much. Typical modern nets land ~1.5–3. Not knowing your own T is an instant credibility hit.
5. Partly — label smoothing penalizes saturated softmax during training and reduces raw overconfidence, but it doesn't guarantee calibration and interacts with T-fitting (a smoothed model may even need T < 1). "Partly, and here's the interaction" beats a yes/no.
</details>

---

### Topic 7: Why Calibration Breaks Under Shift

**🥉 Bronze:** State the phenomenon: what happens to accuracy and confidence as severity rises?

**🥈 Silver:** Give the mechanism — off-manifold features, linear head extrapolation, softmax saturation — and why it's no longer a uniform inflation.

**🥇 Gold — 5 follow-ups:**

1. Why does clean-fitted T fail specifically at high severity? Mechanism, not observation.
2. What does oracle-T per severity look like, and why is that the decisive exhibit?
3. If severity were observable at test time, would per-severity T solve the problem?
4. Which corruption broke your calibration worst, and reason about why.
5. Do ensembles escape this failure?

<details>
<summary>🔑 Gold check</summary>

1. TS assumes miscalibration is a single global logit-scale factor. Under shift, activations land off-manifold; the linear head extrapolates to large-margin logits; softmax saturates. The required correction now varies with severity *and* corruption type, so the uniform-inflation assumption is violated and no clean scalar fits the family.
2. T*(severity) increases monotonically. Decisive because it shows the *needed* correction moves — it's not that TS is weak, it's that a scalar fitted on one distribution is the wrong member of a family. This is the experiment that turns a citation into your own finding.
3. It helps ECE at each severity, but severity isn't observable in production and corruption *type* also matters — so it collapses into shift detection plus a lookup, which is a different (harder) system. Recognizing it as shift detection in disguise is the Gold move.
4. Reason from your data, don't recite. Directionally: noise pushes off-manifold fast; blur degrades features more gradually. Whatever your `[FILL]` says, the answer must connect the corruption's effect on features to the confidence–accuracy gap.
5. No — they degrade too (Ovadia et al. 2019), just more gracefully; disagreement between members provides some signal that a single model lacks. "More robust, not immune" is the correct claim.
</details>

---

### Topic 8: Acting Under Miscalibration

**🥉 Bronze:** Name two things you'd build for a team deploying under expected shift.

**🥈 Silver:** Describe selective prediction and the risk–coverage curve; explain what shift detection adds that calibration doesn't.

**🥇 Gold — 5 follow-ups:**

1. How do you pick an abstention threshold without labeled shifted data?
2. Your RAG project also has an abstention story. Connect them.
3. What do you refuse to promise this team?
4. When is the right fix training-time rather than post-hoc?
5. What's the monitoring signal that says "stop trusting confidence"?

<details>
<summary>🔑 Gold check</summary>

1. Conservatively from clean validation plus a safety margin; better, buy a small labeled shifted set — and say explicitly that this is the thing worth paying for. Acknowledging you can't do it well without *some* shifted labels is honest and correct.
2. Both convert unreliable outputs into visible abstentions: RAG's "not in context" refusal is selective prediction at the answer level; the calibration project's threshold is selective prediction at the confidence level. Both trade coverage for correctness and both should be reported as risk–coverage, not a single accuracy number. Making this connection unprompted demonstrates you see your projects as one body of work.
3. That calibration management restores accuracy. It does not. It controls *risk* given degraded accuracy.
4. When severe shift is expected and routine — augmentation with realistic corruptions attacks the accuracy loss itself, which no post-hoc method can. Post-hoc is for shift you can't anticipate or afford to train for.
5. A distribution-level signal independent of confidence: feature-space distance to training distribution, energy score, or a rising abstention rate. The point is it must not be derived from the same confidence you already distrust.
</details>

---

## 🟢 Self-Scoring Protocol

Update PROGRESS.md: mark each of the 8 topics `[B]`/`[S]`/`[G]`, log every floor-hit with the exact chain and depth reached. Gold on all 8 is required to attempt the boss fight. Below Gold: study the floor-hit map, re-read the refresher at the exact stall point, re-drill.

---

## 🟢 Boss Fight: Systems Bar Raiser

> ⚔️ **Format:** Simulated 50-minute science-depth round across both Session 2 projects. Answer out loud, fully, before reading the next question. This Bar Raiser specializes in methodology — every question attacks the gap between what you measured and what you concluded.

### Round Setup

1. Set a 50-minute timer.
2. Deliver both pitches (~4 min each, ≤ 8 min total).
3. Answer the sequence below without skipping ahead.
4. Score against the rubric.

### The Interrogation

**Opening — RAG project:**

> "You wrote that you traced *every* wrong answer to its root cause. Walk me through the classifier for one wrong answer. Exact rules, in order."

*(Expecting: the ordered decision tree — gold-in-context by a stated overlap rule, then answer-support against context, then support against any retrieved chunk. Plus bucket counts `[FILL]` and the taxonomy's own error rate. Naming boundary cases unprompted is the Gold signal.)*

> **Follow-up 1:** "Your headline is that generation dominates. Prove it — then tell me what would falsify it."

*(Expecting: recall-up/error-flat dissociation with numbers `[FILL]`; the oracle-context result `[FILL]` or an honest admission that it's the missing experiment; then three falsifiers named without prompting. Scope the claim to your generator, corpus, operating points, and judge.)*

> **Follow-up 2:** "Higher k raised recall *and* lengthened context. Separate 'the generator is weak' from 'the context got harder.' "

*(Expecting: recognition that the two are entangled at high k, and the oracle-context condition — minimal context, gold only — as the tie-breaker. Bonus: cite Lost in the Middle for why long context is itself a treatment, not a neutral background.)*

> **Follow-up 3:** "Your judge decided what counts as wrong before any of this ran. Which bucket does its bias inflate, and by how much?"

*(Expecting: the judge named, bias *direction* per bucket, and the recognition that a strict lexical judge inflates 'retrieved-but-ignored' — the bucket that carries the conclusion. Then the audit: run, or designed and named as a gap. "I didn't validate the judge" without the design is a floor-hit.)*

> **Follow-up 4:** "One week, halve end-to-end error. Spend it — and tell me what you refuse to build."

*(Expecting: the failure distribution as budget allocator; generation-side fixes if that's where the mass is (grounding prompts with quoted spans, abstention, claim-support gating); explicit refusal of a reranker if retrieval-miss is the smaller bucket. Plus a frozen judge/eval set and a fresh held-out slice at the end.)*

**Transition — calibration project:**

> **Follow-up 5:** "Define ECE as you computed it. Now give me three ways that number lies."

*(Expecting: formula with M/scheme/n `[FILL]`; binning sensitivity, mass concentration in top bins, within-bin cancellation; and that ECE is top-label-only and gameable. The degenerate-model example earns the top mark.)*

> **Follow-up 6:** "Temperature scaling. Derive it, and prove it can't change accuracy."

*(Expecting: monotone per-example transform ⇒ argmax preserved; NLL as objective because ECE is piecewise-constant in T; one parameter suffices in-distribution because overconfidence is approximately uniform logit inflation. Your T = `[FILL]`.)*

> **Follow-up 7:** "Now the mechanism: why does your clean-fitted T fail at severity 4? Not that it does — why."

*(Expecting: off-manifold features → linear head extrapolates to large margins → softmax saturates → miscalibration is severity- and corruption-dependent, violating the uniform-inflation assumption. Exhibit: oracle-T rising monotonically with severity `[FILL]`. Stalling here after putting "demonstrated where TS fails" on a resume is the round's worst outcome.)*

> **Follow-up 8:** "This is a known result — Ovadia et al. showed it in 2019. What did *your* project add?"

*(Expecting: no defensiveness. The honest frame: careful reproduction with validated instruments on a specific model/corruption family, plus whatever is genuinely yours — the oracle-T sweep, per-corruption breakdown, or an estimator cross-check. "Reproducing a known result carefully is how you learn to trust your own instruments" is a legitimate and strong answer; claiming novelty you don't have is fatal.)*

> **Follow-up 9:** "A medical-imaging team wants to ship this classifier into a setting with expected shift. What do you tell them to build — and what do you refuse to promise?"

*(Expecting: layered answer, cheapest first — selective prediction with a risk–coverage curve; shift detection independent of confidence; ensembles if budget allows. The refusal: none of this restores accuracy. If severe shift is routine, the fix is training-time. The explicit non-promise is what a Bar Raiser is listening for.)*

### Scoring Rubric

| Question | Gold standard | Floor indicators |
|---|---|---|
| Failure classifier | Ordered rules + boundary cases + bucket counts + own error rate | Names three buckets, no instrumented tests |
| Prove + falsify | Dissociation *and* oracle test; three falsifiers volunteered; claim scoped | States conclusion; no falsifier; unscoped |
| k / context confound | Identifies entanglement; oracle-context breaks it | "More context is just better" |
| Judge bias | Direction per bucket + audit design | "The judge was fine" |
| One-week plan | Distribution allocates budget; names refused build; frozen measurement | Lists popular RAG upgrades |
| ECE definition + attacks | Config stated; 3 pathologies; degenerate example | Formula only, no attacks |
| TS derivation | Monotonicity proof; NLL rationale; knows own T | "It divides logits by T" |
| Shift mechanism | Off-manifold → linear extrapolation → saturation; oracle-T exhibit | "It just stops working" |
| Contribution honesty | Reproduction framed accurately; owns the gap | Overclaims novelty |
| Deployment advice | Layered build + explicit non-promise | Promises calibration fixes accuracy |

**Passing condition:** all 10 at Gold standard. Zero hand-waves, zero unsupported numbers, zero overclaims.

### After the Boss Fight

1. **Update PROGRESS.md:** tier each of the 8 core topics; log XP (10/depth level, max 50/chain; 5 per honest floor-hit); record floor-hits with exact chain and depth.
2. **If cleared:** 🎉 Session 3 unlocks. Update the Streak Tracker. Say **next** when ready.
3. **If not cleared:** study the floor-hit map, re-read the exact refresher where you stalled, re-drill that topic's Gold chain, retry from question one.

---

## 🟢 Summary

- Session 2's bar is methodological: every claim must survive attacks on its definitions, its instruments, and its inferences.
- The two projects share a spine — both make causal-sounding claims from observational evidence, and both are defended by naming the isolating experiment (oracle context; oracle temperature) and the falsifiers.
- Volunteering your own confounds is not weakness; it is the single strongest signal available in this round.
- Passing unlocks Session 3: ML Fundamentals — Core Theory.

**The enemy is not the boss fight. The enemy is a conclusion you can't scope.**
