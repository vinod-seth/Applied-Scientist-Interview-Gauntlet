# Lesson 2 — Defending Calibration Under Distribution Shift

| | |
|---|---|
| **Prerequisites** | Lesson 1; your calibration project's eval artifacts open |
| **Reading time** | ~25 min + drills |
| **Domain tag** | Uncertainty / Evaluation / Robustness |
| **Resume line under fire** | "Evaluated confidence calibration of a pretrained image classifier under graded corruptions using ECE and reliability diagrams… demonstrated where post-hoc temperature scaling fails to transfer to shifted inputs" |

🔬 **Interactive companion:** [▶ Open the Calibration & Temperature Lab in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/02_systems_and_evaluation/02_calibration_and_temperature_lab.ipynb)

> 📍 **Roadmap:** All technical track. This project's trap is the opposite of the RAG project's: the finding ("TS fails under shift") is *known in the literature* — so the bar isn't the result, it's whether you understand it at mechanism level and measured it correctly. Reproducing a known result badly is worse than not doing it.

## 🟢 Learning Objectives

- Define ECE precisely, attack your own estimator (binning pathologies, what ECE can't see), and read a reliability diagram like an instrument.
- Derive temperature scaling — objective, why one parameter, why accuracy is invariant — and give the *mechanism* for its failure under covariate shift.
- Answer the six hardest follow-ups at AS depth, ending with what you'd actually do about shift in a deployed system.

---

## 🟢 The Interrogation Logic of This Bullet

Three claims live in this bullet: a **measurement** (ECE under graded corruptions), a **phenomenon** (confidence stays high while accuracy collapses), and a **negative result** (temperature scaling doesn't transfer). Ovadia et al. established the negative result at scale in 2019 — your interviewer likely knows the paper. So the interrogation runs: do you know your measurement's own error bars? Can you explain *why* the phenomenon occurs, not just that it does? And did your experiment add anything — even just careful reproduction — beyond citing the literature?

The strongest frame: "I reproduced a known failure carefully enough to trust my own instruments, and I can tell you exactly where the instruments bend."

---

## 🔷 Peer-Level Refresher 1: ECE, and How to Attack Your Own Estimator

**Definition.** A classifier is <abbr title="Stated confidence matches empirical accuracy. Of all predictions made at 80% confidence, about 80% should be correct.">calibrated</abbr> if, among predictions made with confidence p, the accuracy is p. <abbr title="Expected Calibration Error: the average gap between confidence and accuracy, computed in confidence bins and weighted by bin population. Lower is better.">ECE</abbr> estimates the expected gap: partition predictions into M confidence bins B₁…B_M and compute

ECE = Σₘ (|Bₘ|/n) · |acc(Bₘ) − conf(Bₘ)|

with conf the mean top-label probability in the bin. Standard setup: M = 15 fixed-width bins over top-label confidence (the "top-label ECE" popularized by Guo et al. 2017). Your configuration: `[FILL: bins, binning scheme, n]`.

**Now attack it — before the interviewer does.** Four pathologies:

1. **Binning bias–variance.** Few bins smooth over real miscalibration (bias); many bins leave sparse bins with noisy accuracy (variance). The measured ECE value moves with M — a number quoted without its binning scheme is not reproducible.
2. **Mass concentration.** Modern networks pile predictions into the top bins (confidence 0.9–1.0); most <abbr title="Fixed-width bins: the confidence range split into equal-width intervals. Most sit nearly empty when predictions cluster near 1.0.">fixed-width</abbr> bins are nearly empty, so the estimate rides on two or three bins. <abbr title="Equal-mass binning: bin edges placed at quantiles so each bin holds the same number of predictions, subdividing where the data actually lives.">Adaptive (equal-mass) binning</abbr> is the standard fix — did you check both? `[FILL]`.
3. **Within-bin cancellation.** Overconfident and underconfident examples inside one bin cancel; ECE can under-report true miscalibration.
4. **ECE is top-label only.** It says nothing about the full probability vector (classwise calibration) and is gameable — a classifier that always predicts the marginal class distribution with matching confidence scores near-zero ECE while being useless. Pair ECE with NLL and Brier score, which are proper scoring rules: `[FILL: did you report them?]`.

**<abbr title="Reliability diagram: plots per-bin accuracy against per-bin confidence. Points on the diagonal are calibrated; bars below it mean overconfidence.">Reliability diagrams</abbr>** are the visual: per-bin accuracy against per-bin confidence; the identity line is perfect calibration; bars sagging below the line = <abbr title="The model's stated confidence exceeds its actual accuracy — it claims more certainty than it earns.">overconfidence</abbr>. Read them with the bin-count histogram attached — a dramatic gap in a bin holding 0.1% of the mass is noise, not signal.

Attribution: the binning ECE estimator is due to Naeini, Cooper & Hauskrecht (2015, AAAI), as popularized for neural networks by Guo et al. (2017), *On Calibration of Modern Neural Networks* (https://arxiv.org/abs/1706.04599).

---

## 🔷 Peer-Level Refresher 2: Temperature Scaling, Derived

<abbr title="Fixing confidence after training, leaving the trained model's weights untouched.">Post-hoc calibration</abbr>: keep the model frozen, learn a single scalar T > 0, and replace softmax(z) with softmax(z/T). Fit T by minimizing <abbr title="Negative log-likelihood: the loss penalizing low probability assigned to the true class. Smooth and convex in T, unlike ECE.">NLL</abbr> on a held-out **<abbr title="Data drawn from the same distribution the model was trained on — the opposite of the shifted data this project studies.">in-distribution</abbr>** validation set. Three facts you must produce on demand:

1. **Why accuracy is invariant:** z/T is a strictly <abbr title="Order-preserving. Dividing by a positive scalar never reorders logits, so the predicted class cannot change.">monotone</abbr> transform of the logits per example, so <abbr title="The index of the largest logit — i.e. the predicted class.">argmax</abbr> never changes. Only the confidence *distribution* moves — T > 1 flattens (fixes overconfidence), T < 1 sharpens.
2. **Why NLL, not ECE, as the objective:** ECE is piecewise-constant in T (bin memberships jump), so it's ugly to optimize; NLL is smooth, convex in T for fixed logits, and a proper scoring rule whose minimization improves calibration as a side effect.
3. **Why one parameter works so well in-distribution:** Guo et al.'s empirical surprise — overconfidence in modern networks is close to a uniform logit-scale inflation (driven by NLL overfitting late in training: the network keeps sharpening confidence on already-correct training examples), so one global rescale largely undoes it. Vector/matrix scaling add parameters, *can* change accuracy (they're not monotone per-example in the same way), and tend to overfit small validation sets.

Your fitted value: `[FILL: T on clean validation]`. Knowing your own T (typically 1.5–3 for modern CNNs/ViTs) and what it means (T > 1 ⇒ the raw model was overconfident) is a two-second credibility check.

## 🟢 Concept Check

Why does temperature scaling provably never change top-1 accuracy?

* [ ] Because T is fitted on a held-out set
* [x] Because dividing logits by a positive scalar is strictly monotone per example — the ordering of logits, hence the argmax, is preserved
* [ ] Because NLL and accuracy are equivalent objectives
* [ ] It can change accuracy when T < 1

Your ECE with 15 fixed-width bins is 3.1%; with 15 equal-mass bins it's 5.8%. What's the most likely explanation?

* [ ] A bug — the two must agree
* [ ] Equal-mass binning is biased upward by construction
* [x] Predictions are concentrated near confidence 1.0, so fixed-width bins are mostly empty and smooth over miscalibration that equal-mass bins resolve
* [ ] The model is underconfident

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Q1: option 2.** Monotonicity is the whole argument: for any example, zᵢ > zⱼ ⟺ zᵢ/T > zⱼ/T for T > 0. The fitting set (option 1) is about generalization of the calibration, not accuracy invariance.

**Q2: option 3.** With mass piled into the top confidence range, fixed-width binning averages a wide slice of miscalibrated predictions into one or two bins (within-bin cancellation and smoothing), while equal-mass bins subdivide exactly where the data lives. Neither number is "the" ECE — which is precisely why quoting ECE without the binning scheme is indefensible, and why this discrepancy is a feature of your report, not a bug.
</details>

---

## 🔷 Peer-Level Refresher 3: Shift, and Why One Scalar Can't Follow It

**The benchmark methodology.** Graded corruptions (noise, blur, brightness, JPEG, …) at severities 1–5 applied to the test set — the ImageNet-C/CIFAR-10-C protocol of Hendrycks & Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (https://arxiv.org/abs/1903.12261). Your model and data: `[FILL: architecture (timm id), dataset, corruption set]`.

**The phenomenon.** Under increasing severity, accuracy falls fast while mean confidence falls slowly — the confidence–accuracy gap (and ECE) grows with shift. Established at scale by Ovadia et al. (2019), *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift* (https://arxiv.org/abs/1906.02530). Your curve: `[FILL: accuracy and ECE per severity]`.

**The mechanism — this is the Gold-tier answer.** Temperature scaling fits one scalar to the *clean* validation distribution's logit geometry. Under <abbr title="The input distribution changes while the label relationship stays the same — e.g. the same objects photographed through noise or blur.">covariate shift</abbr>, inputs move <abbr title="Off-manifold: into regions of feature space the model never saw in training, where its learned behaviour is extrapolation rather than fitting.">off the training manifold</abbr>; feature activations land in regions the classifier head still maps to large-margin logits (the head is a linear map — it extrapolates confidently by construction; softmax then saturates). The miscalibration is no longer a uniform logit inflation — it grows with severity and varies by corruption type — so no single T fitted on clean data can be right for the whole family of shifted distributions. The clean diagnostic: fit an **<abbr title="The best possible T for a shifted distribution, found by refitting on its own labels. Not deployable: a diagnostic showing how far the correction moved.">oracle temperature</abbr> per severity** — you'll find T*(severity) increases monotonically, which is the direct exhibit that one scalar cannot serve all severities. Did you run it? `[FILL: oracle-T per severity, or name it as the missing experiment]`.

**What actually helps under shift** (know the landscape, one line each): <abbr title="Train several models with different random seeds and average their predictions. Their disagreement carries uncertainty signal a single model lacks.">deep ensembles</abbr> remain the strongest simple baseline for shift-robust uncertainty (Lakshminarayanan et al. 2017, *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*, https://arxiv.org/abs/1612.01474 — and the best performer in Ovadia et al.'s study); MC dropout is cheaper and weaker; <abbr title="Selective prediction: the model abstains below a confidence threshold instead of answering, trading coverage for correctness. Reported as a risk-coverage curve.">selective prediction/abstention</abbr> converts bad calibration into controlled <abbr title="The fraction of inputs the model chooses to answer rather than abstain on.">coverage</abbr>; <abbr title="Monitoring whether inputs still resemble the training distribution, using a signal independent of the model's own confidence.">shift detection</abbr> (feature-space distance, energy scores) flags when confidence shouldn't be trusted at all.

---

## 🔷 The Six Hardest Follow-Ups (With Model Answers)

### 🔷 Follow-up 1: "Define ECE exactly as you computed it. Now tell me three ways that number lies."

**Why they ask:** quoting a metric you can't attack is Bronze; the attack is the AS bar.

**Model answer:** the formula with your exact configuration (`[FILL: M, scheme, n]`), then pathologies 1–3 from Refresher 1 — binning sensitivity, mass concentration in top bins, within-bin cancellation — with the fix you applied or would apply (equal-mass binning cross-check, reporting NLL/Brier alongside). Bonus: the degenerate always-predict-the-marginal classifier that scores near-zero ECE, proving low ECE ≠ useful model.

**The chain:** "what happened when you changed M?" → "how big are your top-two bins' populations?" → "is your severity-5 ECE difference even outside estimator noise?" (bootstrap the eval set for a CI — `[FILL or propose]`).

### 🔷 Follow-up 2: "Why are modern networks overconfident in the first place? Mechanism, not citation."

**Why they ask:** the project presumes miscalibration exists; understanding *why* separates users of the fact from owners of it.

**Model answer:** "Training with cross-entropy keeps rewarding sharper distributions on examples the model already classifies correctly — accuracy saturates but NLL keeps improving by inflating logit magnitudes, which is confidence overfitting. Guo et al. traced the effect empirically to capacity growth and reduced regularization: bigger models, less weight decay, batch norm — each correlates with worse calibration at equal or better accuracy. The signature that supports this story: miscalibration is close to a uniform logit-scale inflation, which is exactly why a single temperature largely fixes it in-distribution."

**The chain:** "would label smoothing prevent it?" (partly — it penalizes saturated softmax during training; interacts with T-fitting) → "does this apply to LLMs' token probabilities?" (directionally yes post-RLHF, differently pretrained — honest boundary: say what you know) → "why does *more* capacity worsen it?"

### 🔷 Follow-up 3: "Derive temperature scaling. Why NLL as the fitting objective, and when would vector scaling beat it?"

**Why they ask:** it's the derivation-depth check on the project's central tool.

**Model answer:** Refresher 2, items 1–3, delivered as one continuous argument: monotone transform → accuracy invariant; NLL smooth and proper vs. ECE piecewise-constant; one parameter suffices because in-distribution overconfidence is approximately uniform inflation. Vector scaling wins when miscalibration is class-heterogeneous (per-class temperature, e.g., imbalanced or fine-grained classes) — at the cost of possible accuracy change and validation overfitting. Your T: `[FILL]`.

**The chain:** "prove NLL in T is convex for fixed logits" (one-dimensional; composition of log-sum-exp — be able to sketch it) → "why fit on validation, not training?" (training logits are the overfit ones; T would come out near 1) → "what does T = 2.4 tell you about the model?"

### 🔷 Follow-up 4: "Give me the mechanism for why the clean-fitted T fails at severity 4 — not the observation that it does."

**Why they ask:** the resume's key phrase is "demonstrated *where* it fails"; the mechanism is what makes the demonstration yours.

**Model answer:** Refresher 3's mechanism paragraph: shifted inputs leave the training manifold; the linear head extrapolates to large-margin logits on off-manifold features; softmax saturates; the inflation is severity- and corruption-dependent, so it's no longer the uniform rescale that one scalar can undo. The exhibit: oracle-T grows with severity (`[FILL or propose]`) — a *family* of temperatures is needed, and clean-fitted T is the wrong member for every shifted distribution.

**The chain:** "if you could detect severity at test time, does per-severity T solve it?" (helps ECE, but severity isn't observable in production and corruption type matters too — this is shift detection in disguise) → "which of your four corruptions broke calibration worst, and why might that be?" (`[FILL]`; blur destroys features gradually, noise pushes off-manifold fast — reason from your data, don't recite) → "does this failure also apply to ensembles?" (they degrade too — Ovadia et al. — just more gracefully).

### 🔷 Follow-up 5: "Your accuracy collapsed while confidence stayed high. Quantify that decoupling from YOUR run — and defend that it isn't an artifact."

**Why they ask:** the phenomenon sentence is the resume's most quotable claim; they want it to be load-bearing.

**Model answer:** the numbers: accuracy and mean confidence per severity, ECE trajectory, and the reliability-diagram description at severity 1 vs. 5 — all `[FILL]`. Artifact defenses, volunteered: estimator checks (two binning schemes agree directionally), sample size per severity cell (`[FILL: n]`), and the sanity anchor that clean-set ECE matched literature values for your architecture family before any corruption was applied.

**The chain:** "plot in your head: mean confidence vs. severity and accuracy vs. severity — which curve's *shape* carries the finding?" → "per-corruption or averaged? What does averaging hide?" → "someone claims your JPEG-corruption result is just resolution mismatch with the model's training augmentation — how do you check?"

### 🔷 Follow-up 6: "You've shown confidence can't be trusted under shift. A medical-imaging team wants to deploy your classifier anyway. What do you tell them to build?"

**Why they ask:** converts the analysis into judgment; also the LP bridge (Insist on the Highest Standards, Earn Trust).

**Model answer:** "Three layers, cheapest first. **Selective prediction:** set a confidence threshold on the *calibrated* model and report the risk–coverage curve — the team chooses an operating point where the model abstains rather than guesses; under shift, abstention rate rising is the safety valve. **Shift detection:** monitor a feature-space distance or energy score against the training distribution; alarms decouple 'model is confident' from 'model is in familiar territory' — which my project showed are different claims. **Ensembles if the budget allows:** the consistent best performer for shift-robust uncertainty in Ovadia et al.'s comparison, at N× inference cost. And the honest caveat: all of this manages miscalibration, none of it restores accuracy — if severity-4-like inputs are expected in production, the right fix is training-time (augmentation with realistic corruptions), not post-hoc."

**The chain:** "how do you pick the abstention threshold without shifted labels?" (conservative: from clean validation + safety margin; better: a small labeled shifted set — say what you'd pay for) → "what's the monitoring dashboard?" ⚠️ JD-DEPENDENT → "when is ECE the *wrong* metric for this team entirely?" (asymmetric error costs — a calibrated 90% means something different when false negatives kill; expected-cost curves beat ECE).

---

## 🔷 Hands-On Lab: Instrument Reps

30 minutes.

1. On paper: write the ECE formula, then compute it by hand for a toy set of 10 predictions in 2 bins. Check yourself in the [Calibration & Temperature Lab](02_calibration_and_temperature_lab.ipynb), which implements the same estimator from scratch.
2. Fill the Session 2 calibration rows of the Metric Vault in [PROGRESS.md](../../PROGRESS.md): architecture, dataset, per-severity accuracy/ECE, fitted T, oracle-T if run. Empty → `QUALITATIVE-ONLY`.
3. Run the lab's transfer-failure section and describe — out loud, one minute — why the with-TS curve and without-TS curve converge as severity grows. If your explanation uses the word "because" fewer than twice, it isn't a mechanism yet.

## 🟢 Concept Check

A team fits temperature on clean validation data and reports "our model is now calibrated." Which single experiment most directly tests whether that claim survives deployment drift?

* [ ] Refitting T weekly
* [x] Evaluating ECE (and oracle-refit T) on graded corrupted/shifted copies of the eval set — if oracle T grows with severity, the clean-fitted calibration provably degrades off-distribution
* [ ] Increasing the bin count
* [ ] Switching from ECE to accuracy

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The graded-corruption protocol is a controlled dial for distribution shift; tracking both ECE-under-fixed-T and the oracle-refit T across the dial shows not just *that* calibration degrades but *why* (the needed correction moves and one scalar can't follow). Weekly refits (option 1) help only if you have labeled production data — which is the expensive thing you usually lack.
</details>

---

## 🟢 Summary & Resources

- ECE is an estimator with a configuration, not a constant of nature — quote it with bins and scheme, cross-check with equal-mass binning, and pair it with proper scoring rules.
- Temperature scaling: monotone ⇒ accuracy-invariant; NLL-fitted; works in-distribution because overconfidence is approximately uniform logit inflation.
- Under shift the inflation stops being uniform — off-manifold features meet a linear head that extrapolates confidently — so oracle-T grows with severity and no clean-fitted scalar can serve the family.
- The deployment answer is layered: selective prediction, shift detection, ensembles — manage miscalibration, and be honest that none of it restores accuracy.

**References:** Guo et al. 2017 (arXiv:1706.04599, building on Naeini et al. 2015, AAAI) · Hendrycks & Dietterich 2019 (arXiv:1903.12261) · Ovadia et al. 2019 (arXiv:1906.02530) · Lakshminarayanan et al. 2017 (arXiv:1612.01474)

**Next:** [Lesson 3 — Tiered Challenges & Boss Fight: Systems Bar Raiser](03_tiered_challenges_and_boss_fight.md)
