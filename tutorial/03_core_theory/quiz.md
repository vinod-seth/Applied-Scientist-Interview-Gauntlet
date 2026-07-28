# Session 3 — Chapter Quiz Bank

| | |
|---|---|
| **Prerequisites** | Lessons 1–4 |
| **Time** | ~35 min |
| **Rules** | Closed notes. Say each answer out loud before selecting — a breadth round is spoken, and the gap between "I recognize it" and "I can say it" is what this bank finds. |

12 quiz questions plus 2 reflection prompts. This bank is not a score to collect — it exists to surface the fundamentals you *think* you know cheaply, before an interviewer surfaces them expensively.

---

## 📝 Chapter Quiz

**Q1.** In the bias–variance decomposition for squared error, you double your model's capacity and observe that training error falls while validation error rises. Which term is growing?

* [ ] Bias²
* [x] Variance — the fit is now more sensitive to the particular training sample, so it tracks noise that doesn't generalize
* [ ] The irreducible noise σ²
* [ ] All three grow together

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Rising capacity reduces bias (the model class can now express the truth) and raises variance (the fit wobbles more across samples). A widening train–validation gap is the signature of variance. σ² is a property of the data-generating process and never changes with the model — the most common wrong answer here is picking it because "noise" sounds like the thing that grows.
</details>

**Q2.** An L2 penalty $\lambda\lVert w\rVert^2$ corresponds to which Bayesian object, and what does increasing $\lambda$ mean?

* [ ] A Laplace prior; increasing λ widens it
* [x] A zero-mean Gaussian prior; increasing λ *tightens* it (smaller prior variance τ², since λ = σ²/τ²)
* [ ] A uniform prior; λ has no Bayesian meaning
* [ ] The likelihood term, not the prior

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** $\log p(w)$ for $w\sim\mathcal N(0,\tau^2 I)$ is $-\lVert w\rVert^2/2\tau^2$ + const, which is exactly the L2 penalty with $\lambda = 1/2\tau^2$ (or $\sigma^2/\tau^2$ once you fold in the Gaussian likelihood's noise term). A **larger** λ is a **narrower** prior — a stronger belief that weights are near zero. Getting the direction backwards is the trap; the Core Theory Lab verifies the correspondence numerically.
</details>

**Q3.** You have 40 training examples and 500 features, and you want an interpretable model that identifies which handful of features matter. Which regularizer, and why?

* [ ] L2, because it's more stable
* [x] L1 — its Laplace prior has a sharp peak at zero, so the MAP solution sets most coefficients *exactly* to zero, giving feature selection
* [ ] Neither; use early stopping
* [ ] Both are identical in effect here

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2 (L1).** L2 shrinks coefficients smoothly toward zero but essentially never *to* zero, so all 500 features survive and interpretability is lost. L1's constraint region is a diamond whose corners lie on the axes, so the optimum typically lands on a corner where coordinates are exactly zero. Caveat worth volunteering: L1 is unstable among *correlated* features (it picks one arbitrarily) — elastic net exists for that reason.
</details>

**Q4.** "Minimizing cross-entropy" and "maximum likelihood estimation" are related how?

* [ ] They are different objectives that happen to have similar optima
* [x] They are the same objective — cross-entropy is the negative log-likelihood of the categorical model
* [ ] Cross-entropy is MLE plus a regularization term
* [ ] MLE applies only to regression; cross-entropy only to classification

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** $-\log p(y\mid x,w)$ summed over the data *is* cross-entropy for a softmax model, and *is* MSE (up to constants) for a Gaussian output model. So choosing a loss is choosing an output noise distribution — a framing that impresses because most candidates treat losses as an arbitrary menu. Option 3 describes MAP, not MLE.
</details>

**Q5.** *Mechanism.* Why does momentum help on an ill-conditioned ("ravine") loss surface?

* [ ] It increases the learning rate over time
* [x] It accumulates a velocity, so oscillating components across the steep direction partially cancel while the consistent flat direction accelerates
* [ ] It normalizes each parameter's gradient by its running second moment
* [ ] It reduces the number of parameters being updated

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Option 3 is *RMSProp/Adagrad* — the adaptive-learning-rate fix, which is a different mechanism attacking the same problem. Being able to distinguish these two ("momentum fixes oscillation, adaptivity fixes per-parameter scaling") is exactly what a follow-up probes. The Core Theory Lab shows momentum beating SGD roughly 6× at a matched learning rate.
</details>

**Q6.** A colleague sets `weight_decay=0.1` with plain Adam (L2 added to the loss) and reports it "makes no difference." What's happening?

* [ ] The value is too small to matter
* [x] The decay term enters the gradient and is then divided by $\sqrt{\hat v}$ — which is proportional to that same gradient — so the adaptive scaling normalizes the knob away
* [ ] Adam ignores the weight_decay argument entirely
* [ ] Weight decay only affects bias parameters

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** This is precisely the observation that motivated AdamW. In the Core Theory Lab you can watch the final weight stay identical to five decimal places as λ sweeps from 0 to 1.0 under L2-in-Adam, while AdamW's weight shrinks monotonically. The fix: decouple — apply $\eta\lambda w$ directly to the weights, outside the adaptive scaling.
</details>

**Q7.** Which statement about L2 regularization and weight decay is correct?

* [ ] They are equivalent for all optimizers
* [x] They are equivalent for plain SGD, but not for adaptive optimizers like Adam
* [ ] They are never equivalent
* [ ] They are equivalent only when the learning rate is constant

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** For vanilla SGD, adding $\lambda\lVert w\rVert^2$ to the loss produces exactly the update $w \leftarrow w - \eta(g + 2\lambda w)$, which is gradient step + shrinkage — the same thing. Adaptive optimizers break the equivalence because the L2 gradient gets rescaled by the second-moment estimate. This one-sentence distinction is a reliable senior-signal in an ML-breadth round.
</details>

**Q8.** A rare-event classifier (prevalence 0.5%) has ROC-AUC 0.95 but users say nearly every alert is a false alarm. What is the explanation?

* [ ] The model is overfitting the training set
* [x] FPR divides by the huge negative class, so many false positives barely move it; precision divides by predicted positives, so the same false positives crush it. PR-AUC would have exposed this.
* [ ] ROC-AUC was computed incorrectly
* [ ] The model needs recalibration; ROC-AUC and precision always agree after calibration

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** This is the imbalance trap, reproduced in the Core Theory Lab: ROC-AUC ≈ 0.95 alongside PR-AUC ≈ 0.26 and ~6 false alarms per true hit at a realistic operating point. Calibration (option 4) is a different property entirely — a perfectly calibrated model can still have terrible precision when positives are rare. Always quote PR-AUC against its baseline, the prevalence.
</details>

**Q9.** You standardize features using statistics computed over the entire dataset, then run 5-fold cross-validation. Your reported score is:

* [ ] Unbiased, since standardization uses no label information
* [x] Optimistically biased — the scaler absorbed distributional information from the held-out folds, so each fold is no longer truly unseen
* [ ] Pessimistically biased
* [ ] Unaffected, because standardization is a linear transform

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Leakage does not require labels. Any quantity *fitted* on data — scaler statistics, feature selection, vocabulary, imputation values, a calibration temperature — must be fitted inside the training folds only. That "unsupervised steps are safe" intuition (option 1) is exactly what makes this leak so common in real pipelines.
</details>

**Q10.** You are predicting customer churn from a table with many rows per customer. What split design is correct?

* [ ] Random row-level k-fold, stratified by churn
* [x] Group split by customer — otherwise the same customer appears in train and test and the model memorizes the entity rather than learning churn
* [ ] Any split; rows are exchangeable
* [ ] Leave-one-out cross-validation

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** This is the same failure as QQP's transitivity leakage from your own project: random pair-splits let duplicate clusters straddle train and test. The rule generalizes — **split by the unit you must generalize to.** New customers → split by customer; future traffic → split by time.
</details>

**Q11.** In your ESCI work you reported macro-F1 rather than accuracy. What does macro-F1 still fail to tell you on its own?

* [ ] Whether the classes are imbalanced
* [x] *Which* class collapsed — it averages per-class F1 equally, so a single failing class lowers the mean without identifying itself
* [ ] Whether the model beats a random baseline
* [ ] The model's calibration

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Macro-F1 makes minority-class collapse *visible* (unlike accuracy or micro-F1), which is why it was the right choice — but it is still a single averaged number. Always carry the per-class F1 table alongside it. Calibration (option 4) is genuinely also invisible to macro-F1, but it is invisible to every threshold metric, so option 2 is the sharper answer to "what does *this* metric hide."
</details>

**Q12.** *Judgment.* A stakeholder asks "is the model good?" What is the first thing you establish before naming any metric?

* [ ] The size of the test set
* [x] What decision the number will drive, and who pays for each error type — the cost structure determines which metric is honest
* [ ] Which model architecture was used
* [ ] Whether the training loss converged

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Metric selection is a statement about asymmetric costs. Screening for a serious disease is recall-dominated; paging a human on-call is precision-dominated; acting on predicted probabilities additionally requires calibration, which no threshold metric can see. Leading with the cost structure — rather than reciting a metric list — is the difference between a junior and a senior answer.
</details>

---

## 🪞 Reflection Prompts

Reflection prompt 1 — *The chain.* Without notes, write the causal chain that connects model capacity to a number you can trust: bias–variance → regularization → likelihood/MAP → optimizer → validation design. Then mark every link where you had to guess.

<details>
<summary>🔑 Evaluation Criteria</summary>

A strong answer names each link *and its connective tissue*, not five isolated definitions: capacity sets the bias–variance position; regularization controls capacity by encoding a prior; that prior turns MLE into MAP, so "regularized loss" and "MAP estimate" are one object; the optimizer merely walks that penalized objective (with the schedule often mattering more than the choice); and validation design decides whether the resulting number means anything — leak in the split and everything upstream is fiction. Links you had to guess are your study list for Session 4.
</details>

Reflection prompt 2 — *Your own evidence.* For each of your four projects, name one Session 3 fundamental it demonstrates, and one it exposed you on. Be specific: "ESCI → macro-F1 under class imbalance (demonstrated); regularization choices I never ablated (exposed)."

<details>
<summary>🔑 Evaluation Criteria</summary>

A good response ties abstractions to artifacts you can defend: QQP transitivity leakage as a grouped-split lesson; calibration-under-shift as validation-distribution mismatch; QLoRA/AdamW as decoupled weight decay in practice; ESCI as metric-selection under imbalance. The "exposed" column matters more than the "demonstrated" one — an interviewer asking "what would you do differently?" is the most common closing question in a breadth round, and a specific, honest answer outperforms a polished non-answer every time.
</details>

---

**Next:** [Rapid-Fire Rehearsal & Mock Breadth Round](04_rapid_fire_and_mock_round.md) — then Session 4.
