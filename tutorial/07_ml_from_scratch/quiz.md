# Session 7 — Chapter Quiz Bank

| | |
|---|---|
| **Prerequisites** | Lessons 1–5, including all four blank-editor drills |
| **Time** | ~40 min |
| **Rules** | Closed notes, blank editor beside you. For any question about an implementation, **write the line** before selecting — this session is about what you can produce, and recognising the right option is not the same as typing it. |

12 quiz questions plus 2 reflection prompts. Together with the twelve concept-check questions inside the lessons, this chapter carries **24 distinct questions**. Nothing here is scored; the bank exists to find the implementations you *think* you have.

---

## 📝 Chapter Quiz

**Q1.** In `scaled_dot_product_attention`, you write `k.T` instead of `np.swapaxes(k, -1, -2)` on a 4-D array of shape `(B, H, T, d_head)`. The result is:

* [ ] Identical — `.T` transposes the last two axes
* [x] Wrong — `.T` reverses **all** axes, giving `(d_head, T, H, B)`, so the matmul either fails or silently contracts over the batch dimension
* [ ] Slower but correct
* [ ] Correct only when `B == 1`

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** This is a real bug people write under time pressure, because `.T` *is* right for the 2-D case they practised on. Option 4 is the tempting near-miss: even at `B == 1` the head axis is still reversed against the sequence axis, so it is wrong there too. `np.swapaxes(k, -1, -2)` — or `k.transpose(0, 1, 3, 2)` — is the form that generalises.
</details>

**Q2.** You scale attention scores by $d_k$ instead of $\sqrt{d_k}$. The effect is:

* [ ] Identical after training, since the model can rescale $W_Q$
* [x] Over-correction — scores shrink toward zero, the softmax approaches uniform, and attention loses its ability to select; the correct normaliser is the **standard deviation** $\sqrt{d_k}$, matching the units of the score
* [ ] The softmax saturates toward one-hot
* [ ] Gradients explode

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The derivation fixes the exponent: $\mathrm{Var}(q\cdot k) = d_k$, so the *standard deviation* is $\sqrt{d_k}$ and that is what restores unit scale. Option 3 is the failure mode of **no** scaling, so it is the right answer to a different question — worth being able to distinguish instantly. Option 1 contains a half-truth: the model can partly compensate, but the damage is done at initialisation, which is when it matters.
</details>

**Q3.** Converting `(B, T, d_model)` to multi-head form, the reshape must split:

* [ ] The time axis, into `(H, T/H)`
* [x] The **feature** axis, into `(H, d_head)` — then a transpose moves heads next to batch, giving `(B, H, T, d_head)`
* [ ] The batch axis, into `(B/H, H)`
* [ ] Nothing; a single transpose is enough

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Option 1 is the specific catastrophe: splitting time would give different heads different *positions* rather than different feature subspaces, silently scrambling the sequence. Option 4 is the second bug — the transpose alone cannot create the head axis, and reshaping alone cannot order it correctly. Both operations, in that order, and reversed for the merge.
</details>

**Q4.** RoPE produces relative position because:

* [ ] The angles are subtracted between positions before the dot product
* [x] Rotations are **orthogonal**, so $\langle R_m q,\, R_n k\rangle = \langle q,\, R_{n-m} k\rangle$ — the score depends on $n-m$ alone, never on $m$ and $n$ separately
* [ ] The frequencies decay geometrically with dimension
* [ ] It is applied after the softmax

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** One line of linear algebra is the entire justification for the method, and being able to state it separates having implemented RoPE from having imported it. Option 3 is true but answers a different question — the geometric frequency ladder controls the *range* of positions representable, not the relative-position property. A useful corollary: because rotations preserve norms, RoPE cannot rescale the scores the way an additive embedding can.
</details>

**Q5.** Subtracting the maximum inside `logsumexp` is:

* [ ] An approximation that is accurate to about $10^{-7}$
* [x] **Exact.** $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$ for any $m$ — the $e^{-m}$ factor cancels identically, so this is algebra rather than a tolerance
* [ ] Only valid when all logits are positive
* [ ] Required only in float16

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The word *exact* is the one to use out loud — candidates who describe it as "a trick for stability" often cannot say whether it changes the answer. It does not. And option 4 understates the range: `exp` overflows above roughly **88.7 in float32**, which untrained or diverging logits reach easily.
</details>

**Q6.** Cross-entropy is preferred over squared error for classification largely because its gradient with respect to the logits is $p - y$, which:

* [ ] Is always positive, simplifying optimization
* [x] Is bounded in $[-1, 1]$ and **does not vanish when the model is confidently wrong** — whereas squared error on a softmax carries an extra $p(1-p)$ factor that goes to zero exactly in that case
* [ ] Requires no softmax in the forward pass
* [ ] Guarantees a convex objective for any network

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The confidently-wrong case is the whole argument, and it is the version to say aloud: the gradient you most need is precisely the one MSE destroys. Option 4 is a trap worth rejecting explicitly — cross-entropy is convex in the *logits* of a linear model, and not remotely convex in the parameters of a deep network.
</details>

**Q7.** A finite-difference gradient check should use central differences in float64 because:

* [ ] float64 is faster for small arrays
* [x] The central difference has $O(h^2)$ truncation error against the forward difference's $O(h)$, and float32 differencing noise alone can produce ~$10^{-3}$ relative error on a perfectly correct gradient
* [ ] Central differences work at kinks where forward differences fail
* [ ] float64 avoids the need to choose $h$

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Both halves are load-bearing, and both are the reason a check "fails" on correct code. Option 3 inverts the truth in an instructive way: at a kink — a ReLU exactly at zero, or a `max` whose winner flips between $z+h$ and $z-h$ — the derivative does not exist and *every* differencing scheme fails legitimately. The fix is to perturb away from the kink, not to change the gradient.
</details>

**Q8.** The numerically stable binary cross-entropy from logits is $\max(z,0) - zy + \log(1+e^{-|z|})$. The $-|z|$ in the exponent matters because:

* [ ] It makes the loss symmetric in $y$
* [x] The exponent is then never positive, so `exp` can only **underflow toward zero** and never overflow — the $\max(z,0)$ term has already extracted the dominant part algebraically
* [ ] It cancels the $\max(z,0)$ term
* [ ] It is required for the gradient to be $p - y$

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** "Underflow is safe, overflow is not" is the compact statement of the whole design — an underflowing term contributes nothing to the sum, while an overflowing one destroys it. `np.log1p` completes the job by preserving precision when that exponential is tiny, where `log(1 + x)` would lose it to rounding.
</details>

**Q9.** Computing k-means distances as $\|x\|^2 - 2x^\top c + \|c\|^2$ instead of by broadcasting:

* [ ] Is mathematically different and gives different clusters
* [x] Gives the same result with `(N, k)` memory instead of `(N, k, D)` and a BLAS matmul — but it is a difference of large similar numbers, so cancellation can yield small **negative** squared distances that must be clipped
* [ ] Is slower but more numerically accurate
* [ ] Only works for normalized data

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The memory argument is what makes it necessary rather than clever: at $N=10^6$, $k=100$, $D=128$ the broadcast intermediate is about **51 GB**. Naming the precision cost alongside the speed gain is what makes this a trade-off answer rather than a tip — and clipping at zero, or centring the data, is the fix.
</details>

**Q10.** k-means++ seeds centroids with probability proportional to $D(x)^2$. Its guarantee is that:

* [ ] Lloyd's algorithm then converges to the global optimum
* [x] The expected inertia **of the seeding alone**, before any Lloyd iteration runs, is within $O(\log k)$ of the global optimum — uniform seeding has no such bound
* [ ] Convergence takes at most $k$ iterations
* [ ] Empty clusters cannot occur

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The phrase *before a single iteration runs* is what makes the guarantee striking, and it is the version worth saying. Option 1 is the overclaim to avoid: the objective is non-convex and finding its global optimum is NP-hard, so restarts still help — k-means++ makes each restart better, it does not make restarts unnecessary. Empty clusters (option 4) remain possible and still need an explicit policy.
</details>

**Q11.** At each beam-search step you should take the top *b* candidates:

* [ ] Per beam, so every beam survives and diversity is preserved
* [x] **Globally**, across all beams' expansions pooled together — per-beam selection cannot let one strong beam contribute two continuations, which is much of the point of beam search
* [ ] By sampling proportional to probability
* [ ] Only among beams that have not emitted EOS, discarding the rest

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Option 1 is a real and weaker algorithm rather than an implementation detail — it forces exactly one survivor per beam and cannot concentrate the search where the probability mass actually is. Option 4 contains a genuine bug from Lesson 4: finished beams must be **retired** to a separate pool, not discarded, or you only ever return sequences that ran to `max_len`.
</details>

**Q12.** The strongest available test of a beam-search implementation is:

* [ ] Checking the output looks fluent
* [x] On a toy vocabulary — say $V=4$, length 4, so 256 sequences — **enumerate every sequence** and assert a sufficiently wide beam finds the true argmax; this checks the whole search against ground truth
* [ ] Confirming it runs without errors on a long input
* [ ] Comparing it against a larger beam width

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Decoders are unusually verifiable and almost nobody exploits it. The companion identity is beam width 1 equalling greedy decoding *exactly* — cheap, but it cannot catch anything multi-beam, which is why the enumeration test earns its place. Option 4 is circular: a larger beam is a different heuristic, not ground truth, and for open-ended generation it is not even reliably better.
</details>

---

## 🪞 Reflection Prompts

Reflection prompt 1 — *Audit your own transformer against what you just wrote.* Open your ROPE project's attention code beside a blank file. Re-implement multi-head attention with RoPE from scratch, without looking, then diff the two. Answer: (a) which layout did your original use — interleaved pairs or split-half — and did you remember correctly before checking? (b) does your original apply RoPE to V anywhere? (c) is there a single assertion in that file that would fail if the implementation were wrong? (d) what were the exact layers, heads, and $d_{model}$ — Claim Vault #8 marks these as things you must know cold.

<details>
<summary>🔑 Evaluation Criteria</summary>

This prompt exists because Claim Vault #8 and #9 are the only résumé claims an interviewer can verify **in the room**. Everything else is defended with words; these are defended by typing.

A strong outcome is not "my code was perfect". It is knowing precisely what your code does — including anything you would now write differently. If the diff reveals a bug, that is a *gift*: a candidate who says "I re-implemented it recently and found that I'd applied the rotation with the split-half convention while my comment said interleaved" is demonstrating exactly the self-auditing behaviour the Dive Deep principle describes. A candidate who has never re-opened the file is one follow-up away from being exposed.

Part (c) is the one most likely to return "no", and that is the finding. Most research code contains no test that could distinguish a correct implementation from a plausible one — which is precisely why this session teaches property-based tests that cost one line. Adding the leakage test to your own repository is a genuine improvement you can make this week, and `[FILL: metric]` any figure you cannot source.
</details>

Reflection prompt 2 — *Close a piece of Gap #5 by hand.* Your audit scores ML breadth **3/6** and flags everything outside NLP as untested. Pick one classical model you have **never implemented** — a decision tree with its splitting criterion, naive Bayes, PCA via the SVD, or a GMM with EM. Implement it in NumPy in under an hour. Then answer: (a) what objective does it optimise, stated as a formula? (b) what property-based test proves your implementation correct, in the style of `assert monotone inertia`? (c) where does it fail, and what replaces it there? (d) can you now say you "know" it — and what would you say instead if asked in an interview?

<details>
<summary>🔑 Evaluation Criteria</summary>

Part (b) is the real exercise. Every model in this session had a free property test — inertia never increases, EM's log-likelihood never decreases, PCA's components are orthonormal and the reconstruction error falls monotonically with rank, a decision tree's chosen split must reduce impurity. Finding that property *is* understanding the objective, which is why the question is phrased this way.

Part (d) matters for honesty. One implementation does not make you an expert, and claiming it will lose the follow-up. The defensible sentence is specific: "I've implemented EM for a Gaussian mixture from scratch, so I can derive the E and M steps and I know the log-likelihood is monotone by construction — I haven't used it at production scale." That is a stronger answer than a vague claim of familiarity, and it is unfalsifiable in the good sense: it is true.

Do this once a fortnight against the runway in [Chapter 5](../../dossier/05_gap_map_and_study_plan.md) and Gap #5 closes as evidence rather than as reading. Never claim a model you have not run.
</details>

---

**Next:** [Mock Round — The 45-Minute ML Implementation Round](05_mock_round.md) if you have not run it, then [Session 8 — Behavioral: Leadership Principles & STAR](../08_leadership_principles_star/00_locked.md) *(locked)*.
