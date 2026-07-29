# Session 4 — Chapter Quiz Bank

| | |
|---|---|
| **Prerequisites** | Lessons 1–5 |
| **Time** | ~40 min |
| **Rules** | Closed notes, paper to hand. Derive each answer out loud before selecting — this session's questions have exact answers, and recognizing one is not the same as producing it. |

12 quiz questions plus 2 reflection prompts. Nothing here is scored; the bank exists to find the derivations you *think* you have before an interviewer finds them.

---

## 📝 Chapter Quiz

**Q1.** For a linear layer $z = Wx + b$ with incoming gradient $\delta_z$, what is $\partial L/\partial W$ and why does it have that form?

* [ ] $W^\top \delta_z$ — a matrix product with the transposed weights
* [x] $\delta_z x^\top$ — an outer product, which is why it has exactly the same shape as $W$
* [ ] $\delta_z \odot x$ — an elementwise product
* [ ] $x^\top \delta_z$ — a scalar

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Each weight $W_{ij}$ connects input $j$ to output $i$, so its gradient is $\delta_{z_i} x_j$ — the outer product assembles exactly that. Option 1 is $\partial L/\partial x$, the *input* gradient, and confusing the two is the most common error here. Note what option 2 contains: $x$ itself, which is why forward activations must be stored until the backward pass arrives.
</details>

**Q2.** Why does a residual connection prevent vanishing gradients?

* [ ] It normalizes the gradient magnitude at each layer
* [x] The block's Jacobian becomes $I + \partial F/\partial h$, so an identity path always carries gradient undiminished — vanishing would require every path to decay
* [ ] It reduces the number of layers gradients pass through
* [ ] It clips gradients that grow too large

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The identity term is the whole mechanism. Option 1 describes normalization (a complementary but different fix) and option 4 describes clipping (which addresses *exploding*, not vanishing). The complementary framing worth adding: the block only learns a correction to identity, so a layer with nothing to contribute can output ≈0 harmlessly.
</details>

**Q3.** LayerNorm computes its statistics over which axis?

* [ ] The batch, for each feature
* [x] The features of a single token, independently per token — which is what makes it batch-size independent
* [ ] Both batch and features jointly
* [ ] The sequence dimension

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Option 1 is BatchNorm. The axis is the answer to nearly every normalization question: LayerNorm's per-token computation is exactly why the same token produces identical output alone or in a batch of 512 — verifiable in the Deep Learning Lab, where the difference is 0.00e+00.
</details>

**Q4.** Why is BatchNorm unsuitable for an autoregressive LLM?

* [ ] It has too many parameters
* [x] Decoding generates one token at a time so there is no meaningful batch, and BatchNorm's inference-time running averages are a different computation from training
* [ ] It cannot handle GPU training
* [ ] It only works for convolutional layers

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Three compounding reasons: variable-length sequences make batch statistics ragged, small batches make them noisy, and the train/inference mismatch is fatal during single-token generation. In the lab, BatchNorm at batch size 1 raises an error outright — variance over one example is zero.
</details>

**Q5.** The claim that BatchNorm works by reducing internal covariate shift is:

* [ ] Correct and well-established
* [x] Largely overturned — Santurkar et al. 2018 injected distributional noise *after* BatchNorm and training still improved; the better account is that it smooths the optimization landscape
* [ ] Correct only for CNNs
* [ ] Never claimed by the original paper

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The original paper (Ioffe & Szegedy 2015) *did* make the claim (so option 4 is wrong), which is why it became folklore. Being able to say the mechanism was tested and replaced is a strong senior signal, because most candidates repeat the 2015 story as settled.
</details>

**Q6.** RMSNorm differs from LayerNorm by:

* [ ] Normalizing over the batch instead of features
* [x] Dropping the mean subtraction, keeping only the RMS rescale — on the finding that re-scaling, not re-centering, carries the benefit
* [ ] Removing the learned scale parameter
* [ ] Applying normalization after the residual add

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** It keeps $\gamma$ (option 3 is wrong) and usually drops $\beta$ along with the mean. The payoff is one fewer statistic and one fewer parameter set — a few percent faster at scale with no measured quality cost, which is why Llama-family models adopted it. Option 4 describes Post-LN placement, a separate axis.
</details>

**Q7.** A model's context grows from 2k to 8k. Attention compute increases by roughly:

* [ ] 4×
* [x] 16× — attention is $O(n^2)$, so a 4× length increase quadruples twice over
* [ ] 8×
* [ ] It stays constant

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** $(8192/2048)^2 = 16$. Worth adding unprompted: the FFN is only $O(nd^2)$, so at short context the FFN dominates and attention takes over as $n$ grows — the crossover in the lab lands at $n = d_{ff}$, where attention reaches 50% of FLOPs.
</details>

**Q8.** Roughly how large is the KV cache for a 7B-class model (32 layers, 32 heads, $d_{head}$ 128, fp16) at 8k context, for one sequence?

* [ ] ~500 MB
* [x] ~4.3 GB — $2 \times 32 \times 32 \times 128 \times 8192 \times 2$ bytes
* [ ] ~40 GB
* [ ] It does not depend on context length

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The number that drives production decisions: batch 8 concurrent requests and the cache reaches ~34 GB, exceeding the ~14 GB of fp16 weights. That pressure is exactly what produced MQA and GQA. Being able to do this arithmetic live is worth more than naming the techniques.
</details>

**Q9.** GQA with 8 groups on a 32-head model:

* [ ] Reduces training compute by 4×
* [x] Shrinks the KV cache ~4× at close to MHA quality, while leaving query/output projections — and therefore training compute — essentially unchanged
* [ ] Reduces the number of query heads to 8
* [ ] Makes attention linear in sequence length

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** GQA reduces *key/value* heads, not query heads (option 3 is the classic mix-up). It is an inference-memory optimization; option 1 misstates what it saves. Option 4 confuses it with linear attention, which changes the math.
</details>

**Q10.** FlashAttention is best described as:

* [ ] A sparse approximation that skips low-scoring pairs
* [x] An exact, I/O-aware implementation — tiling plus an online softmax so the $n \times n$ matrix is never written to HBM
* [ ] A replacement of softmax with a linear kernel
* [ ] A quantization scheme for attention weights

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Exactness is the distinguishing property, and it is what separates FlashAttention from options 1 (sparse) and 3 (linear), both of which change the math and can cost quality. It performs *more* FLOPs than the naive version and is still 2–4× faster, because memory bandwidth — not arithmetic — was the binding constraint.
</details>

**Q11.** Your loss goes to NaN, and the failure reproduces exactly on the same batch with a fixed seed. This points at:

* [ ] Accumulated optimizer instability
* [x] A data or numerical edge case in that batch — a corrupted label, an all-padding sequence, or a fully-masked softmax row
* [ ] A learning rate that is too high
* [ ] Hardware failure

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Deterministic reproduction is the discriminating evidence. Instability that builds over thousands of steps depends on the whole trajectory and will not pin to one batch; a bad example will. Volunteering that test — "I'd re-run the same batch with the same seed" — is what a debugging question is really probing.
</details>

**Q12.** You cannot drive training loss near zero on a single batch of 8 examples. The most useful conclusion is:

* [ ] The learning rate needs tuning
* [x] There is a bug in the model or data pipeline — a healthy setup can memorize 8 examples, so this separates "bug" from "tuning" before touching any hyperparameter
* [ ] The model lacks capacity
* [ ] More data is needed

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The single-batch overfit test is the highest-value debugging move in deep learning because it *classifies* the entire problem space: pass means the machinery works and you have a tuning problem; fail means stop tuning and go find the bug. Options 3 and 4 are nearly always wrong at n=8.
</details>

---

## 🪞 Reflection Prompts

Reflection prompt 1 — *The arc.* Without notes, write the causal story connecting: vanishing gradients → residual connections → normalization → depth becomes practical → transformers scale → attention's $O(n^2)$ becomes the bottleneck → FlashAttention → the KV cache becomes the inference bottleneck → GQA. Mark every link where you had to guess.

<details>
<summary>🔑 Evaluation Criteria</summary>

A strong answer is one causal narrative, not nine facts. The load-bearing transitions: the gradient at layer $k$ is a *product* of Jacobians (hence exponential in depth); residuals change that Jacobian to $I + \partial F/\partial h$, guaranteeing an undiminished path; normalization then keeps per-layer scale controlled so large learning rates are stable; with depth solved the bottleneck moves from *trainability* to *cost*, where attention's quadratic term dominates at long context and FlashAttention removes the memory round-trips without changing the math; and at inference the binding cost changes again to the KV cache, which grows every token and is read in full per token — bandwidth-bound, which GQA addresses. Links you guessed are your Session 5 study list.
</details>

Reflection prompt 2 — *Your own model, quantified.* Take your from-scratch RoPE transformer from Session 1. Using its real configuration (`[FILL: layers / heads / d_model / d_head]`), compute: (a) its KV cache at 2k context, (b) what GQA with 2 groups would save, and (c) whether attention or the FFN dominates its FLOPs at your training sequence length.

<details>
<summary>🔑 Evaluation Criteria</summary>

The point is instantiating general arithmetic on a model you actually built — which is what an interviewer asks immediately after you claim you wrote a transformer from scratch. A good answer shows the formula, substitutes your real numbers, and lands on a conclusion: at a small $d_{model}$ and short sequences the FFN almost certainly dominates, and your KV cache is negligible — so GQA would buy you nothing at your scale. Saying *that* — "this optimization doesn't apply to my model, and here's the arithmetic showing why" — is stronger than reciting GQA's benefits generically.
</details>

---

**Next:** [Mock Round — Deep Learning Breadth](05_mock_round.md), then Session 5.
