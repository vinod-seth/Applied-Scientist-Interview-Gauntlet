# Lesson 1 — Defending <abbr title="Quantized Low-Rank Adaptation: combines 4-bit NF4 quantization of the frozen base model with LoRA adapters, enabling fine-tuning of large models on a single consumer GPU.">QLoRA</abbr> on <abbr title="Shopping Queries Dataset (Exact, Substitute, Complement, Irrelevant): a large-scale Amazon benchmark for product-search relevance with four ordinal labels across multiple locales.">ESCI</abbr>: <abbr title="Mapping continuous values (e.g., model weights) to a smaller set of discrete levels, reducing storage and sometimes compute cost at the price of precision.">Quantization</abbr>, Memory, and Metrics

| | |
|---|---|
| **Prerequisites** | Session 1 README; your QLoRA run logs open in another window |
| **Reading time** | ~25 min + drills |
| **Domain tag** | <abbr title="Parameter-Efficient Fine-Tuning: a family of methods (LoRA, adapters, prompt tuning, etc.) that update only a small fraction of a model's parameters, drastically reducing memory and compute.">PEFT</abbr> / <abbr title="Natural Language Processing: the branch of AI focused on enabling machines to understand, generate, and reason about human language.">NLP</abbr> / Evaluation |
| **Resume line under fire** | "Fine-tuned <abbr title="Qwen2.5-1.5B: a 1.5-billion-parameter causal language model from the Qwen family, used here as the base model for QLoRA fine-tuning.">Qwen2.5-1.5B</abbr> with QLoRA (4-bit NF4 + <abbr title="Low-Rank Adaptation: learns a weight update as the product of two thin matrices instead of updating the full weight, dramatically reducing trainable parameters.">LoRA</abbr> adapters), training <2% of parameters on a single 16GB GPU" |

🔬 **Interactive companion:** [▶ Open the Memory Budget Verifier notebook in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/01_finetuning_and_architecture/01_memory_budget_verifier.ipynb)

> 📍 **Roadmap:** 🟢 sections rebuild the defense-level intuition; 🔷 sections carry the derivations and the model answers. Everything here is technical-track — there is no non-technical audience in an AS loop.

## 🟢 Learning Objectives

- Reconstruct NF4 quantization and the QLoRA memory budget from first principles, without notes.
- Answer the six hardest follow-ups on this project at Applied Scientist depth.
- Identify which claims in your own pitch are backed by run logs (`[FILL]` slots) and which must stay qualitative.

---

## 🟢 The Interrogation Logic of This Bullet

Your resume line makes four factual claims: a model, a method, a parameter fraction, and a hardware ceiling. Each claim implies you made a *choice*, and every choice invites the same three questions: *why this, why not the alternative, what did it cost you?* The lesson is organized around those chains.

One rule before anything else: the numbers in your answers come from your logs. Where this lesson shows `[FILL: ...]`, pull the value from the Metric Vault in [PROGRESS.md](../../PROGRESS.md). If a slot is empty, your answer for that topic is qualitative — direction and mechanism, no digits. An honest "I'd have to check the log for the exact value" survives; an invented number ends the round.

---

## 🔷 Peer-Level Refresher 1: What NF4 Actually Is

Skip the "quantization compresses weights" framing — you need the mechanism.

**<abbr title="Placing quantization levels at the quantiles of an assumed distribution, so each level covers equal probability mass instead of equal numeric width.">Quantile quantization</abbr>.** A k-bit <abbr title="A function that maps continuous (or high-precision) values to a fixed set of discrete levels, reducing storage at the cost of precision.">quantizer</abbr> maps continuous values to 2^k levels. Uniform (<abbr title="4-bit integer format with evenly spaced levels — simple, but wastes levels where the weight distribution is sparse.">int4</abbr>) spacing wastes levels where the data is sparse. Pretrained weight tensors are empirically ~zero-mean normal, so <abbr title="NormalFloat4: QLoRA's 4-bit storage format whose 16 levels sit at standard-normal quantiles, matching how pretrained weights are actually distributed.">NF4</abbr> places its 16 levels at the *quantiles of a standard normal* — each bin holds equal probability mass — then normalizes levels into [−1, 1]. That is the sense in which Dettmers et al. call it "information-theoretically optimal": for a <abbr title="Standard normal distribution: a Gaussian with mean 0 and variance 1, the canonical bell curve. Pretrained weights empirically approximate this shape.">N(0,1)</abbr> input, equal-mass bins minimize expected quantization error among equal-probability-assignment quantizers. It is optimal *given the normality assumption* — say that caveat out loud in an interview; it converts a memorized phrase into an understood one.

**Blockwise scaling.** Weights aren't globally N(0,1); each block of 64 values is scaled by its <abbr title="Absolute maximum: the largest magnitude in a block, used to rescale the block into the quantizer's range so one outlier only distorts its own block.">absolute maximum (absmax)</abbr> before quantization. Small blocks isolate outliers at the cost of storing one <abbr title="32-bit floating point: full-precision format used for optimizer states and scaling constants where numerical accuracy matters most.">fp32</abbr> constant per block.

**<abbr title="Quantizing the per-block scaling constants themselves, cutting their overhead from ~0.5 to ~0.127 bits per parameter.">Double quantization</abbr>.** Those per-block constants themselves get quantized (<abbr title="32-bit floating point: full-precision format used for optimizer states and scaling constants where numerical accuracy matters most.">fp32</abbr> → 8-bit, in blocks of 256), cutting the constant overhead from ~0.5 to ~0.127 bits per parameter. Small, but it's the difference-maker at the memory ceiling — and knowing the actual number signals you read the paper, not a blog.

**Forward/backward.** NF4 is a *storage* format. Compute happens in <abbr title="bfloat16: a 16-bit float with the same exponent range as fp32 but fewer mantissa bits — the standard compute dtype for stable mixed-precision training.">bf16</abbr>: each block is dequantized on the fly, <abbr title="Matrix multiplication: the core linear-algebra operation in neural network forward passes, where weight matrices transform input activations.">matmul</abbr> runs in bf16, and the dequantized weights are treated as constants during <abbr title="Backpropagation: the algorithm that computes gradients of the loss with respect to every parameter by applying the chain rule backward through the network.">backprop</abbr>. No <abbr title="A trick that passes gradients through a non-differentiable op by pretending it was the identity. Needed for quantization-aware training, not here.">straight-through estimator</abbr> is needed — the base weights are frozen; gradients only exist for the LoRA parameters, and the input gradient ∂L/∂x flows through the dequantized Wᵀ exactly as through any frozen linear layer.

Citation: Dettmers et al. (2023), *QLoRA: Efficient Finetuning of Quantized LLMs* (https://arxiv.org/abs/2305.14314).

---

## 🔷 Peer-Level Refresher 2: LoRA Math and Why <2% Works

For a frozen weight W ∈ ℝ^{d×k}, <abbr title="Low-Rank Adaptation. Freezes the pretrained weights and learns each update as the product of two thin matrices — far fewer numbers to train.">LoRA</abbr> learns ΔW = (α/r)·B·A with A ∈ ℝ^{r×k} (<abbr title="Gaussian initialization: weights are drawn from a normal distribution. For LoRA's A matrix this provides a random starting direction so the product B·A is non-degenerate once B moves from zero.">Gaussian init</abbr>) and B ∈ ℝ^{d×r} (<abbr title="Zero initialization: all entries set to 0. LoRA's B matrix starts at zero so the initial update ΔW = B·A = 0, meaning training begins from the pretrained model exactly.">zero init</abbr>), so training starts at ΔW = 0. <abbr title="How thin the two learned matrices are. Larger r buys a more expressive update and more trainable parameters; too small cannot fit the task.">Rank r</abbr> caps the update's rank; <abbr title="A scaling factor that keeps the update's effective magnitude roughly constant as rank changes, so you need not retune the learning rate per rank.">α/r</abbr> is a scaling that decouples learning-rate tuning from r. Which modules you targeted (q,k,v,o only? all linear layers, as the QLoRA paper recommends?) is a `[FILL: target modules]` fact you must know cold.

Why can a rank-16 update adapt a 1.5B-parameter model? The standard evidence: fine-tuning objectives have low <abbr title="The number of directions in weight space that actually matter for a task. If it is small, two thin matrices can express the whole adaptation.">*intrinsic dimensionality*</abbr> — Aghajanyan et al. (2020), *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning* (https://arxiv.org/abs/2012.13255) — pretrained representations already span the needed features, and adaptation is mostly re-weighting, which lives in a low-dimensional subspace.

Know the honest limit too: LoRA is not free. Biderman et al. (2024), *LoRA Learns Less and Forgets Less* (https://arxiv.org/abs/2405.09673) show low-rank adapters underperform full fine-tuning when the target task demands genuinely new capabilities, while forgetting less of the base model. For a 4-class relevance task over text the base model already "understands", the <abbr title="Low-rank constraint: restricting the weight update to a matrix of rank r, which limits the number of independent directions the adaptation can use. Mild when the task's intrinsic dimensionality is small.">low-rank constraint</abbr> is mild — that's the argument for why your setup is a reasonable trade, and it's a better answer than "LoRA is just as good as full fine-tuning" (it isn't, in general).

Citations: Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models* (https://arxiv.org/abs/2106.09685).

## 🟢 Concept Check

Why does NF4 outperform uniform int4 on pretrained weights?

* [ ] NF4 uses more bits per weight in outlier blocks
* [x] NF4 places quantization levels at normal-distribution quantiles, matching the empirical weight distribution so each level carries equal probability mass
* [ ] NF4 stores weights in floating point rather than integers
* [ ] NF4 avoids blockwise scaling entirely

Why is no straight-through estimator needed when backpropagating through the quantized base model?

* [ ] Because 4-bit quantization is differentiable
* [ ] Because gradients are computed in int4
* [x] Because the base weights are frozen — gradients are needed only for LoRA parameters and the layer input, both of which flow through the *dequantized* bf16 weights as constants
* [ ] Because paged optimizers approximate the gradient

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Q1: option 2.** Uniform spacing wastes levels in low-density regions of a bell-shaped distribution. Equal-probability-mass bins minimize expected error under the (empirically justified) normality assumption. Outliers are handled separately, by small-block absmax scaling — not by extra bits.

**Q2: option 3.** Quantization is non-differentiable, but nothing requires differentiating *through* it here: the quantized weights receive no updates. Each forward/backward dequantizes blocks to bf16 and treats them as constants. This is the key difference from quantization-aware training, where an STE would be required.
</details>

---

## 🔷 Peer-Level Refresher 3: The 16GB Memory Budget, Line by Line

This is the single most likely deep-drill on this project, because it's checkable arithmetic. Derive it live, don't recite it. For Qwen2.5-1.5B (~1.5B params):

**Why full fine-tuning doesn't fit.** Standard <abbr title="Mixed-precision training: storing weights and computing forward/backward passes in lower precision (e.g., bf16) while keeping a full-precision (fp32) master copy for stable optimizer updates.">mixed-precision</abbr> <abbr title="AdamW: a variant of the Adam optimizer with decoupled weight decay. Maintains per-parameter first-moment (mean) and second-moment (variance) running averages, each requiring fp32 storage.">AdamW</abbr> keeps: bf16 weights (2 B/param ≈ 3.1 GB), bf16 gradients (≈ 3.1 GB), fp32 master weights (≈ 6.2 GB), fp32 first and second <abbr title="Moments (in optimizer context): running statistics the optimizer maintains per parameter — the first moment is the exponential moving average of gradients, the second moment is the EMA of squared gradients.">moments</abbr> (≈ 12.3 GB). That's ≈ 24.7 GB of state before a single activation — over 16 GB with nothing computed yet. Even an aggressive pure-bf16 setup leaves you nothing for activations at useful sequence lengths.

**The QLoRA budget.** NF4 base weights ≈ 0.78 GB (0.5 B/param) plus ~0.127 bits/param of quantization constants. LoRA adapters at `[FILL: rank/targets]` are on the order of tens of millions of parameters — call it `[FILL: adapter param count]` — and only *they* carry gradients and AdamW states, costing a few hundred MB total. What actually dominates now is **<abbr title="Intermediate tensors saved during the forward pass because backprop needs them. They scale with batch and sequence length, and dominate memory once weights are quantized.">activations</abbr>**, which scale with batch × sequence length × hidden size × layers; <abbr title="Storing only a few layer boundaries and recomputing the rest during the backward pass — trades roughly 30% extra compute for a large activation-memory saving.">gradient checkpointing</abbr> trades ~30% extra compute to store only block boundaries. **<abbr title="Optimizer states held in CUDA unified memory so they can spill to CPU RAM during spikes, preventing out-of-memory crashes rather than reducing steady-state usage.">Paged optimizers</abbr>** (the third QLoRA ingredient people forget) use unified memory to evict optimizer state to CPU RAM on spikes, preventing <abbr title="Out of Memory: a GPU crash that occurs when the total memory demand (weights + activations + optimizer states + gradients) exceeds the GPU's physical VRAM.">OOM</abbr> during long-sequence steps rather than reducing steady-state usage.

Your own observed peak — `[FILL: peak GPU memory]` — is the number that proves you watched `nvidia-smi` and didn't just run a notebook. If you never logged it, say so and describe what you'd expect; do not guess a figure.

---

## 🔷 Peer-Level Refresher 4: The Task Side — ESCI, the Classifier, and Macro-F1

**ESCI.** Reddy et al. (2022), *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search* (https://arxiv.org/abs/2206.06588). Labels are Exact / Substitute / Complement / Irrelevant for (query, product) pairs, in three locales (us/es/jp). Three properties an interviewer will probe: the distribution is heavily skewed toward Exact; the labels are *roughly ordinal* in relevance (E > S > C ≈ I) while <abbr title="Cross-entropy loss: the standard classification objective that measures the divergence between predicted class probabilities and the true one-hot label. Treats classes as unordered categories.">cross-entropy</abbr> treats them as unordered; and <abbr title="Split hygiene: ensuring training and test sets have no data leakage — no shared or near-duplicate examples that would inflate evaluation metrics.">split hygiene</abbr> matters — the same product (or near-duplicate query) appearing in train and test inflates results. Know which subset, locale(s), and split you used: `[FILL: data configuration]`.

**How a <abbr title="Causal language model: an autoregressive model that predicts each token conditioned only on preceding tokens, generating text left-to-right. GPT-family and Qwen are causal LMs.">causal LM</abbr> emits 4 classes.** You must be able to state your exact mechanism — this is *your* code. The two standard designs: (a) a <abbr title="Classification head: a linear projection layer placed on top of the language model's hidden states that maps the final representation to class logits (one per label).">classification head</abbr> over the hidden state of the **last non-padding token** (<abbr title="Causal attention (also: masked self-attention): each token can only attend to itself and earlier tokens, enforcing left-to-right information flow. Only the last token has seen the entire input.">causal attention</abbr> means only that token has seen the full pair), which is what `AutoModelForSequenceClassification` does; or (b) a <abbr title="Verbalizer: a technique that maps class labels to specific vocabulary tokens ('Exact', 'Substitute', etc.) and reads the model's next-token logits over those tokens as class scores — no extra parameters needed.">verbalizer</abbr> — score the <abbr title="Logits: the raw, unnormalized scores output by the model's final linear layer before softmax. Higher logit = higher predicted probability for that class or token.">logits</abbr> of four label tokens in a prompt template. Each has failure trapdoors: for (a), padding side and <abbr title="Pad-token configuration: causal LMs like Qwen often ship without a dedicated padding token. You must assign one (commonly eos_token) and configure padding side so batched sequences align correctly.">pad-token configuration</abbr> (Qwen ships without a pad token; if you set `pad_token = eos`, say so and why it's safe for classification); for (b), template sensitivity and multi-token labels. `[FILL: which design + config]`.

**<abbr title="The unweighted mean of per-class F1 scores. Every class counts equally however rare, so a collapsed minority class drags the average down.">Macro-F1</abbr>.** Unweighted mean of per-class F1 — each class counts equally regardless of <abbr title="The number of true examples of a class in the evaluation set.">support</abbr>, which is exactly why it's the right headline metric when Exact dominates: accuracy or <abbr title="Micro-F1: F1 computed by aggregating true positives, false positives, and false negatives across all classes. Dominated by the majority class — a model that nails Exact but misses everything else can still score high.">micro-F1</abbr> would let a majority-class-happy model look strong. Now the part that wins the exchange: say what macro-F1 *hides*. It hides *which* class is failing (Complement, usually, with the least data), it hides asymmetric error costs (E→I is a catastrophic search failure; E→S is mild, and macro-F1 charges the same), and it ignores the ordinal structure. Your per-class F1 breakdown — `[FILL: per-class F1]` — is the strongest artifact you can bring to this discussion.

## 🟢 Concept Check

Your model predicts Exact for almost everything and still shows 70%+ accuracy. Which metric property explains why macro-F1 catches this failure?

* [ ] Macro-F1 weights classes by their support
* [x] Macro-F1 averages per-class F1 with equal weight, so collapsed minority classes (each with F1 near 0) drag the average down hard
* [ ] Macro-F1 is threshold-free
* [ ] Macro-F1 penalizes confident wrong answers more than accuracy does

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct answer: option 2.** Equal class weighting is the whole point of "macro." A degenerate majority-class predictor scores high accuracy and high *micro*-F1 (which aggregates over instances), but near-zero F1 on S/C/I pulls macro-F1 toward 0.25 of its potential. The trade-off to mention unprompted: macro-F1 can *over*-reward tiny-class gains — a two-point jump on Complement moves the headline as much as two points on Exact, despite a fraction of the traffic. That cuts both ways in your comparison against DeBERTa, and noticing it is top-tier.
</details>

---

## 🔷 The Six Hardest Follow-Ups (With Model Answers)

Each entry: the question, why it gets asked, a model answer at AS depth, and the deeper chain behind it. Model answers are written in first person as *your* voice — replace every `[FILL]` with your logged value or drop to qualitative.

### 🔷 Question 1 (Follow-up): "Why NF4 and not int4 or <abbr title="FP4: a 4-bit floating-point format that uses an exponent/mantissa split to cluster levels near zero. Less principled than NF4 for normally distributed weights.">FP4</abbr>? You said 'information-theoretically optimal' — optimal in what sense?"

**Why they ask:** tests whether a resume keyword is understood or transplanted.

**Model answer:** "Pretrained weight matrices are empirically close to zero-mean normal. Uniform int4 spaces its 16 levels evenly, so most levels sit where weights are rare; FP4 clusters levels near zero but with an ad-hoc <abbr title="Exponent/mantissa split: in floating-point formats, the exponent sets the magnitude range and the mantissa (significand) sets precision within that range. FP4's split is ad-hoc for weight distributions.">exponent/mantissa split</abbr>. NF4 places levels at the quantiles of a standard normal, so each level covers equal probability mass — that minimizes expected quantization error for normally distributed inputs, which is the precise sense of 'optimal.' It's conditional on the normality assumption, and it's applied per 64-value block with absmax scaling so outliers only distort their own block. In my run the accuracy cost versus bf16 inference was `[FILL: comparison if measured / 'something I didn't isolate — I'd ablate NF4 vs int4 holding everything else fixed']`."

**The chain behind it (be ready):** derive why equal-mass bins are the error-minimizing choice → why blocks of 64 → what double quantization saves and where → when NF4 breaks (weight distributions with heavy outlier structure; why outlier-aware schemes like <abbr title="Int8 decomposition (LLM.int8()): a mixed-precision scheme that identifies outlier features and keeps them in fp16 while quantizing the rest to int8, handling heavy-tailed weight distributions NF4 cannot.">int8 decomposition</abbr> exist).

### 🔷 Question 2 (Follow-up): "Walk me through your 16GB. Where does every gigabyte go, in full fine-tuning versus your setup?"

**Why they ask:** the arithmetic is verifiable in real time; hand-waving is instantly visible.

**Model answer:** reproduce Refresher 3 as live arithmetic, on paper, ending with: "…so post-QLoRA the binding constraint isn't parameters, it's activations — which is why I ran `[FILL: seq length / batch size / grad-accum / checkpointing config]` and saw peak usage of `[FILL: peak memory]`."

**The chain:** why fp32 optimizer moments even in mixed precision (numerical stability of the second-moment <abbr title="Exponential Moving Average: a weighted running average that gives more weight to recent values. In Adam, both the first moment (gradient mean) and second moment (gradient variance) are EMAs.">EMA</abbr>) → what gradient checkpointing actually stores → what paged optimizers do and *when* they trigger → "could you have full fine-tuned a 0.5B model instead? Would that have been better?" (honest answer: maybe — and it's a fair <abbr title="Ablation: a controlled experiment where you remove or change exactly one factor to measure its isolated effect. The gold standard for understanding which design choices actually matter.">ablation</abbr> you can propose).

### 🔷 Question 3 (Follow-up): "How exactly does your <abbr title="Causal decoder: a Transformer that generates tokens autoregressively (left-to-right), using masked self-attention so each position only sees prior tokens. Repurposed here for classification.">causal decoder</abbr> produce a 4-class label? Walk me from input string to logit."

**Why they ask:** separates people who ran a script from people who own their pipeline.

**Model answer (head variant — adapt to yours):** "The query and product text go through a template `[FILL: template]`, <abbr title="Tokenization: converting raw text into a sequence of integer token IDs from the model's vocabulary. Subword tokenizers (like BPE) split unknown words into known pieces.">tokenized</abbr> to max length `[FILL]`. Causal attention means only the final non-padding token attends to the whole pair, so the classification head is a linear layer on that token's <abbr title="Last hidden state: the final-layer representation vector for a token. For classification, this vector is projected through the classification head to produce logits.">last hidden state</abbr>, trained with cross-entropy over 4 logits. Qwen has no native pad token, so I `[FILL: pad strategy]`; with right-padding you must index the last *real* token per sequence, which is a classic silent bug — the head reads padding otherwise."

**The chain:** why the last token and not <abbr title="Mean pooling: averaging all token representations to get a single sequence vector. Works well for bidirectional models but is problematic for causal models where early tokens have incomplete context.">mean pooling</abbr> (<abbr title="Causal masking: the attention mask pattern that prevents each token from attending to future tokens, preserving the autoregressive property. Early tokens see less context than later ones.">causal masking</abbr> makes earlier tokens partially-informed) → verbalizer alternative and when it wins (few-shot, no new parameters) → template sensitivity → "did you check the tokenizer splits product titles sanely in all locales?"

### 🔷 Question 4 (Follow-up): "Why compare a 1.5B decoder with adapters to a ~184M fully fine-tuned encoder? Is that comparison fair — and whichever won, why?"

**Why they ask:** experimental-design judgment is the core AS skill; this question has no memorizable answer.

**Model answer:** "It's deliberately a *practitioner's* comparison, not a controlled one: 'best achievable under a 16GB budget with a modern decoder + PEFT' versus 'the standard strong baseline everyone deploys for pair classification.' <abbr title="DeBERTa-v3: Decoding-enhanced BERT with disentangled Attention, version 3. A ~184M-parameter bidirectional encoder with ELECTRA-style pretraining, exceptionally strong for text-pair classification tasks.">DeBERTa-v3</abbr> as a <abbr title="Cross-encoder: a model architecture that concatenates both inputs (e.g., query + product) into a single sequence and encodes them jointly, allowing full bidirectional attention between the pair.">cross-encoder</abbr> has the structural advantage here — <abbr title="Bidirectional attention: every token attends to every other token in the sequence (past and future), giving each position full context. Used in BERT-family encoders but not causal decoders.">bidirectional attention</abbr> over the concatenated pair is purpose-built for this task, and <abbr title="ELECTRA-style pretraining: instead of masked-language-model prediction, the model learns to detect which tokens were replaced by a small generator. More sample-efficient than MLM for encoder pretraining.">ELECTRA-style pretraining</abbr> with <abbr title="Gradient-disentangled embedding sharing: DeBERTa-v3's technique for sharing embeddings between the generator and discriminator while stopping gradients from the generator, preventing training instability.">gradient-disentangled embedding sharing</abbr> makes it a very strong 184M (He et al. 2021, *DeBERTaV3*, https://arxiv.org/abs/2111.09543). The decoder brings scale and better raw language priors but pays twice: causal attention restricts pair interaction, and the low-rank constraint limits adaptation — consistent with the 'LoRA learns less' result. My result: `[FILL: both macro-F1 values]`, and the per-class gap concentrated in `[FILL: class]`. To make it a fair *scientific* comparison I'd need to hold data, sequence length, and tuning budget fixed and add QLoRA-on-DeBERTa and full-FT-on-a-small-decoder arms — I'd present my version as an engineering result, not a controlled <abbr title="Ablation: a controlled experiment where you remove or change exactly one factor to measure its isolated effect. The gold standard for understanding which design choices actually matter.">ablation</abbr>."

**The chain:** what <abbr title="Confounds: uncontrolled variables that differ between experimental conditions, making it impossible to attribute observed differences to the factor you intended to test.">confounds</abbr> remain (LR tuning per model, epochs, truncation) → "design the controlled version" → "if the decoder lost, why publish the comparison at all?" (because negative results under honest constraints are informative — and saying that is Earn Trust).

### 🔷 Question 5 (Follow-up): "Exact dominates ESCI. Beyond choosing macro-F1, what did you do about the imbalance — and what would you do next?"

**Why they ask:** metric choice is passive; they want an *intervention* and its trade-offs.

**Model answer:** "Measurement first: per-class F1 and the full <abbr title="Confusion matrix: an N×N table (N = number of classes) showing how many examples of each true class were predicted as each other class. Reveals which specific class pairs the model confuses.">confusion matrix</abbr> — mine showed `[FILL: confusion pattern / 'the S/C boundary as the dominant confusion' if true]`. Interventions, in the order I'd try them: <abbr title="Class-weighted cross-entropy: multiplies each class's loss contribution by a weight (typically inverse class frequency), forcing the model to pay more attention to rare classes.">class-weighted cross-entropy</abbr> (cheap, risks over-firing minority classes), <abbr title="Focal loss (Lin et al. 2017): modifies cross-entropy by down-weighting examples the model already classifies confidently, focusing training on hard/misclassified examples. Particularly effective for class imbalance.">focal loss</abbr> (down-weights easy majority examples), <abbr title="Minority oversampling: duplicating or synthetically generating examples of underrepresented classes so the model sees them more often. Risk: at small dataset scales, the model may memorize the repeated examples.">minority oversampling</abbr> (duplication risks memorization at 20K scale), and — the one specific to ESCI — exploiting the <abbr title="Ordinal structure: when class labels have a natural ordering (E > S > C > I in relevance). Standard cross-entropy ignores this order; ordinal objectives can penalize far-apart misclassifications more heavily.">ordinal structure</abbr>: E>S>C>I is a relevance gradient, so an <abbr title="Ordinal objective: a loss function that respects the natural ordering of labels, penalizing predictions farther from the true label more heavily (e.g., E→I costs more than E→S). Cumulative-link models and soft ordinal targets are common approaches.">ordinal objective</abbr> (e.g., <abbr title="Cumulative-link model: an ordinal regression approach that models the probability of exceeding each category threshold, naturally encoding the label ordering.">cumulative-link</abbr> or soft ordinal targets) penalizes E→I more than E→S, which plain <abbr title="CE (Cross-Entropy): shorthand for cross-entropy loss, the standard classification objective.">CE</abbr> can't. What I actually ran: `[FILL: what you did, honestly — 'none beyond the metric choice' is a legitimate answer if you then own the next step]`."

**The chain:** why <abbr title="Oversampling: repeating minority-class examples in the training set. In expectation it has the same effect as class-weighted loss, but it changes optimization dynamics because the model sees the same examples multiple times per epoch.">oversampling</abbr> ≈ weighted loss in expectation but differs in optimization dynamics → <abbr title="Threshold moving: adjusting the decision boundaries between classes post-training (e.g., lowering the threshold for predicting a minority class) to optimize a target metric like F1 without retraining.">threshold moving</abbr> vs. loss reweighting for F1 targets → does <abbr title="Label noise: errors in the ground-truth annotations, common in crowd-sourced datasets like ESCI. Sets an upper bound on achievable model performance regardless of architecture or training.">label noise</abbr> in crowd-sourced ESCI bound achievable F1? → "how would you *detect* that bound?" (<abbr title="Annotator-agreement analysis: measuring how often independent human annotators assign the same label to the same example. Low agreement indicates high label noise and bounds model performance.">annotator-agreement analysis</abbr>, if available; training on relabeled subsets).

### 🔷 Question 6 (Follow-up): "You trained <2% of parameters. What did that constraint cost you, and how do you know?"

**Why they ask:** the resume line advertises the efficiency; the bar is knowing its price.

**Model answer:** "The honest answer is I can bound it but didn't fully measure it. The evidence that low-rank costs something: Biderman et al. 2024 show LoRA underperforms <abbr title="Full fine-tuning: updating all model parameters during training, as opposed to PEFT methods that freeze most weights. Achieves maximum expressiveness but requires storing gradients and optimizer states for every parameter.">full fine-tuning</abbr> as target-task novelty grows. My task is 4-way classification over text the base model already models well, so the constraint should be mild — the adaptation is mostly reweighting existing features, which is the <abbr title="Intrinsic-dimensionality argument: the hypothesis (Aghajanyan et al. 2020) that fine-tuning objectives live in a low-dimensional subspace of weight space, so a low-rank adapter can capture most of the needed adaptation.">intrinsic-dimensionality argument</abbr>. What I'd run to measure it on my setup: <abbr title="Rank sweep: training the same model at multiple LoRA ranks (e.g., r=4, 8, 16, 32, 64) and comparing performance. A plateau indicates the task's intrinsic dimensionality is at or below that rank.">rank sweep</abbr> `[FILL: if you ran one — r values and macro-F1 curve]`; if macro-F1 plateaus by r=16, the constraint isn't binding. The measurable proxy I do have is the DeBERTa full-FT baseline: `[FILL: gap]`, though that confounds architecture with adaptation method."

**The chain:** what happens mechanically when r doubles (parameter count linear in r; expressible update rank doubles) → why α/r scaling exists → "would you rather double r or add <abbr title="Target modules: the specific layers in the model where LoRA adapters are inserted. Common choices: just attention (q,k,v,o) or all linear layers. The QLoRA paper found all-linear coverage outperformed higher rank on fewer modules.">target modules</abbr>?" (QLoRA paper: all-linear coverage beat higher rank) → <abbr title="Adapter merging: folding the LoRA update (ΔW = B·A) back into the base weight matrix at inference time. Eliminates adapter overhead, but if the base is quantized you must dequantize first, losing the memory savings.">merging adapters</abbr> into a quantized base at inference (must <abbr title="Dequantize: convert weights from the compact quantized format (e.g., NF4) back to a compute-ready format (e.g., bf16 or fp32). Required for matmul and for merging LoRA adapters.">dequantize</abbr>; you lose the memory benefit or keep the adapter separate).

---

## 🔷 Hands-On Lab: Build Your Pitch Menu

30 minutes, produces the artifact you'll use in the assessment and the real loop.

1. From your run logs, fill every QLoRA row of the Metric Vault in [PROGRESS.md](../../PROGRESS.md). Empty slots get marked `QUALITATIVE-ONLY`.
2. Write your 4-minute project pitch (see [round playbook](../../playbooks/round_playbook.md), beat structure: result → framing → decision tour → invite the drill).
3. Underline every technical term in the pitch. For each, self-assess Level 1 / 2 / 3 honestly. **Delete or downgrade every hook you cannot at least derive.** A pitch with six defensible hooks beats one with twelve fragile ones.
4. Say the pitch out loud, timed. Twice.

Expected output: a pitch of ≤ 4 minutes containing zero unfilled numbers and zero hooks you cannot defend. You'll verify it against the mock round in lesson 3.

## 🟢 Concept Check

You doubled LoRA rank from 16 to 32 and macro-F1 didn't move. What's the *best-supported* conclusion?

* [ ] The model is underfitting and needs a higher learning rate
* [ ] NF4 quantization error is the bottleneck
* [x] The adaptation needed for this task already fits in a rank-16 <abbr title="Subspace: a lower-dimensional subset of the full parameter space. If the task's useful adaptation lives in a rank-16 subspace, increasing rank beyond 16 adds no benefit.">subspace</abbr> — the low-rank constraint isn't the binding limitation
* [ ] Macro-F1 is insensitive to rank by construction

During training you notice loss is normal but eval predictions are all "Exact". Which bug is the *most likely* culprit in a causal-LM classification setup?

* [ ] LoRA alpha set too high
* [x] The classification head is reading a padding-token position instead of the last real token, so it sees near-constant input at eval batch sizes
* [ ] Double quantization corrupted the base weights
* [ ] The macro-F1 implementation is broken

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Q1: option 3.** A flat rank-response curve is direct evidence the update's intrinsic dimension is ≤ r. It says nothing about learning rate or quantization — those would depress both runs equally.

**Q2: option 2.** The classic silent failure: with right-padding and a naive "last position" index, the head reads pad embeddings whenever the batch's max length exceeds the sequence's. Train-time batching may mask it (similar lengths), eval batching exposes it. Constant input → constant (majority) prediction while training loss looks plausible. This is a debugging-instinct question: the answer that names a *mechanism* beats the answer that names a hyperparameter.
</details>

---

## 🟢 Summary & Resources

- NF4 = quantile quantization under a normality assumption, blockwise absmax, double-quantized constants; storage-only, compute in bf16, no <abbr title="STE (Straight-Through Estimator): a trick that passes gradients through a non-differentiable quantization step by pretending it was the identity function. Not needed here because quantized weights are frozen.">STE</abbr> because the base is frozen.
- The 16GB story is arithmetic you derive live: <abbr title="Optimizer states: the per-parameter bookkeeping maintained by the optimizer (e.g., AdamW's first and second moments), typically stored in fp32. For a 1.5B model, these alone consume ~12 GB.">optimizer states</abbr> kill full <abbr title="FT (Fine-Tuning): updating model weights on task-specific data. 'Full FT' means updating all parameters, as opposed to PEFT which updates only a small subset.">FT</abbr>; activations are QLoRA's real constraint.
- Your classifier mechanism, data configuration, and per-class F1 are *yours to know exactly* — they're the difference between owning the project and having run it.
- The DeBERTa comparison is an engineering result; present it that way and propose the controlled version unprompted.
- Every number: run logs or silence. `[FILL]` slots left empty are qualitative topics, and saying "I'd have to check" is a scored *positive*.

**References:** Dettmers et al. 2023 (QLoRA, arXiv:2305.14314) · Hu et al. 2021 (LoRA, arXiv:2106.09685) · Aghajanyan et al. 2020 (arXiv:2012.13255) · Biderman et al. 2024 (arXiv:2405.09673) · He et al. 2021 (DeBERTaV3, arXiv:2111.09543) · Reddy et al. 2022 (ESCI, arXiv:2206.06588)

**Next:** [Lesson 2 — Defending the RoPE Transformer](02_defending_the_rope_transformer.md)
