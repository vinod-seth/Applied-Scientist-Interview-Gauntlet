# Lesson 6 — Mock Round: The LLM Round

| | |
|---|---|
| **Prepares** | A 45–60 minute LLM round: design questions with no single right answer, arithmetic on demand, and one evaluation question that decides the round |
| **Prerequisites** | Lessons 1–5 worked through, including their rapid-fire drills |
| **Time** | ~60 min (mock round + design questions + honest self-assessment) |

> 📍 **How this round differs from Session 4's.** The DL round rewarded *derivation* — algebra and shapes. This one rewards **judgment under a stated constraint**. Most questions here have several defensible answers, and the score comes from naming the trade-off and the measurement that would settle it. An answer with no cost in it is incomplete no matter how correct the mechanism.

---

## 🟢 Before You Start

1. Have your Session 1 numbers to hand — your QLoRA rank, target modules, and observed peak memory. Two questions here invite you to instantiate general arithmetic on your own run.
2. Have your Session 2 RAG numbers too. The retrieval questions get much stronger when you can say "in my own study, the split was `[FILL: failure distribution]`."
3. Speak **out loud**, and end every design answer with the measurement you would run. That habit is the single biggest difference between a passing and a strong LLM round.

---

## 🟢 The Bar for This Round

| Property | What it sounds like | What fails |
|---|---|---|
| **Quantified** | Bytes per parameter, tokens per parameter, recall@k, standard error | "It uses less memory" |
| **Traded** | "X buys this, costs that, and here's when I'd take the other one" | A recommendation with no cost attached |
| **Measured** | The specific run or test that settles the question | "We'd evaluate it" |
| **Bounded** | Naming what the technique does *not* fix | Presenting a method as universally better |

```mermaid
flowchart LR
    Q["Question"] --> M["Mechanism<br/>what it actually does"]
    M --> N["Number<br/>bytes · tokens · recall · n"]
    N --> T["Trade-off<br/>and when you'd choose otherwise"]
    T --> E["The measurement<br/>that would settle it"]
    E --> S["Stop.<br/>Invite the follow-up."]
```

---

## 🟢 The Mock Round

Fourteen questions across all five lessons, delivered in sequence. The interviewer reads each aloud; you answer, then move on. At the end you get a consolidated evaluation of the whole round.

<RehearsalStudio rubric="mechanism" minSeconds="40" maxSeconds="120" prompt="Answer each out loud with mechanism, a number, the trade-off, and the measurement that would settle it — then stop and invite the follow-up." questions="How much memory does full fine-tuning a 7B model need, and where does it actually go? || State LoRA precisely: the parameterization, the initialization, and what alpha does. || Is QLoRA's memory saving the same saving as LoRA's? Be exact. || When is PEFT the wrong choice? || Why sample at all — isn't the most likely text the best text? || Why does top-p exist when top-k already truncates the tail? || Your generation loops on a repeated phrase. Walk me through your diagnosis. || What is speculative decoding, and why doesn't it change the output distribution? || I have a fixed compute budget. How big a model and how many tokens, and why? || Modern models are trained far past 20 tokens per parameter. Why is that not a mistake? || RLHF versus DPO — state both objectives and the real trade. || Why does a hybrid BM25-plus-dense retriever beat either alone? || RAG, fine-tuning, or long context — give me your decision rule. || A prompt change moved accuracy from 71% to 74% on 200 examples. Did it improve?" />

> [!TIP]
> If a question has several defensible answers, say so and then commit to one: "There are two reasonable designs here; I'd take the first, because…". Interviewers read that as senior behavior. Refusing to choose reads as evasion.

---

## 🟢 The Three Set-Piece Questions

These three carry the round. Each is open-ended, each runs 3–5 minutes in a real interview, and each deserves separate rehearsal because the structure of the answer — not the facts in it — is what is being scored.

<RehearsalStudio rubric="mechanism" minSeconds="90" maxSeconds="200" prompt="Design question, ~4 minutes: 'Design a retrieval system for a company support knowledge base. Start from nothing.' Give the stages, say what each one prevents, and finish with the three numbers you would measure to attribute a failure to a stage." />

<RehearsalStudio rubric="mechanism" minSeconds="90" maxSeconds="200" prompt="Design question, ~4 minutes: 'Design the evaluation suite for that assistant. How would you know it got better?' Answer in layers, say what each layer catches, and name at least one thing you would deliberately measure separately from answer correctness." />

<RehearsalStudio rubric="mechanism" minSeconds="75" maxSeconds="180" prompt="Constraint question, ~3 minutes: 'You have one 24 GB GPU, a 7B base model, and 50k labeled examples for a domain classification task. What do you do, and what would change your mind?' Do the memory arithmetic out loud, name the alternative you rejected, and say what result would make you switch to it." />

---

## 🟢 Honest Self-Assessment

The bar is **"I stated the mechanism, gave a number, named the trade-off, and said what I'd measure — out loud, without notes."** Recognizing a technique is not the same as placing it.

| # | Topic | Lesson | Can you produce it cold? |
|---|---|---|---|
| 1 | The ≈16 bytes/param breakdown, and what freezing removes | 1 | ☐ Clean ☐ Shaky ☐ No |
| 2 | LoRA: $W_0 + \frac{\alpha}{r}BA$, $B=0$ init, $r(d+k)$ params | 1 | ☐ Clean ☐ Shaky ☐ No |
| 3 | LoRA vs QLoRA — which term each one removes | 1 | ☐ Clean ☐ Shaky ☐ No |
| 4 | The landscape and the mergeability axis | 1 | ☐ Clean ☐ Shaky ☐ No |
| 5 | Why likelihood-maximizing text degenerates | 2 | ☐ Clean ☐ Shaky ☐ No |
| 6 | Top-k's failure mode and what top-p fixes | 2 | ☐ Clean ☐ Shaky ☐ No |
| 7 | Repetition: three causes, in diagnosis order | 2 | ☐ Clean ☐ Shaky ☐ No |
| 8 | Speculative decoding is exact; the bandwidth argument | 2 | ☐ Clean ☐ Shaky ☐ No |
| 9 | $C \approx 6ND$ and the Chinchilla allocation | 3 | ☐ Clean ☐ Shaky ☐ No |
| 10 | Why serving-oriented models are overtrained | 3 | ☐ Clean ☐ Shaky ☐ No |
| 11 | RLHF and DPO objectives, and the off-policy trade | 3 | ☐ Clean ☐ Shaky ☐ No |
| 12 | Two-stage retrieval; why hybrid; why RRF | 4 | ☐ Clean ☐ Shaky ☐ No |
| 13 | RAG vs fine-tuning vs long context, as a rule | 4 | ☐ Clean ☐ Shaky ☐ No |
| 14 | Paired testing: SE arithmetic and McNemar's | 5 | ☐ Clean ☐ Shaky ☐ No |
| 15 | Judge validity: biases, fixes, and the human audit | 5 | ☐ Clean ☐ Shaky ☐ No |

**How to read your result**

| Clean count | What it means | Next step |
|---|---|---|
| **12–15** | LLM round-ready | Move to Session 6; revisit any single ☐ Shaky before the loop |
| **7–11** | You know the material but aren't placing it in the design space | Re-run the three set-piece questions — a framing problem, not a knowledge problem |
| **≤ 6** | Real gaps | Re-read the lesson behind each gap, run its rapid-fire drill, then re-attempt |

Log every ☐ Shaky and ☐ No in the Gap Log in [PROGRESS.md](../../PROGRESS.md) with the exact point your depth ran out.

---

## 🟢 The Synthesis Question

LLM interviewers often close by asking you to connect the whole stack under one constraint. This is the highest-value single answer in the session.

<RehearsalStudio rubric="mechanism" minSeconds="90" maxSeconds="180" prompt="Answer out loud, ~2.5 minutes: 'You're given a base model and a business problem: answer customer questions about a product catalogue that changes weekly. Take me from that sentence to a shipped system, and tell me where each decision could go wrong.' Give one causal narrative, not a list of techniques." />

<details><summary>✅ Model answer — the arc</summary>

"I'd start by splitting the problem into **knowledge** and **behavior**, because they take different solutions.

The catalogue changes weekly, so the knowledge side is **retrieval**, not fine-tuning — an index update is instant and auditable, while fine-tuning facts is unreliable, unattributable, and would have to be redone every week. The behavior side — answer format, tone, when to refuse — is where fine-tuning belongs, and I would start with prompting and only fine-tune if prompting plateaus. If I do fine-tune, it's **LoRA or QLoRA**, because full fine-tuning costs about 16 bytes per parameter — roughly 112 GB at 7B — of which only 2 bytes are weights; freezing the base deletes the other 14, and 4-bit quantization then shrinks what remains.

For retrieval I'd build **two stages with opposite objectives**: a cheap hybrid first stage — BM25 plus a dense encoder fused by reciprocal rank fusion, tuned for recall@100, since product queries contain SKUs and model numbers that lexical search handles and embeddings lose — then a cross-encoder reranker down to about five passages for precision. I'd put the best passages at the start and end of the context, because attention to the middle is measurably weaker.

At generation I'd use **temperature 0 or near it** — this is a factual task where the answer should be reproducible — with **citations required**, so hallucination is detectable rather than suspected. If a schema is consumed downstream, constrained decoding for that span only.

Then the part that decides whether any of it worked: **evaluation in layers.** Component-level recall@k so a retrieval regression can't hide inside end-to-end accuracy; end-to-end correctness *and groundedness separately*, since an answer can be right but unsupported; refusal scored as a correct outcome when the catalogue doesn't contain the answer; and every production failure added to a regression suite. If I use an LLM judge, I audit it against a hundred hand-labeled items and report agreement, and I never use it where an exact check exists.

**Where it goes wrong:** the index goes stale, or an embedder upgrade silently invalidates every vector; ANN search loses answers the embedder actually found, which is invisible unless I measure against exact search; the eval set is synthetic and biased toward questions the corpus answers well; and a 3-point improvement gets shipped as real when 200 examples can't distinguish it from noise."
</details>

---

## 🟢 Summary

- This round rewards **judgment under a constraint**, not derivation. Every answer needs a number, a trade-off, and the measurement that would settle it.
- The three set-piece questions — design the retrieval system, design the evaluation, spend one 24 GB GPU — are the highest-probability open questions in an applied LLM round.
- The synthesis arc — **split knowledge from behavior → retrieve for knowledge, adapt for behavior → decode for the task → evaluate in layers** — connects all five lessons, and offering it unprompted is what a strong candidate does.

**Session complete.** Next: the [Chapter Quiz](quiz.md), then [Session 6 — Coding: DSA & Algorithms](../06_dsa_algorithms/00_locked.md) *(locked)*.
