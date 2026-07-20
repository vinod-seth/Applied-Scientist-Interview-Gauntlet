# Chapter 4 — The Introduction

| | |
|---|---|
| **Question covered** | "Tell me about yourself" — and its variants |
| **Why it gets a chapter** | It is asked in **every round of every loop**, and it is the only answer you fully control |
| **Reading time** | ~20 min, then rehearsal |
| **Depends on** | [Ch 0 Target Lock](00_target_lock.md) · [Ch 1 Dossier](01_company_dossier.md) · [Ch 2 Audit](02_resume_audit_and_score.md) · [Ch 3 Tech Stack](03_your_tech_stack_defined.md) |

> 📍 **The core principle: your introduction is a menu.** Every term you volunteer is a dish the interviewer may order. You are not trying to sound impressive — you are trying to *steer the next forty minutes onto ground you have chosen*. A shorter introduction full of defensible hooks beats a longer one full of fragile ones.

---

## 🟢 What This Question Is Actually For

It is not small talk and not a warm-up. In an Amazon loop it does three jobs:

1. **Sets the agenda.** Whatever you name, they can ask about. Most candidates hand over the agenda by accident.
2. **Tests compression.** Can you summarise your own work at the right altitude? A scientist who cannot compress cannot present.
3. **Opens the LP file.** Bar Raisers are trained on all 16 Leadership Principles `[OFFICIAL]`. Your framing seeds **Learn and Be Curious** and **Deliver Results** before a single behavioural question is asked.

**Length: 60–90 seconds.** Past ~2 minutes you are monologuing, and interviewers stop listening while planning how to interrupt.

---

## 🔷 The Structure

Four beats. Do not reorder them — this order front-loads relevance for a listener deciding how interested to be.

| Beat | Seconds | Job |
|---|---|---|
| **1. Position** | ~10 | Who you are right now, in one line |
| **2. Spine** | ~25 | The thread connecting your work — not a project list |
| **3. Proof** | ~30 | *One* project, with a result |
| **4. Aim** | ~15 | Why this role, at this company |

### Why "spine" before "proof"

Listing four projects makes you sound like a portfolio. Naming the *thread* makes you sound like a scientist with a research direction — and it lets you drop three projects from the introduction while keeping their credit. They are still on the resume; you have simply not put them on the menu.

Your spine is unusually clean, and you should use it:

> **You build LLM systems and then interrogate whether they actually work.** Two projects build (QLoRA fine-tuning, from-scratch RoPE transformer); two projects measure and break (RAG failure-mode analysis, calibration under shift).

That is a genuine research identity, and it maps directly onto **Dive Deep** and **Insist on the Highest Standards**.

---

## 🔷 Your Base Introduction

⚠️ **Every `[FILL: metric]` must come from your run logs before you speak this aloud.** Delivering it with a gap is fine — you say "I'd have to check the exact figure." Delivering it with an invented number is the one unrecoverable error.

> "I'm Laksh — I'm partway through an M.Tech in Robotics and AI at IIT Guwahati, graduating in 2027, and my work is concentrated in NLP and large language models.
>
> The thread across my projects is that I build LLM systems and then try to break them. On the building side, I fine-tuned Qwen2.5-1.5B with QLoRA for query–product relevance on Amazon's ESCI dataset, and I implemented a transformer encoder from scratch with rotary position embeddings. On the breaking side, I did a failure-mode analysis of a RAG pipeline and a study of confidence calibration under distribution shift.
>
> The one I'd point to first is the ESCI work: four-class relevance, heavily imbalanced, so I used macro-F1 as the primary metric and reached `[FILL: macro-F1]`, benchmarked against a fully fine-tuned DeBERTa-v3 cross-encoder at `[FILL: baseline]`. `[FILL: one sentence on what that comparison taught you]`.
>
> I'm aiming at Applied Scientist roles because the part I care about is the measurement — deciding whether a model is actually good, not just whether the loss went down. That's what drew me to Amazon specifically; search relevance is where that question has real consequences."

**Word count ≈ 175 → roughly 75 seconds spoken.** Time yourself; most people run 20% long on first attempt.

---

## 🔷 The Menu: What You Just Put On It

Every underlined concept above is now orderable. Cross-check each against your Chapter 3 self-check:

| Hook volunteered | Chapter 3 tier | Safe to serve? |
|---|---|---|
| QLoRA | Tier 1 #1 | Only if ✅ |
| Qwen2.5-1.5B | Tier 2 #22 | Only if ✅ |
| ESCI / 4-class relevance | Tier 3 #23 | Only if ✅ |
| macro-F1 | Tier 1 #4 | Only if ✅ **and the number is in the vault** |
| DeBERTa-v3 cross-encoder | Tier 2 #14, #21 | Only if you can say *why* a cross-encoder is structurally advantaged |
| RoPE / from-scratch transformer | Tier 1 #5, #6 | Only if ✅ — and expect the derivation |
| RAG failure-mode analysis | Tier 1 #10, Tier 3 #30 | Strong ground; welcome the question |
| Calibration / distribution shift | Tier 1 #8, #9 | Only if ✅ |

**The rule: if any row is ⚠️ or ❌ in Chapter 3, cut that hook from the introduction.** It stays on your resume; you simply stop advertising it in the first ninety seconds. You can always raise it later once you have found your footing.

---

## 🔷 Deliberate Omissions

What you left out, and why. This is the part candidates skip and the part that most improves outcomes.

| Omitted | Why |
|---|---|
| **NPTEL certificates and percentages** | Coursework percentages carry little weight for a scientist role and dilute stronger signals |
| **700+ LeetCode** | Belongs in the coding round, not here. Naming it early invites "so, competitive programming?" and reframes you as a coder rather than a scientist |
| **CUDA, SQL, Computer Vision** | Chapter 2 Finding D — undemonstrated. Never volunteer an undemonstrated skill |
| **Two of the four projects, in detail** | Named in one clause each, not described. Keeps the menu short; they remain available |
| **"Passionate about AI"** | Says nothing, and consumes seconds |

### The one you should *not* omit

**Amazon ML Summer School 2026 — invite-only.** It is your most Amazon-specific credential and it is currently buried under NPTEL entries on the resume (Chapter 2, Finding E).

Do not force it into the base introduction — it fits awkwardly and can read as name-dropping. Instead, deploy it when asked *"why Amazon?"*:

> "I was selected for Amazon's ML Summer School in 2026, which is where I first saw how the science work here connects to real search problems — that's a large part of why I'm targeting Applied Scientist here specifically."

---

## 🔷 Per-Round Variants

The base introduction is the science-round version. Adjust the **Proof** and **Aim** beats:

### For a coding round
Compress beats 2–3 hard; they want to start coding.

> "…my work is mostly NLP and LLMs — fine-tuning, transformer internals, and evaluation. I also implement a lot from scratch, and I've solved 700+ DSA problems in C++, so I'm comfortable reasoning through implementation details out loud."

**Here** LeetCode is an asset — it is on-topic and signals fluency. ⚠️ Confirm the language policy first (Chapter 0 checklist #4).

### For an ML breadth round
Signal range rather than depth on one project.

> "…I work across the LLM stack — parameter-efficient fine-tuning, transformer architecture, retrieval systems, and model evaluation. The evaluation side is what I keep returning to: calibration, failure attribution, whether a metric is actually measuring what you think."

### For the behavioural / Bar Raiser round
Lead with trajectory and self-direction — this seeds **Learn and Be Curious** and pre-answers the "how do you learn?" question.

> "…I came into my M.Tech from a CS background and taught myself most of the LLM material through projects — starting with fine-tuning, then implementing a transformer from scratch when I realised I understood the API better than the mechanism, and most recently on evaluation and failure analysis. Each project came from a gap I found in the previous one."

That last clause — *each project came from a gap in the previous one* — is the strongest single sentence available to you, because it is true, it is verifiable from your dates, and it is exactly what **Learn and Be Curious** describes.

---

## 🔷 The Robotics Question

Your M.Tech is **Robotics and AI**; all four projects are NLP/LLM. Expect: *"Why a robotics programme if you're doing NLP?"*

This is a fair question, not a trap. It goes badly only if you sound defensive or improvised.

**Structure:** the programme gave you X, you specialised toward Y deliberately, and here is the evidence.

> "The programme is Robotics and AI, and the AI side is where I concentrated — the coursework in optimisation and learning theory carried directly into what I do now. I chose NLP and LLMs deliberately over the last year because that's where the problems I find most interesting are, and all four of my recent projects are in that space. `[FILL: one specific robotics-course concept you actually use — e.g. state estimation, control theory, probabilistic modelling]`."

⚠️ **Fill that last slot honestly.** If a robotics course genuinely informs your work (probabilistic modelling and uncertainty estimation are plausible bridges to your calibration project), name it — it turns an apparent mismatch into an advantage. If nothing genuinely connects, drop the sentence rather than manufacture a link.

---

## 🔷 Traps

| Trap | Why it costs you |
|---|---|
| **Chronology drift** | Starting at school and walking forward. Nobody asked for a timeline; you burn 90 seconds before reaching anything relevant |
| **The four-project list** | Reciting all four equally gives no signal about what you value, and quadruples the menu |
| **Volunteering an unfilled number** | The one unrecoverable error. Say "I'd have to check the exact figure" — that is a *fine* answer and costs almost nothing |
| **"I'm passionate about AI"** | Unfalsifiable, unmemorable, and a wasted opening |
| **Apologising for being a student** | "I'm just a student, so I haven't…" — you were invited to interview. Never open by discounting yourself |
| **Running past 2 minutes** | The interviewer stops listening and starts planning the interruption |
| **Different story each round** | Interviewers compare notes. Variants should differ in *emphasis*, never in facts |

---

## 🟢 Rehearsal Protocol

1. **Fill the vault first.** Recover `[FILL: macro-F1]` and `[FILL: baseline]` from your run logs. Until then, rehearse with "I'd have to check the exact figure" — never with a placeholder number.
2. **Speak it aloud, timed.** Three times. Not read — spoken. Target 60–90 s.
3. **Record once and listen back.** Uncomfortable, and the fastest fix for pacing and filler words.
4. **Underline every technical term you said**, then check it against Chapter 3. Any ⚠️/❌ gets cut and the introduction re-timed.
5. **Rehearse the two follow-ups you have guaranteed:** *"Tell me more about the ESCI project"* and *"Why Amazon?"* You chose to invite both.

### Readiness check

```markdown
- [ ] Runs 60–90 seconds spoken, not read
- [ ] Contains zero unfilled numbers and zero invented ones
- [ ] Every volunteered hook is ✅ in Chapter 3
- [ ] Spine stated before projects listed
- [ ] Names one project in depth, three in passing
- [ ] Has a specific "why Amazon" that is not flattery
- [ ] Robotics/NLP question answered without defensiveness
- [ ] Variants prepared for coding, breadth, and behavioural rounds
```

**Next:** [Chapter 5 — Gap Map & Study Plan](05_gap_map_and_study_plan.md)
