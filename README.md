# The Gauntlet: Applied Scientist Interview Prep (M.Tech Fresher Edition)

| | |
|---|---|
| **Target role** | Applied Scientist (entry-level, NLP/LLM focus) |
| **Candidate profile** | M.Tech Robotics & AI (IIT Guwahati), B.Tech CS (AI/ML), no industry experience — four flagship projects carry the resume |
| **Format** | 9 sessions, delivered one at a time; each ends in a Boss Fight that gates the next |
| **Version** | 1.1 |
| **Last verified** | 2026-07-13 (model names, paper links, and Leadership Principles list checked on this date) |

---

## 🟢 Who This Course Is For

One specific person: an M.Tech fresher targeting an entry-level Applied Scientist role, whose resume is built on four projects — QLoRA on the ESCI product-search dataset, a from-scratch RoPE transformer, RAG failure-mode analysis, and calibration under distribution shift. Every question, drill, and boss fight is tailored to that resume. If that's not you, the structure still works, but the content won't fit.

**Prerequisites:** strong M.Tech-level ML background. This course skips basics on purpose. If you cannot write the scaled dot-product attention equation from memory, close this and go do that first.

---

## 🟢 The Premise: Why This Is a Game

The Applied Scientist loop is not a quiz. It is a depth interrogation: an interviewer picks one line on your resume and asks "why" and "how" until you run out of floor. Most prep fails for one of two reasons:

1. **Shallow coverage** — you "revised" 40 topics to a depth of two follow-ups, and the interviewer needs five.
2. **Quitting** — prep takes weeks, motivation decays, and you stop.

The game layer attacks both. Depth is the only thing that scores. Consistency is the only thing that keeps the map filling in.

---

## 🟢 Game Rules

### 🟢 Tiered Mastery (per topic)

| Tier | Meaning | Interview reality |
|---|---|---|
| 🥉 **Bronze** | You know it — can state the concept correctly | Survives the *first* question only |
| 🥈 **Silver** | You can derive it — equations, memory math, design rationale from first principles | Survives 2–3 follow-ups |
| 🥇 **Gold** | You can defend it — five consecutive why/how follow-ups without hand-waving | **This is the actual hiring bar.** Nothing below Gold counts as "done" |

### 🟢 XP: Paid for Defense, Never for Completion

- Reading a lesson: **0 XP**. The interviewer does not care what you read.
- Surviving a follow-up chain in a drill: **10 XP per depth level survived** (max 50 per chain).
- Hitting your floor honestly ("I don't know, but here's how I'd find out"): **5 XP**. Honesty at the floor is a scored skill — it is exactly what a Bar Raiser wants to see.
- Hand-waving, buzzword-smuggling, or quoting a number you cannot back with a run log: **0 XP, and the topic drops one tier.** The game never pays for a made-up number.

### 🟢 Boss Fights

Every session ends with a **Bar Raiser Simulation**: a scripted interrogation that drills the session's core topics five levels deep. Clearing it (Gold on the session's core topics) unlocks the next session. Retry as often as you want — each attempt restarts from question one.

### 🟢 The Floor-Hit Map

Every time you run out of depth, log *where*. That log is not a failure record — it is next session's study map. The strongest predictor of clearing a boss fight on retry is a well-kept floor-hit map. The template lives in [PROGRESS.md](PROGRESS.md).

### 🟢 Streaks & the Skill Tree

The skill tree in [PROGRESS.md](PROGRESS.md) shows all 9 sessions and their core topics. Each Gold fills in a node. Streak rule: any scored drill attempt (even a failed one) on a given day maintains the streak. The enemy is not a failed drill — it is a zero-activity day.

---

## 🟢 Non-Negotiable Constraints (read before Session 1)

1. **Never fabricate metrics.** Every number in your answers comes from your own run logs. This course marks every such slot as `[FILL: metric]`. An unsupported resume number becomes a Dive Deep interrogation you will lose; an honest qualitative statement is strictly better than an invented figure.
2. **Depth over coverage.** Gold is always "defends against five follow-ups." There is no partial credit for touching more topics shallowly.
3. **Assumption flags.** Where the answer depends on the specific JD or team, the material flags it with `⚠️ JD-DEPENDENT`. Example: ML system design is *excluded by default* in an entry-level AS loop, but flips *in* for some applied teams — check your recruiter briefing.
4. **Leadership Principles carry heavy weight.** Technically strong candidates fail the AS loop most often on behavioral rounds. Sessions 8–9 are not garnish; they are half the loop. Priority principles for this candidate: Dive Deep, Learn and Be Curious, Invent and Simplify, Deliver Results, Insist on the Highest Standards.

---

## 🟢 Course Map (9 Sessions)

| # | Session | Core topics (boss-fight gated) | Status |
|---|---|---|---|
| 1 | Project Deep-Dives: Fine-Tuning & Architecture | QLoRA/NF4, LoRA math, GPU memory accounting, RoPE derivation, Pre-LN, contrastive loss | ✅ Delivered |
| 2 | Project Deep-Dives: Systems & Evaluation | RAG failure taxonomy, retrieval knobs, ECE, reliability diagrams, temperature scaling | 🔒 Locked |
| 3 | ML Fundamentals: Core Theory | Bias–variance, regularization, probabilistic foundations, optimization, metrics | 🔒 Locked |
| 4 | ML Fundamentals: Deep Learning & Transformers | Backprop mechanics, normalization, attention variants, training pathologies | 🔒 Locked |
| 5 | LLMs, Fine-Tuning & Retrieval | PEFT landscape, decoding, scaling behavior, RAG design, evaluating LLM systems | 🔒 Locked |
| 6 | Coding: DSA & Algorithms | Interview patterns under pressure; clarify → approach → narrate → test → optimize | 🔒 Locked |
| 7 | Coding: ML From Scratch | Attention, losses, k-means, logistic regression, beam search from a blank editor | 🔒 Locked |
| 8 | Behavioral: Leadership Principles & STAR | LP mapping, STAR with quantified results, fresher-sourcing of failure/conflict stories | 🔒 Locked |
| 9 | Mock Loop & Integration | Full 4-round simulated loop, playbook execution, final floor-hit audit | 🔒 Locked |

**Round Playbook** (applies across all sessions): [playbooks/round_playbook.md](playbooks/round_playbook.md)

---

## 🟢 How a Session Works

1. **Scope brief** — every topic an interviewer could raise from this session's material.
2. **Deep-dive lessons** — peer-level explanations plus the 5–6 hardest follow-ups per project, with model answers at AS depth.
3. **Tiered challenges** — Bronze/Silver/Gold drills per core topic.
4. **Boss fight** — the Bar Raiser simulation. Clear it to unlock the next session.
5. **Floor-hit logging** — update [PROGRESS.md](PROGRESS.md) before you close the session.

---

## 🟢 Maintenance

- **Review cycle:** quarterly. LLM tooling and model references rot in 3–6 months.
- Volatile references (model names, library APIs) carry a "last verified" date.
- Changes are tracked in [CHANGELOG.md](CHANGELOG.md).

---

## 🟢 References & Credits

- Amazon Leadership Principles (official list): https://www.amazon.jobs/content/en/our-workplace/leadership-principles
- Shopping Queries (ESCI) dataset: Reddy et al. (2022), *Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search* (https://arxiv.org/abs/2206.06588), Apache-2.0 license.
- All research techniques referenced in lessons are cited inline with authors, paper titles, and links.
