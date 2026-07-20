# Chapter 2 — Resume Audit & Score

| | |
|---|---|
| **Resume audited** | Laksh Seth — Amazon AS application version, as of 2026-07-20 |
| **Scored against** | [Chapter 1 dossier](01_company_dossier.md) · [Chapter 0 target lock](00_target_lock.md) |
| **Reading time** | ~25 min. Read it in one sitting; the findings compound |
| **Level caveat** | ⚠️ Level is `[UNKNOWN]`, so the Company-Relative score is directional |

> 📍 **The scoring contract.** Every point below is computed from a rule you can read. If you disagree with a score, point at the rule and the line of your resume it was applied to. Nothing here is a vibe, and nothing is softened to be pleasant.

---

## 🟢 Headline

| Score | Result | Band |
|---|---|---|
| **Resume-Intrinsic** | **58 / 100** | 🥉 **Bronze** — real content, exposed claims need work |
| **Company-Relative** (Amazon AS, entry-level) | **54 / 100** | **Gap** — identifiable, and largely closable in your 12-month runway |

**These are never averaged.** They answer different questions, and the gap between them is itself information.

### The one-sentence diagnosis

> Your resume describes **methods** in unusual technical detail and reports **outcomes** almost nowhere — which is precisely inverted for a loop whose Bar Raiser standard is *"better than 50% of people currently in similar roles."*

You have built four genuinely relevant projects in six months. The problem is not the work. The problem is that a reader cannot tell whether any of it *succeeded*.

---

## 🔷 Finding A — The Missing Results (severity: **critical**)

This is the highest-value finding in the audit. Read it twice.

Across all four projects, your resume states **no performance outcome**:

| Project | What you claim | Result stated? |
|---|---|---|
| QLoRA / ESCI | "Used macro-F1 as the primary metric given heavy class imbalance" | ❌ **Names the metric, never reports its value** |
| QLoRA / ESCI | "Benchmarked against a fully fine-tuned DeBERTa-v3 cross-encoder" | ❌ **Never says which won** |
| ROPE Transformer | "Optimized semantic vector clustering" | ❌ No metric at all; "optimized" is unfalsifiable |
| RAG Failure-Mode | "isolating generation rather than retrieval as the dominant error source" | ⚠️ A *finding* — but no magnitudes |
| Calibration | "confidence remained high while accuracy collapsed" | ⚠️ A *direction* — but no ECE values |

### Why this is critical rather than cosmetic

1. **You introduce the metric and then withhold it.** Naming macro-F1 and omitting the number is worse than never mentioning it — an interviewer reads that as a result you would rather not state. Expect: *"What was your macro-F1?"* in the first five minutes, every time.
2. **"Benchmarked against DeBERTa-v3" implies a comparison you don't report.** The unstated outcome invites the assumption that you lost. (If you did lose, that is fine and defensible — see the remedy.)
3. **Deliver Results is a named Leadership Principle** (#14, `[OFFICIAL]`). On current evidence it is your weakest LP, and it is weak *on paper before you enter the room*.
4. **It contradicts your own strongest instinct.** Your RAG and calibration projects are *measurement* projects — you clearly care about evaluation. The resume hides that.

### Remedy (highest priority in Chapter 5)

- Recover the numbers from your run logs into the [Metric Vault](../PROGRESS.md). They exist; you ran these experiments.
- Put the headline result in **the first clause** of the first bullet of each project.
- **If a result is unflattering, state it anyway with the honest framing.** "QLoRA on Qwen2.5-1.5B reached `[FILL]` macro-F1 versus `[FILL]` for a fully fine-tuned DeBERTa-v3 — the encoder won, which is the expected outcome for pair classification and the reason I'd default to a cross-encoder for this task" is a *strong* bullet. It shows judgement, not failure.
- Where no number exists and cannot be recovered, **cut the metric name** rather than leave it dangling.

> [!IMPORTANT]
> Do not invent a number to fill these. An unsupported figure becomes a Dive Deep interrogation you lose. An honest qualitative claim outperforms a fabricated quantitative one — and this course will never score you higher for inventing one.

---

## 🔷 Finding B — Structural: No Industry Experience (severity: **high, partially closable**)

No internships, no publications, no industry roles. For an Applied Scientist req competing against candidates with internships or papers, this is a real gap.

**What it is not:** disqualifying. Entry-level AS hires from strong M.Tech programmes are routine, and your project portfolio is unusually well-chosen.

**What it is:** a reason your evidence must carry more weight than a comparable candidate's.

**Closable portion (you have ~12 months):**
- A publication or workshop paper from your M.Tech work — the single highest-value addition available to you
- A research collaboration or lab project with an external/industry partner
- Open-source contribution to a library you already use (PEFT, FAISS, sentence-transformers)

**Structural portion:** you cannot manufacture two years of industry experience by July 2027. That part gets a *framing* strategy in Chapter 4, not a study task.

**Asset you are under-using:** *Selected for Amazon ML Summer School 2026 — invite-only.* This is Amazon-specific, selective, and currently buried in Achievements below your NPTEL certificates. It should be far more prominent. See Finding E.

---

## 🔷 Finding C — Two Absolute Claims That Invite Attack (severity: **medium**)

| Claim | The problem | Fix |
|---|---|---|
| "trace **every** wrong answer to its root cause" | "Every" is an absolute. One unclassifiable case falsifies it, and your own taxonomy has genuine boundary cases (multi-hop questions, partially-correct answers) | "traced wrong answers to root cause via a three-way classifier" — drop the absolute, keep the substance |
| "**Optimized** semantic vector clustering" | Unfalsifiable and unmeasured. An interviewer will ask "optimised from what to what?" and there is no answer | Either attach a metric or replace with what you actually did: "trained embeddings with a custom contrastive objective" |

Both are small edits with outsized effect: they remove two guaranteed interrogation openings that you currently cannot win.

---

## 🔷 Finding D — Skills-Section Exposure (severity: **medium**)

Your Technical Skills block lists **~35 named technologies**. Everything there is fair game, and the block currently includes items your project bullets never demonstrate:

- **CUDA** — listed under Tools; appears in the ROPE project's tech tag but no bullet describes a CUDA-level contribution. If you did not write kernels, this invites a question you will lose.
- **SQL** — listed; no project uses it. Expect "tell me about a complex query you wrote."
- **Computer Vision** — listed under ML & Deep Learning; only the calibration project touches images, and there as a *consumer* of a pretrained classifier.
- **C++** — strong (700+ LeetCode) but attached to no project.

**The rule:** the skills list is a menu. Every line is a dish the interviewer may order. **Remove anything you cannot defend for two minutes** — not because it is dishonest, but because it converts a strong resume into an interrogation surface you did not choose.

Chapter 3 gives you the full term-by-term self-check to decide what stays.

---

## 🔷 Finding E — Ordering & Emphasis (severity: **medium**)

| Issue | Effect | Fix |
|---|---|---|
| Amazon ML Summer School buried in Achievements, below NPTEL certificates | Your most Amazon-relevant credential reads as an afterthought | Move up; consider a line in the header/summary area |
| NPTEL percentages (87%, 80%) given prominence | Coursework percentages carry little weight for an AS role and consume scarce space | Demote to a single line or cut |
| Projects ordered QLoRA → ROPE → RAG → Calibration | Reverse-chronological is fine, but your **RAG failure-mode** project is arguably your strongest *scientific* work (it reports a finding) | Consider leading with the project you most want drilled |
| No summary/positioning line | A screener spends ~30 seconds; nothing frames "NLP/LLM scientist, M.Tech, evaluation-focused" | Add 1–2 lines. This also pre-empts the robotics/NLP question |
| M.Tech is *Robotics and AI*; all projects are NLP/LLM | Reads as a mismatch to a screener who does not read closely | Address in the summary line and in your introduction (Chapter 4) |

---

## 🔷 Finding F — What Is Genuinely Strong (do not "fix" these)

An audit that only criticises is unusable. These are your assets:

1. **Project selection is excellent.** QLoRA/PEFT, transformer internals, RAG evaluation, calibration — this maps almost exactly onto what an NLP/LLM Applied Scientist does. Very few fresher resumes are this well-targeted.
2. **The ESCI project uses Amazon's own dataset.** That is a deliberate, visible signal and it is working.
3. **Two of four projects are *evaluation* projects.** RAG failure-mode analysis and calibration-under-shift are self-criticism work. That is rare in freshers, and it is direct evidence for **Insist on the Highest Standards** and **Dive Deep**.
4. **Technical specificity is above average.** "4-bit NF4 + LoRA adapters", "Pre-LayerNorm", "Rotary Position Embeddings" — you name real mechanisms, not categories. This is the half of the equation you already do well.
5. **Coherent six-month trajectory.** Feb→Jul 2026 shows escalating sophistication. That narrative is usable evidence for **Learn and Be Curious**.

---

## 🔷 Resume-Intrinsic Score — Full Derivation

Recompute it yourself; every line cites the criterion and the evidence.

### A. Evidence & Verifiability — **9 / 30**

| Criterion | Max | Score | Why |
|---|---|---|---|
| Every quantitative claim traces to a log/repo/publication | 12 | **3** | Config-level numbers (20K+ pairs, <2%, 16GB) are verifiable; **no performance metric appears at all**, so the criterion mostly cannot be met |
| Projects have inspectable artifacts | 10 | **4** | QLoRA bullet explicitly claims a reproducible repo (good); the other three cite no artifact. GitHub link exists but per-project repos are unnamed |
| No claim the candidate cannot demonstrate on request | 8 | **2** | CUDA, SQL, Computer Vision, and "optimized clustering" are not demonstrable from the stated work (Finding D) |

### B. Specificity — **13 / 20**

| Criterion | Max | Score | Why |
|---|---|---|---|
| Concrete methods named, not category words | 8 | **8** | **Full marks.** NF4, Pre-LayerNorm, RoPE, bge-small, ECE — consistently precise |
| Outcomes stated with magnitude and metric | 7 | **1** | Finding A. Two projects state a *direction*; none states a magnitude |
| Candidate's own contribution distinguishable | 5 | **4** | Solo projects, unambiguous ownership. Minor deduction: no statement of what was hard or self-designed |

### C. Depth Signals — **12 / 20**

| Criterion | Max | Score | Why |
|---|---|---|---|
| Claims reflect design decisions, not tool usage | 8 | **6** | "framing search relevance as a supervised NLP task", "custom contrastive loss", from-scratch encoder — real decisions. Some bullets still read as tool inventory |
| Trade-offs or limitations acknowledged | 6 | **4** | The RAG and calibration projects are inherently limitation-focused; but no bullet states a trade-off the candidate *chose* |
| No term the candidate cannot define one level down | 6 | **2** | ⚠️ **Provisional** — recomputed after your Chapter 3 self-check. Scored low pending Finding D's undemonstrated terms |

### D. Interrogation Surface Hygiene — **9 / 15**

| Criterion | Max | Score | Why |
|---|---|---|---|
| No undefendable buzzword padding | 6 | **3** | ~35 listed technologies including several undemonstrated (Finding D) |
| Claim density defensible in 45 minutes | 5 | **4** | Four projects is right; each is drillable |
| No internal contradictions | 4 | **2** | Skills list vs. project evidence (CUDA/SQL/CV); Robotics degree vs. all-NLP portfolio unaddressed |

### E. Communication — **15 / 15**

| Criterion | Max | Score | Why |
|---|---|---|---|
| First clause carries the result | 6 | **6** | Bullets lead with strong action verbs and the substantive act |
| Consistent tense, person, formatting | 4 | **4** | Clean and consistent throughout |
| Scannable in 30 seconds | 5 | **5** | Well-structured, appropriate length, clear hierarchy |

> Note the shape of this score: **Communication is perfect; Evidence is nearly empty.** You write well about work whose outcomes you do not report. That single asymmetry is the whole audit.

### Total

```
A Evidence & Verifiability      9 / 30
B Specificity                  13 / 20
C Depth Signals                12 / 20
D Interrogation Hygiene         9 / 15
E Communication                15 / 15
                              ──────────
  RESUME-INTRINSIC             58 / 100   →  🥉 Bronze
```

**Bronze (50–69):** real content, but exposed claims need work before the loop. **Finding A alone is worth roughly +18** — recovering your metrics moves you to Silver without writing a single new line of code.

---

## 🔷 Company-Relative Score — Amazon AS, Entry-Level

⚠️ Level `[UNKNOWN]`; scored against Chapter 1's provisional entry-level bar. Re-run when your recruiter confirms.

### A. Role Match — **21 / 30**
Required NLP/LLM skills are present and current. Deductions: no production/deployment evidence; classical ML and statistics unrepresented; no publication.
*Scored against: Chapter 1 Round Types 1–2.*

### B. Level Fit — **12 / 25**
Scope is single-owner course-scale projects. No evidence of ambiguous problem definition, cross-person collaboration, or work carried to a consumer. **The most common silent rejection reason for strong fresher candidates**, and the hardest to see from inside.
*Scored against: Chapter 1 "What Changes By Level".*

### C. Round Readiness — **13 / 25**

| Round type | Weight | Readiness | Note |
|---|---|---|---|
| Project depth | 8 | **5** | Strong material; undermined by missing results |
| ML breadth | 6 | **3** | Untested surface outside NLP |
| Coding | 5 | **4** | 700+ LeetCode is real evidence; protocol untested; ⚠️ language policy unconfirmed |
| Behavioural / Bar Raiser | 6 | **1** | **Zero prepared stories, no industry material.** Largest single gap in the course |

### D. Signal Alignment — **8 / 20**
Against the 16 published LPs `[OFFICIAL]`:

| LP | Evidence on resume | Score |
|---|---|---|
| Dive Deep | Strong — failure-mode decomposition, calibration analysis | 3/4 |
| Learn and Be Curious | Strong — self-directed six-month trajectory | 3/4 |
| Invent and Simplify | Moderate — from-scratch transformer | 2/4 |
| **Deliver Results** | **Weak — no outcome stated anywhere** (Finding A) | **0/4** |
| Insist on the Highest Standards | Moderate — evaluation projects imply it, nothing states it | 0/4 |

### Total

```
A Role Match           21 / 30
B Level Fit            12 / 25
C Round Readiness      13 / 25
D Signal Alignment      8 / 20
                      ──────────
  COMPANY-RELATIVE     54 / 100   →  Gap
```

**Gap (40–59):** identifiable and, in your case, **largely closable** — you have ~12 months, and the two biggest deductions (Deliver Results at 0/4, Behavioural at 1/6) are addressable without new research.

---

## 🔷 Claim Extraction Table

Every substantive claim, with defensibility and required action. **Every HIGH row appears in the Chapter 5 gap map.**

| # | Claim | Type | Backing artifact | Risk | Action |
|---|---|---|---|---|---|
| 1 | "4 classes, 20K+ pairs" | config | ESCI dataset | LOW | Confirm exact subset/locale |
| 2 | "Fine-tuned Qwen2.5-1.5B with QLoRA (4-bit NF4 + LoRA adapters)" | method | run config | LOW | — |
| 3 | "training <2% of parameters" | metric | training config | MED | Recover exact % and count |
| 4 | "on a single 16GB GPU" | config | run environment | LOW | Know peak memory if logged |
| 5 | "Benchmarked against a fully fine-tuned DeBERTa-v3 cross-encoder" | comparison | **none stated** | **HIGH** | Report both scores or reframe |
| 6 | "Used macro-F1 as the primary metric" | metric | **value never given** | **HIGH** | **Recover the number** |
| 7 | "reproducible repo … Colab/Kaggle quickstart" | artifact | repo | LOW | Ensure it actually runs |
| 8 | "custom Pre-LayerNorm Transformer encoder from scratch" | method | code | LOW | Know layers/heads/d_model cold |
| 9 | "Implemented RoPE and Multi-Head Attention" | method | code | LOW | Derivation required (Session 1) |
| 10 | "Optimized semantic vector clustering … custom Contrastive Loss" | outcome | **none** | **HIGH** | Attach metric or rewrite (Finding C) |
| 11 | "trace **every** wrong answer to its root cause" | absolute | instrumentation | **HIGH** | Drop the absolute (Finding C) |
| 12 | "isolating generation rather than retrieval as the dominant error source" | finding | sweeps | MED | Scope the claim; know falsifiers |
| 13 | "ECE and reliability diagrams" under corruptions | method | eval scripts | MED | Report values; know binning scheme |
| 14 | "demonstrated where temperature scaling fails to transfer" | finding | sweeps | MED | Own the "known result" question |
| 15 | "700+ DSA problems in C++" | count | LeetCode profile | LOW | Verifiable; good |
| 16 | "Selected for Amazon ML Summer School 2026" | credential | selection email | LOW | **Promote it** (Finding E) |
| 17 | Skills: CUDA, SQL, Computer Vision | claim | **no project evidence** | **HIGH** | Defend or remove (Finding D) |

---

## 🔷 Claim Vault

Every number your resume implies, and whether you can source it. **Fill from run logs only.**

| Resume number | Value | Source artifact | Status |
|---|---|---|---|
| QLoRA macro-F1 (test) | `[FILL: metric]` | | ❌ UNVERIFIED |
| QLoRA per-class F1 (E/S/C/I) | `[FILL: metric]` | | ❌ UNVERIFIED |
| DeBERTa-v3 baseline macro-F1 | `[FILL: metric]` | | ❌ UNVERIFIED |
| Trainable parameter % and count | `[FILL: metric]` | | ❌ UNVERIFIED |
| Peak GPU memory | `[FILL: metric]` | | ❌ UNVERIFIED |
| ROPE model: eval metric + value | `[FILL: metric]` | | ❌ UNVERIFIED |
| ROPE model: layers/heads/d_model/params | `[FILL: metric]` | | ❌ UNVERIFIED |
| RAG: failure-mode distribution | `[FILL: metric]` | | ❌ UNVERIFIED |
| RAG: recall@k curve | `[FILL: metric]` | | ❌ UNVERIFIED |
| Calibration: per-severity accuracy/ECE | `[FILL: metric]` | | ❌ UNVERIFIED |
| Calibration: fitted temperature T | `[FILL: metric]` | | ❌ UNVERIFIED |

**11 of 11 unverified.** This table is the most actionable page in the course. Filling it is task #1 in Chapter 5.

---

## 🟢 What This Audit Is Not Saying

To be precise about the verdict, because Bronze/Gap read harsher than intended:

- **Not** that the projects are weak. They are well-chosen and current.
- **Not** that you are unqualified. You are mid-M.Tech with 12 months of runway.
- **Not** that the score is fixed. Roughly **+18 intrinsic and +10 relative** are available from recovering numbers and preparing behavioural stories — neither requires new research.

What it *is* saying: **your resume currently under-sells work you have actually done**, and the loop will not give you credit for results you did not state.

---

## 🟢 Re-Score Checkpoint

Session 9 re-runs this identical rubric and reports per-dimension deltas. Do not change the rubric between runs; a score that improved because the criteria softened means nothing.

**Next:** [Chapter 3 — Your Own Tech Stack, Defined](03_your_tech_stack_defined.md)
