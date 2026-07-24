# Chapter 5 — Gap Map & Study Plan

| | |
|---|---|
| **Derived from** | [Ch 1 bar](01_company_dossier.md) − [Ch 2 audit](02_resume_audit_and_score.md), ordered by risk |
| **Runway** | ~12 months to application (2026–27 cycle), start ~July 2027 |
| **Ordering rule** | Risk-weighted gap, **not** what is pleasant to study |
| **Reading time** | ~20 min, then it becomes your operating schedule |

> 📍 **The ordering will feel wrong, and that is deliberate.** Strong technical candidates over-prepare their strengths — more transformer internals feels productive because it is enjoyable and you are already good at it. The highest-value items below are mostly *not* technical study. Trust the ordering; the arithmetic is shown.

---

## 🟢 The Gap Map

Every gap traces to a specific Chapter 2 finding or Chapter 1 bar statement. Nothing here is free-floating advice.

`priority = (bar − current) × P(tested) × round_weight`

| # | Gap | Source | Bar−Current | P(tested) | Weight | **Priority** | Type |
|---|---|---|---|---|---|---|---|
| **1** | **No results reported on any project** | Ch2 Finding A | 4/4 | 1.0 | 8 | **32** | 🟢 Closable |
| **2** | **Zero prepared behavioural stories** | Ch2 C:1/6 | 4/4 | 1.0 | 6 | **24** | 🟢 Closable |
| **3** | Deliver Results: no evidence (LP #14) | Ch2 D:0/4 | 4/4 | 0.9 | 6 | **21.6** | 🟢 Closable |
| **4** | Undemonstrated skills on resume | Ch2 Finding D | 3/4 | 0.7 | 8 | **16.8** | 🟢 Closable (this week) |
| **5** | ML breadth outside NLP untested | Ch2 C:3/6 | 3/4 | 0.9 | 6 | **16.2** | 🟢 Closable |
| **6** | Level Fit: single-owner project scope | Ch2 B:12/25 | 3/4 | 0.8 | 6 | **14.4** | 🟠 Partly structural |
| **7** | Two absolute/unfalsifiable claims | Ch2 Finding C | 2/4 | 0.8 | 8 | **12.8** | 🟢 Closable (this week) |
| **8** | Coding protocol untested (≠ ability) | Ch2 C:4/5 | 2/4 | 1.0 | 5 | **10** | 🟢 Closable |
| **9** | No publication / industry experience | Ch2 Finding B | 4/4 | 0.5 | 5 | **10** | 🔴 Structural |
| **10** | Resume ordering & positioning | Ch2 Finding E | 2/4 | 0.6 | 8 | **9.6** | 🟢 Closable (this week) |
| **11** | Robotics/NLP coherence unaddressed | Ch0, Ch2 E | 2/4 | 0.6 | 6 | **7.2** | 🟢 Closable |
| **12** | Level/JD unknown → prep unfocused | Ch0 | 3/4 | — | — | **blocker** | 🟢 One email |

### Read the top of that table carefully

**The three highest-priority items are not technical study.** They are: recover numbers you already generated, write stories about work you already did, and edit a document you already have. Ranks 1–4 total **94 priority points**; every technical-study item below them totals less.

This is the finding of the chapter: **your largest gains are in evidence and presentation, not in new knowledge.**

---

## 🔷 Closable vs. Structural

| Type | Meaning | Strategy |
|---|---|---|
| 🟢 **Closable** | Can be fixed before you apply | Study task or edit — schedule it |
| 🟠 **Partly structural** | Can be improved, not eliminated | Improve what is reachable, frame the rest |
| 🔴 **Structural** | Cannot be manufactured in 12 months | **Framing strategy only.** Do not schedule busywork against it |

### Gap #9 — the structural one, handled honestly

You cannot manufacture two years of industry experience by July 2027. What you *can* do, in descending value:

1. **A workshop paper or publication from your M.Tech work** — the single highest-value addition available. Your RAG failure-mode analysis is closest to publishable: it has a finding, a method, and a falsifiable claim.
2. **A research collaboration** with a lab or industry partner through IIT Guwahati.
3. **A substantive open-source contribution** to a library you already use (PEFT, FAISS, sentence-transformers). A merged non-trivial PR is verifiable evidence of working to someone else's standard — which is precisely what Gap #6 (Level Fit) is missing.

**The framing** (for Chapter 4 and Session 8): you have no industry experience, and the answer is not to apologise for it. It is to point at a portfolio that is *unusually* well-targeted, self-directed, and evaluation-literate. Most freshers cannot describe a failure taxonomy they built themselves.

---

## 🔷 The Plan

Paced against ~12 months. ⚠️ **If IIT Guwahati placement season governs your timeline, applications may open far earlier than a 2027 start implies — confirm with your placement cell, because that compresses everything below.**

### 🔴 Week 1 — The Free Wins

Nothing here requires new learning. This is the highest value-per-hour week in the entire plan.

```markdown
- [ ] Email/ask recruiter or placement cell the Chapter 0 checklist   (Gap #12 — unblocks everything)
- [ ] Recover ALL 11 Claim Vault numbers from your run logs           (Gap #1 — worth ~+18 intrinsic)
- [ ] Rewrite each project's first bullet to lead with its result     (Gaps #1, #3)
- [ ] Complete the Chapter 3 self-check honestly                      (feeds Ch2 Depth Signals)
- [ ] Cut or defend CUDA, SQL, Computer Vision                        (Gap #4)
- [ ] Fix "every wrong answer" and "optimized clustering"             (Gap #7)
- [ ] Promote Amazon ML Summer School above NPTEL; add a 2-line summary (Gaps #10, #11)
```

**Expected effect: 58 → ~76 intrinsic, 54 → ~62 relative.** No new research. This is why it is Week 1.

> [!IMPORTANT]
> If a number cannot be recovered, **cut the metric name from the resume** rather than leave it dangling — and never invent one. A missing number is a small weakness; an invented one is disqualifying.

### 🟠 Weeks 2–6 — Defend What You Built

```markdown
- [ ] Session 1 assessment: QLoRA + RoPE depth        (Gap #1 defence)
- [ ] Session 2 assessment: RAG + calibration depth   (Gap #1 defence)
- [ ] Every ⚠️/❌ from Chapter 3 Tiers 1–2 → Gold
- [ ] Rehearse the Chapter 4 introduction to the readiness checklist
```

### 🟡 Weeks 7–16 — Close the Breadth Gap

```markdown
- [ ] Sessions 3–5: core theory, deep learning, LLM/retrieval breadth   (Gap #5)
- [ ] Classical ML + statistics: the untested surface outside NLP
- [ ] Begin the publication/OSS track                                    (Gap #9 — start early, it has the longest lead time)
```

### 🟢 Weeks 17–28 — Coding & Behavioural

```markdown
- [ ] Sessions 6–7: DSA protocol + ML-from-scratch     (Gap #8 — protocol, not ability)
- [ ] Session 8: build the 12-story bank               (Gap #2 — LARGEST behavioural gap)
- [ ] Quantify every story's Result                    (Gap #3)
- [ ] ⚠️ Confirm coding language policy, then drill in that language
```

### 🔵 Weeks 29+ — Integration & Re-Score

```markdown
- [ ] Session 9: full mock loop
- [ ] Re-run Chapter 2's rubric — identical criteria, per-dimension deltas
- [ ] Re-verify Chapter 1 dossier (it expires 2027-01-20)
- [ ] Honest go/no-go
```

---

## 🔷 Gap #2 — Sourcing Behavioural Stories Without Industry Experience

Your **largest single scored gap** (Round Readiness 1/6) and the least intuitive to close, so it gets specifics. Session 8 builds the full bank; these are your raw materials.

| LP | Where your story comes from |
|---|---|
| **Dive Deep** | The RAG failure taxonomy — you decomposed failures nobody asked you to decompose |
| **Learn and Be Curious** | The six-month trajectory: each project came from a gap in the previous one |
| **Insist on the Highest Standards** | Building the calibration study — deliberately looking for where your own model was overconfident |
| **Invent and Simplify** | Writing the transformer from scratch rather than importing it |
| **Deliver Results** | ⚠️ **Weakest.** Needs a story where you shipped something under a constraint — the 16GB GPU limit is your best candidate |
| **Ownership** | A project failure you fixed without being told to |
| **Earn Trust** | Reporting a result that was worse than hoped (the DeBERTa comparison, if it went against you) |
| **Bias for Action** | Choosing to start a project with incomplete information |

**Legitimate fresher sources, in rough order of strength:** research project setbacks · course-team conflicts · TA/mentoring work · club or hostel leadership · open-source interactions · a self-inflicted engineering failure you diagnosed.

**The rule from Session 8:** stories must be *true*. Details of true stories survive interrogation; details of embellished ones do not, and a Bar Raiser is trained to keep asking.

---

## 🔷 What NOT To Do

Explicitly out of scope, because the instinct to do them is strong:

| Temptation | Why it's wrong |
|---|---|
| **Start a fifth project** | You have four and cannot report results for any. A fifth adds surface area, not evidence. **Finish reporting before building more** |
| **Study more transformer internals first** | Sessions 1–2 material is your strength. Enjoyable ≠ high-priority. It ranks below behavioural prep |
| **Grind LeetCode volume** | 700+ is already sufficient evidence. Your coding gap is *protocol*, not ability (Gap #8) |
| **Add CUDA/SQL projects to justify the skills list** | Cheaper and more honest to remove the line (Gap #4) |
| **Rewrite the resume before recovering numbers** | The numbers determine the rewrite. Order matters |

---

## 🟢 Feedback Loop

This plan is derived, so it goes stale when its inputs change. Re-run it when:

- ✅ Recruiter answers the Chapter 0 checklist → level confirmed → **re-run the Company-Relative score**
- ✅ Claim Vault filled → **re-run the Intrinsic score** (biggest single jump)
- ✅ Chapter 3 self-check done → **Depth Signals recomputed**
- ✅ Any assessment passed → Round Readiness rises
- ⚠️ Dossier expires **2027-01-20** → re-verify Chapter 1 before applying

---

## 🟢 If You Only Do One Thing

**Recover the eleven numbers in your Claim Vault this week.**

They already exist in your run logs. They are worth more than any study you could do in the same time, they unblock the resume rewrite, they fix your weakest Leadership Principle, and they turn four projects that sound like coursework into four projects that sound like results.

**Next:** [Session 1 — Project Deep-Dives: Fine-Tuning & Architecture](../tutorial/01_finetuning_and_architecture/README.md)
