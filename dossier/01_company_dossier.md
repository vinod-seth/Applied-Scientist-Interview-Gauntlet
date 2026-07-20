# Chapter 1 — The Amazon Applied Scientist Loop

| | |
|---|---|
| **Last verified** | **2026-07-20** — sources fetched live on this date |
| **Volatility** | **High.** Hiring loops change without notice; re-verify before applying |
| **Level assumed** | `[UNKNOWN]` — every bar statement below is therefore ⚠️ provisional |
| **Provenance** | Every process claim carries a tag. Untagged claims are defects |

> 📍 **How to use this chapter.** Amazon publishes more about its hiring than most employers, so much of what follows is `[OFFICIAL]`. What Amazon deliberately does *not* publish — the exact round count and content for a specific role — stays `[UNKNOWN]` here rather than being filled with a plausible guess. Your recruiter closes those gaps.

---

## 🟢 How to Read the Tags

| Tag | Meaning | Trust level |
|---|---|---|
| `[OFFICIAL]` | Amazon's own published material, fetched and dated | High |
| `[REPORTED]` | Credible secondhand | Medium — attribute, don't assert |
| `[COMMON: unverified]` | Widely circulated, no authoritative source | **Low — confirm with recruiter** |
| `[RECRUITER]` | What your recruiter told you | **Highest — overrides everything above** |
| `[UNKNOWN]` | Not established | Do not fill with a guess |

When your recruiter answers, their statement wins over every tag on this page.

---

## 🔷 The Loop, As Amazon Publishes It

Amazon names four stages: **Online Application → Assessments → Phone Screening → Interview Loop.** `[OFFICIAL: amazon.jobs/how-we-hire/interviewing-at-amazon, 2026-07-20]`

The same page states plainly: *"Our application and interview process differs from role to role."* That sentence is why this chapter refuses to give you a fixed round count.

### Published specifics

| Fact | Value | Provenance |
|---|---|---|
| Interviewers in the loop | **"anywhere from two to seven Amazon employees,"** a mix of managers, potential colleagues, and stakeholders from related teams | `[OFFICIAL: aboutamazon.com, 2026-07-20]` |
| Phone screens | **"one or two initial phone screens,"** possibly followed by role-specific assessments | `[OFFICIAL: aboutamazon.com, 2026-07-20]` |
| End-to-end duration | **"approximately three to six weeks"**; longer for more senior roles | `[OFFICIAL: aboutamazon.com, 2026-07-20]` |
| Feedback after phone screen | Within **two business days** | `[OFFICIAL: aboutamazon.com, 2026-07-20]` |
| Feedback after on-site | Within **five business days** | `[OFFICIAL: aboutamazon.com, 2026-07-20]` |
| Offer/negotiation window | A few days to ~1.5 weeks | `[OFFICIAL: aboutamazon.com, 2026-07-20]` |
| Exact AS round count & order | `[UNKNOWN]` | Not published per-role |

⚠️ **Read the "two to seven" range carefully.** It is officially published, but it spans the entire company across all roles and levels. It tells you the loop is *multi-interviewer*; it does **not** tell you your number. That remains Chapter 0 checklist item #2.

**Planning consequence:** because the shape is role-dependent, Sessions 1–9 are organised by **round type** (project depth, ML breadth, coding, behavioural), not by a fixed sequence. Round types are stable across variations; counts are not.

---

## 🔷 The Bar Raiser

Amazon publishes this program in detail, so it is one of the few loop mechanics you can prepare against with confidence.

| Fact | Provenance |
|---|---|
| Bar Raisers are **objective third-party advisers** in the interview process | `[OFFICIAL: aboutamazon.com/news/workplace/amazon-bar-raiser, 2026-07-20]` |
| They are **not involved in day-to-day interaction with the candidate**, and typically sit **outside the business** the candidate is interviewing for | `[OFFICIAL]` |
| They focus on hiring **for Amazon**, not for a specific team — maintaining a **long-term** view over an immediate hiring need | `[OFFICIAL]` |
| They are nominated by managers, peers, or other Bar Raisers, then complete **extensive training on the 16 Leadership Principles** | `[OFFICIAL]` |
| The standard: **every person hired should be better than 50% of those currently in similar roles** — that is "raising the bar" | `[OFFICIAL]` |
| The program began in 1999, originally called the *Barkeeper Program*; renamed because the goal is raising the standard, not merely maintaining it | `[OFFICIAL]` |
| Scale: **more than 3,600 Bar Raisers** across Amazon | `[OFFICIAL]` |
| Whether a Bar Raiser is in *your specific* loop | `[UNKNOWN]` — ask your recruiter |

> [!IMPORTANT]
> **The "better than 50% of current employees in similar roles" standard is the single most useful published fact in this chapter.** It reframes the whole loop: you are not being measured against a fixed syllabus, you are being measured against the people already doing the job. That is why depth beats coverage, and why an unsupported claim is disproportionately damaging — it is evidence about the *median* of your work, not a lost point.

Because the Bar Raiser sits outside the hiring team and holds a long-term view, they are structurally the person least impressed by enthusiasm for the role and most interested in whether your reasoning holds under pressure.

---

## 🔷 Round Type 1 — Project / Science Depth

**What it scores:** whether your resume's claims survive interrogation, and whether you reason like a scientist about your own work. `[COMMON: unverified]` as a discretely-named round; **certain** as a line of questioning given the Bar Raiser's LP training and Dive Deep.

**How candidates fail it:**
- Quoting a number they cannot source
- Naming a technique they cannot defend one level deeper
- Narrating chronology instead of decisions
- Treating "why not X?" as an attack rather than a design discussion

**Your exposure:** four recent, coherent NLP/LLM projects — strong material. Your specific risk is that **not one reports a result** (Chapter 2, Finding A). Against a "better than 50% of current scientists" bar, a project with no stated outcome is hard to place.

**Prep:** Sessions 1–2 (delivered).

---

## 🔷 Round Type 2 — ML Breadth

**What it scores:** fundamentals beyond your specialisation. `[COMMON: unverified]` as a distinct round.

**How candidates fail it:** wrong answers on fundamentals (bias–variance, backprop, regularisation) are close to fatal for a *Scientist* title regardless of project depth.

**Your exposure:** your resume is heavily NLP/LLM. Classical ML, statistics, and optimisation are unrepresented — untested surface rather than known weakness. ⚠️ Robotics/CV breadth may also be probed given your programme.

**Prep:** Sessions 3–5.

---

## 🔷 Round Type 3 — Coding

**What it scores:** correctness under pressure and *protocol* — clarify → approach before code → narrate → test → optimise. `[COMMON: unverified]`.

**How candidates fail it:** strong competitive programmers fail by skipping the protocol — recognising the pattern and typing immediately skips two scored beats.

⚠️ **JD-DEPENDENT:** language policy `[UNKNOWN]`. Your 700+ LeetCode is C++; ML-implementation questions typically expect Python/NumPy/PyTorch. Confirm before the loop — Chapter 0 checklist item #4.

**Prep:** Sessions 6–7.

---

## 🔷 Round Type 4 — Behavioural / Bar Raiser

**What it scores:** the 16 Leadership Principles, drilled with the depth of a technical topic. Bar Raiser training is explicitly built on them. `[OFFICIAL]`

**How candidates fail it:** this is where technically strong candidates most often fail:
- "We" answers that hide individual decisions
- Results with no quantification
- Generic, recycled stories
- Any tension between a story and the resume — assume it has been read

**Your exposure — the structural one.** No industry experience, so every story comes from academic projects, coursework, teaching, or club/hostel leadership. Workable, but requires deliberate sourcing. See Chapter 5.

**Prep:** Session 8.

---

## 🔷 The 16 Leadership Principles

`[OFFICIAL: amazon.jobs/content/en/our-workplace/leadership-principles — fetched 2026-07-20]`

Verified complete list, in published order:

| # | Principle | # | Principle |
|---|---|---|---|
| 1 | Customer Obsession | 9 | Bias for Action |
| 2 | Ownership | 10 | Frugality |
| 3 | Invent and Simplify | 11 | Earn Trust |
| 4 | Are Right, A Lot | 12 | **Dive Deep** |
| 5 | **Learn and Be Curious** | 13 | Have Backbone; Disagree and Commit |
| 6 | Hire and Develop the Best | 14 | **Deliver Results** |
| 7 | **Insist on the Highest Standards** | 15 | Strive to be Earth's Best Employer |
| 8 | Think Big | 16 | Success and Scale Bring Broad Responsibility |

The five this course weights most heavily (**bolded** above, plus Invent and Simplify):

| Principle | Why it dominates for this role |
|---|---|
| **Dive Deep** | The AS loop *is* a depth interrogation. The technical rounds' behavioural twin |
| **Learn and Be Curious** | Directly answerable via your self-taught project trajectory |
| **Invent and Simplify** | Your from-scratch RoPE build is native material |
| **Deliver Results** | ⚠️ **Your weakest LP on current evidence** — no project reports an outcome |
| **Insist on the Highest Standards** | Answerable through your failure-mode and calibration work, which are *self-criticism* projects |

> [!NOTE]
> This weighting is **the course's reading of the role**, not an Amazon-published ranking. Amazon publishes the 16 principles without stated weights. If asked which matter most, present it as your interpretation — never as internal doctrine.

---

## 🔷 What Changes By Level

`[UNKNOWN]` — level unconfirmed, so this is directional only.

| Dimension | Entry-level expectation (⚠️ provisional) |
|---|---|
| **Depth** | Deep on *your own* work; breadth expected but not encyclopaedic |
| **Scope** | Individual project ownership |
| **Ambiguity** | Given a defined problem; not expected to define it |
| **Publications** | ⚠️ JD-DEPENDENT — not assumed required |
| **System design** | ⚠️ Excluded by default; flips in if the JD names production systems |

---

## 🟢 Confirm With Your Recruiter

Every `[COMMON: unverified]` and `[UNKNOWN]` above resolves here:

```markdown
- [ ] Exact number of rounds and their order (published range is 2–7 interviewers, company-wide)
- [ ] Level, and what the bar is at that level
- [ ] Is there an ML system design round?
- [ ] Permitted coding language(s)
- [ ] Is a Bar Raiser in this loop?
- [ ] Which team / problem space?
- [ ] Are publications expected for this role?
- [ ] Timeline: campus placement cycle or rolling applications?
```

**When answered:** replace the tags with `[RECRUITER: <date>]`, then re-run Chapter 2's Company-Relative score — it is computed against this chapter.

---

## 🟢 Staleness Rule

Sources fetched **2026-07-20**. This dossier expires **2027-01-20**, or immediately upon any recruiter statement that contradicts it.

Given your ~12-month runway, **you will re-verify this chapter at least once before applying.** Re-fetch all four sources below and diff them against this page — Amazon has amended the LP list before (two principles were added in 2021), and reciting a stale list signals you did not check.

## 🟢 Sources

- [Amazon Leadership Principles](https://www.amazon.jobs/content/en/our-workplace/leadership-principles) — amazon.jobs
- [Interviewing at Amazon](https://www.amazon.jobs/content/en/how-we-hire/interviewing-at-amazon) — amazon.jobs
- [What is an Amazon bar raiser?](https://www.aboutamazon.com/news/workplace/amazon-bar-raiser) — aboutamazon.com
- [6 things every candidate should know about interviewing at Amazon](https://www.aboutamazon.com/news/workplace/amazon-interview-process-phone-screens-loops) — aboutamazon.com

**Next:** [Chapter 2 — Resume Audit & Score](02_resume_audit_and_score.md)
