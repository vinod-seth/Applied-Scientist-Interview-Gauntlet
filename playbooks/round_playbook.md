# The Round Playbook: Four AS Round Types

| | |
|---|---|
| **Prerequisites** | README game rules |
| **Reading time** | ~20 minutes |
| **Domain tag** | Interview strategy |
| **Last verified** | 2026-07-13 |

> 📍 **Roadmap:** This file is round strategy, not technical content. Read it once before Session 1, then re-read the relevant round section before each boss fight. Session 9 executes this playbook end to end.

## 🟢 Learning Objectives

- Run the correct ordered flow for each of the four AS round types.
- Name the levers you control in each round, and use only the ones you can defend to the floor.
- Recognize the traps where control of a round is lost, before you are inside one.

---

## 🟢 Round 1: Project / Science-Depth

The round your resume was written for. One project, 45–55 minutes, five-plus levels deep.

### 🟢 Ideal flow (ordered beats) — lead with the result

1. **Result first, one sentence.** "I fine-tuned Qwen2.5-1.5B with QLoRA on ESCI 4-class relevance and reached `[FILL: macro-F1]` macro-F1 against a fully fine-tuned DeBERTa-v3 baseline at `[FILL: baseline]`." The result is the hook; the story earns the time.
2. **Problem framing, two sentences.** Why the task is non-trivial: class imbalance, ordinal labels, 16GB compute ceiling.
3. **Decision tour, not a chronology.** Walk the 3–4 decisions that mattered (quantization scheme, adapter placement, metric choice, baseline design) and *why*, not the timeline of what you did week by week.
4. **Invite the drill.** Stop talking at ~4 minutes. Depth questions coming at you is the round working as intended.
5. **Close on limitations + next step.** Unprompted honesty about what you'd do differently reads as Insist on the Highest Standards.

### 🟢 Candidate levers

- **The pitch is a menu.** Every term you volunteer ("NF4", "double quantization", "contrastive loss") is a dish the interviewer may order. Only surface hooks you can defend five levels down. If your paged-optimizer knowledge is Bronze, don't say "paged optimizers" — say "memory-efficient optimizer states" and be ready for one follow-up.
- **Choose which project leads.** You'll usually be asked "pick a project." Pick the one with the deepest Gold coverage, not the most impressive-sounding one.
- **Numbers you introduce set the terrain.** A metric you quote must have a run log behind it. A number you can't decompose (per-class, per-slice) is a liability, not an asset.
- **Own the transitions.** After surviving a chain, bridge: "…and that limitation is actually what motivated my calibration project." You choose the next battlefield.

### 🟢 Traps (where control is lost)

- **The volunteered buzzword.** The single most common way freshers lose this round: naming a technique they half-know. The interviewer *will* order from your menu.
- **Chronology drift.** Narrating week-by-week instead of decision-by-decision burns your depth time on filler.
- **Defending a fabricated or fuzzy number.** Once an interviewer smells an unsupported figure, the rest of the round becomes an audit. Unrecoverable.
- **Fighting the follow-up.** Treating "why not X?" as an attack instead of a design discussion. The correct move is to steelman X, then justify your choice — or concede honestly.

---

## 🟢 Round 2: ML Breadth

Rapid topic-hopping across fundamentals. Less depth per topic than Round 1, but zero tolerance for wrong basics.

### 🟢 Ideal flow (ordered beats)

1. **Answer the asked question first, in one or two sentences.** Breadth rounds punish preamble.
2. **Then add one level of depth voluntarily** — the equation, the trade-off, the failure mode. This is what separates "knows it" from "understands it."
3. **Flag conditions.** "That holds when classes are balanced; under imbalance I'd switch to…" Conditional answers signal real practice.
4. **Let the interviewer steer.** Topic changes are normal, not a sign you failed the last one.

### 🟢 Candidate levers

- **Precision of definitions.** Crisp one-liners (bias–variance, regularization-as-prior, why cross-entropy) are preparable and buy credibility for harder questions.
- **Bridging to your projects.** "ECE — I measured exactly this in my calibration project" converts a breadth question into home turf. Use sparingly: once or twice per round.
- **Honest floor management.** "I know the result but haven't derived it" beats a fake derivation every time, and it's scored that way.

### 🟢 Traps

- **Guessing at fundamentals.** A wrong answer on backprop or bias–variance is close to fatal in an AS loop; a confident wrong answer is worse than a slow right one.
- **Over-bridging.** Steering every question back to QLoRA looks like a one-trick script.
- **Depth-baiting.** Some interviewers offer a rabbit hole ("want to go deeper on that?") to test judgment. Take it only if you're Gold there.

---

## 🟢 Round 3: Coding

DSA and/or ML-implementation. For this candidate (700+ LeetCode) the risk is not ability — it's protocol.

### 🟢 Ideal flow (ordered beats) — clarify → approach-before-code → narrate → test → optimize

1. **Clarify** constraints, input sizes, edge semantics, and expected output *before anything else*. Two or three targeted questions, not twenty.
2. **State the approach before code.** Name the pattern, the data structure, and the complexity, and get an explicit or implicit nod. Coding an unapproved approach is how strong coders fail.
3. **Narrate while coding.** Silence is scored as absence of thought. Say what invariants you're maintaining.
4. **Test before claiming done.** Walk one normal case and two edge cases by hand, out loud. Find your own off-by-one before the interviewer does.
5. **Optimize on request or by offer.** "This is O(n log n); I can get O(n) with a hash map if you'd like" — offer, don't unilaterally rewrite.

### 🟢 Candidate levers

- **Language choice.** C++ is your fluency home; use it unless the round is ML-implementation, where Python/NumPy/PyTorch is expected. Decide *before* the loop, not live. ⚠️ JD-DEPENDENT: some AS loops require Python for all coding rounds — ask the recruiter.
- **The clarifying questions are a display.** Asking about duplicates, empty input, and value ranges *is* the first scored answer of the round.
- **Complexity statements.** Volunteering time/space complexity before being asked is a lever nobody takes away from you.

### 🟢 Traps

- **Jumping to code.** The 700-problem reflex: recognize the pattern, start typing. In an interview that skips two scored beats (clarify, approach) — you can write a perfect solution and still lose the round.
- **Silent debugging.** Freezing into quiet fix-mode when a bug appears. Narrate the hypothesis, the check, the fix.
- **Optimizing prematurely.** Getting clever before a correct baseline exists.
- **Ignoring hints.** Interviewer hints are steering, and refusing them is scored as Are Right, A Lot failing — update on evidence.

---

## 🟢 Round 4: Behavioral / Bar Raiser

The round that fails the most technically strong candidates. The Bar Raiser is from outside the team, trained to drill, and holds veto power.

### 🟢 Ideal flow (ordered beats)

1. **Listen for the LP behind the question.** "Tell me about a time you missed a deadline" is Deliver Results + Earn Trust, not a trivia prompt.
2. **STAR, with the R quantified.** Situation and Task in ≤3 sentences; Action is the bulk and must be *your* actions ("I", not "we"); Result carries a number from your logs or an honest qualitative outcome — never an invented figure.
3. **Survive the drill.** The Bar Raiser's follow-ups ("what exactly did you check?", "what did the other person say?", "what would you do differently?") are Dive Deep applied to your own story. Prepared stories must be *true* — details of true stories survive interrogation; details of embellished ones don't.
4. **Reflection close.** End each story with what changed in how you work. This is where Learn and Be Curious is scored.

### 🟢 Candidate levers

- **The story bank.** ~12 pre-built stories mapped to LPs (built in Session 8), each drilled to five follow-ups like a technical topic. Fresher sourcing: project failures (a training run that wasted a week), course-team conflicts, TA/mentoring, hostel/club leadership, research dead-ends. Real beats impressive.
- **One story, multiple LPs.** Your QLoRA memory-budget struggle is simultaneously Dive Deep, Frugality, and Deliver Results — you choose which angle to lead with per question.
- **Honest failure selection.** Choose failures with real stakes and a real lesson. "My failure is I work too hard" is an automatic credibility hit.

### 🟢 Traps

- **"We" answers.** The Bar Raiser needs *your* decision under pressure; team narration gives them nothing to score — and follow-ups will expose that you're hiding in the plural.
- **Recycled generic stories.** Bar Raisers hear the same three internet stories weekly. A specific, small, true story outperforms a grand vague one.
- **Unquantified results.** "It went well" ends a STAR at Bronze. If no metric exists, quantify scope honestly: "cut re-run time from overnight to under an hour" — only if true.
- **Contradicting your own resume.** The Bar Raiser has read it. Any tension between the story and the resume line becomes the interrogation.

---

## 🟢 Summary & Resources

- Four rounds, four different scoring systems. The project round rewards defended depth; breadth rewards precise speed; coding rewards protocol as much as correctness; behavioral rewards true, quantified, drilled stories.
- Amazon Leadership Principles (official): https://www.amazon.jobs/content/en/our-workplace/leadership-principles
- ⚠️ JD-DEPENDENT global note: entry-level AS loops typically exclude a dedicated ML system design round; teams shipping production LLM systems sometimes include one. If your recruiter confirms one, Session 5's RAG-design material upgrades from breadth answer to full round prep.
