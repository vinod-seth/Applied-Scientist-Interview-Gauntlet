# Changelog

All notable changes to this course are documented here. Versioning follows `MAJOR.MINOR`.

## [1.9] — 2026-07-29

Guideline-compliance pass over Session 5, run against all three reference packs after delivery. `interview_prep/01` §8 states that round-prep chapters apply the standard technical-course guidelines in full, and §5's glossary-density exception is scoped to the Chapter 3 tech-stack list only — so generic guidelines 01, 02 and 04 bind here and were not fully met on first delivery.

### Fixed
- **Citation error.** Card et al., *With Little Power Comes Great Responsibility* was cited as arXiv:2010.02405, which is an unrelated few-shot NER paper. Corrected to **arXiv:2010.06595** in the lesson and in both interview suites. Found by verifying all **55 arXiv IDs** in the session against the arXiv API — every other identifier resolved to the paper claimed.
- Two papers were cited by method name rather than title; now carry their real titles (*Precise Zero-Shot Dense Retrieval without Relevance Labels* for HyDE, *DeepSeekMath* for GRPO). Self-consistency dated 2022 per guideline 01's canonical list; `Ben Zaken` corrected to `Ben-Zaken`.

### Changed
- **Learning Objectives** (guideline 02 §2.4) added to all six Session 5 lessons — 4–5 measurable outcomes each, placed after the metadata table. Visible word counts rise to ~870–900, still well inside the 20-minute rule.
- **Hover definitions** (guideline 04) brought to 6–11 per lesson, up from 2–5. Newly annotated load-bearing terms include SFT, RLHF, PPO, DPO, RLAIF, GRPO, Bradley–Terry, reward hacking, RRF, nDCG, HNSW, approximate nearest neighbour, HyDE, cross-encoder, oracle-context run, Cohen's κ, groundedness, discordant, canary strings, constrained decoding, rejection sampling and beam search. Every definition is ≤25 words, none repeats the term as a prefix except to expand an acronym, none sits in a heading, and prerequisite-level terms (gradient, cross-entropy, softmax) are deliberately left unannotated.
- **Lesson-level references** reformatted to the guideline 01 §3 citation checklist — authors, year, *italicised exact paper title*, and a working link — replacing the compressed `Author Year (arXiv:ID)` form. 55 references across five lessons.
- **Session 4 annotated.** Its five lessons carried **zero** `<abbr>` definitions, the largest single deviation from guideline 04 in the course; each now carries 5–8, covering gradient checkpointing, saturating activations, initialization, gradient clipping, running statistics, GroupNorm, RMSNorm, internal covariate shift, landscape smoothing, KV cache, MQA, GQA, HBM, online softmax, PagedAttention, arithmetic intensity, mixed precision, loss scaling, warmup, the single-batch overfit test and dying ReLU.

### Known deviations (deliberate, recorded for reviewers)
- **No `companies` tags** on interview questions. Generic guideline 05 §6 asks for them; `interview_prep` §9 explicitly overrides this for Resume-to-Offer courses, since the target company is fixed in Chapter 0.
- **No mixed-audience Lesson Roadmap table** (guideline 02 §2.3, which scopes it to mixed courses). This is a single-reader course and the roadmap was rejected during the Session 1 format review.
- **Uniform lesson skeleton.** The drill-card structure repeats deliberately across lessons; `pedagogical_technical_critique` asks for varied concept-check formatting, and a reviewer may read the consistency as templating.
- **Course-level items still outstanding:** guideline 05 §5's cumulative final assessment and capstone self-assessment rubric are Session 9's deliverables and remain unbuilt.

## [1.8] — 2026-07-29

### Added
- **Session 5 — LLMs, Fine-Tuning & Retrieval** (`tutorial/05_llms_finetuning_retrieval/`), replacing its locked stub. Where Session 4 rewarded derivation, this session rewards **judgment under a stated constraint**: most of its questions have several defensible answers, and the score comes from naming the trade-off and the measurement that would settle it.
  - Session README with the scope brief framed as "the version everyone gives vs. the version that scores", plus an explicit note on how this session relates to Sessions 1 and 2 — it places *your* QLoRA run inside the PEFT landscape and *your* RAG study inside the retrieval design space rather than repeating either.
  - Lesson 1: The PEFT Landscape — the ≈16 bytes/param breakdown and what freezing removes, LoRA stated exactly ($W_0 + \frac{\alpha}{r}BA$, $B=0$ at init, $r(d+k)$ parameters), the mergeability axis across DoRA/IA³/adapters/soft prompts, multi-adapter serving, and the three cases where PEFT is the wrong choice.
  - Lesson 2: Decoding & Sampling — why likelihood-maximizing text degenerates, top-k's failure mode against top-p and min-p, the reshape-then-truncate order of operations, repetition diagnosed in three causes (including the missing-EOS bug), speculative decoding as an **exact** bandwidth optimization, and constrained decoding as a syntax-only guarantee.
  - Lesson 3: Scaling Behavior & the Post-Training Stack — $C \approx 6ND$ and the Chinchilla allocation, why serving-oriented models are deliberately overtrained, emergence as partly a metric artifact, and the capability → format → ranking pipeline with both the RLHF and DPO objectives stated and the off-policy trade named.
  - Lesson 4: The RAG Design Space — the two-stage recall/precision split, hybrid retrieval with rank-based RRF, why cross-encoders cannot be indexed, the RAG vs fine-tuning vs long-context decision rule, and ANN search as a second independent recall loss.
  - Lesson 5: Evaluating LLM Systems — the two gates (instrument validity, then statistics), paired analysis with McNemar's, judge bias and the human audit with Cohen's $\kappa$, benchmark contamination with the GSM1k evidence, four-layer suite design, and what arena ratings hide.
  - Lesson 6: Mock Round — 14 turn-by-turn questions, three **set-piece design questions** (design the retrieval system; design the evaluation; spend one 24 GB GPU), a 15-row self-assessment, and a synthesis question tracing knowledge/behavior → retrieve/adapt → decode → evaluate.
  - Chapter quiz (12 questions + 2 reflections, one of which re-audits the candidate's own RAG study under Lesson 5's methodology) and six interview suites (junior/mid/senior, 3 each) totalling **54 questions**, each carrying a `rehearsal` block.
  - Armory notebook `llm_systems_lab.ipynb` (CPU-only, no model downloads): the 112 → 14.4 → 3.9 GB memory ladder computed, an SVD study showing a rank-12 update captured at ~98% with under 5% of the parameters, top-p's nucleus measured at 4 vs ~25,000 tokens across two contexts and its interaction with temperature, compute-optimal allocation solved numerically and then re-solved with inference cost, a hybrid retriever scoring 8/8 recall@3 against 5/8 and 6/8 for its components, and McNemar's test, a power curve, and a selection-bias simulation for the evaluation drills.

### Changed
- README course map: Session 5 marked delivered; version bumped to 1.8.
- PROGRESS.md mastery tracker: Session 5's six core topics replace the four placeholders, and the two-column layout re-aligned.
- metadata.json: `module_05` unlocked with eight lessons and per-lesson modes; lesson count 37 → 44; the Session 6 stub's unlock text no longer refers to "clearing" an assessment, which the de-gaming in 1.7 had left behind.
- Session 4's mock round now links forward to the Session 5 README instead of the deleted locked stub.

### Fixed
- Lesson 3 and the Lab both note that the **≈20 tokens-per-parameter rule rests on Chinchilla's isoFLOP analysis, not its parametric fit**: solving the constrained optimization with the paper's published coefficients returns a substantially higher ratio, and Besiroglu et al. 2024 found those coefficients irreconcilable with the paper's own data. The Lab performs the calculation so the discrepancy is visible rather than asserted.

### Removed
- `tutorial/05_llms_finetuning_retrieval/00_locked.md`.

## [1.7] — 2026-07-29

### Added
- **Session 4 — ML Fundamentals: Deep Learning & Transformers** (`tutorial/04_deep_learning_transformers/`), replacing its locked stub. Where Session 3 rewarded compression, this session rewards **derivation**: the questions have exact answers and the interviewer knows them.
  - Session README with the scope brief framed as "the version everyone gives vs. the version that scores," plus an explicit note on how this session relates to Session 1 (which defended *your* transformer) — it assumes that material and pushes past it rather than repeating it.
  - Lesson 1: Backpropagation & Gradient Flow — the linear-layer derivation with shapes, why activations must be stored (and what checkpointing trades), vanishing/exploding as a *product of Jacobians*, and the residual derivative $I + \partial F/\partial h$.
  - Lesson 2: The Normalization Family — LayerNorm's exact computation and axis, three distinct reasons transformers can't use BatchNorm, RMSNorm's deletion, GroupNorm as the middle ground, and the fact that the internal-covariate-shift explanation was **overturned** by Santurkar et al. 2018.
  - Lesson 3: Attention Variants — the $O(n^2)$ arithmetic and where the FFN crossover sits, KV-cache sizing worked on a real 7B configuration, MHA/MQA/GQA trade-offs, and FlashAttention as an **exact** I/O optimization (explicitly distinguished from sparse and linear attention).
  - Lesson 4: Training Pathologies — a diagnosis-first playbook (symptom → mechanism → discriminating test → act), covering NaN spikes, flat loss, overfitting response ordering, mixed-precision failures, dying ReLU, and clipping's limits.
  - Lesson 5: Mock Round — 12 turn-by-turn questions, two dedicated **board questions** (backprop for an MLP; KV-cache arithmetic), an honest self-assessment table, and a synthesis question tracing gradient flow → depth → quadratic attention → inference memory.
  - Chapter quiz (12 questions + 2 reflections) and five interview suites (junior/mid/senior, 3 each) totalling **45 questions**, each suite carrying a `rehearsal` block per the updated guidelines.
  - Armory notebook `deep_learning_lab.ipynb`: hand-derived gradients verified against autograd (exact match), gradient norms measured across 40 layers with and without residuals, LayerNorm's batch-independence demonstrated against BatchNorm's, the $O(n^2)$ and KV-cache tables computed, and fp16 overflow/underflow reproduced with the loss-scaler fingerprint.

### Changed
- PROGRESS.md mastery tracker: Session 4's seven core topics added and the two-column layout re-aligned.
- README course map: Session 4 marked delivered; header table corrected to drop a leftover "assessment that gates the next" (superseded in 1.6) and refresh the version and verification date.

### Changed
- **The whole course is now de-gamed**, so Sessions 1–2 match Session 3 and the updated `course-generation-guidelines/interview_prep` rules. Removed XP, medal tiers, boss fights, and progress gating throughout; kept every drill, interrogation chain, and rubric intact — only the framing changed.
  - **Mastery levels** are now plain language everywhere: **Level 1 — state it correctly**, **Level 2 — derive it**, **Level 3 — defend it against five follow-ups**. Level 3 is described as the hiring bar rather than a medal to collect. PROGRESS.md marks change from `[B]`/`[S]`/`[G]` to `[k]`/`[d]`/`[D]`.
  - **No session is locked.** Session assessments are now "mock rounds" that exist to show where your depth runs out, not to gate the next session. The six locked stubs (Sessions 4–9) lost their "reach Gold and pass" unlock conditions.
  - **Session 1 and 2 lesson 3 renamed** from `03_tiered_challenges_and_boss_fight.md` to `03_depth_drills_and_mock_round.md`, with metadata paths and all cross-links updated.
  - **Resume-score bands** in the Chapter 2 dossier renamed from Gold/Silver/Bronze to **Strong / Sound / Needs Work**, matching the guideline rubric. The candidate's scores are unchanged (58/100 is now "Needs Work" rather than "Bronze").
  - Medal names used as prose quality descriptors ("the Gold move", "Silver at best", "zero Bronze hooks") replaced with plain equivalents.
- PROGRESS.md mastery tracker rebuilt: Session 3's real core topics added, and the two-column ASCII layout re-aligned after drifting.

### Preserved deliberately
- **"Bar Raiser"** and **"Dive Deep"** remain throughout — they are real Amazon interview vocabulary, not game mechanics.
- **"Gold passage" / "gold labels" / "gold context"** remain in the RAG material — standard ML terminology for ground-truth data, unrelated to the medal tiers.
- The repository name (`Applied-Scientist-Interview-Gauntlet`) is unchanged, since every Colab badge and notebook link resolves through it.

## [1.5] — 2026-07-28

### Added
- **Session 3 — ML Fundamentals: Core Theory** (`tutorial/03_core_theory/`), replacing its locked stub. This session targets the ML-breadth round's rapid-fire fundamentals, where the bar is *correct → compressed → caveated*, not depth-per-topic:
  - Session README with the scope brief framed as "the headline everyone knows vs. the caveat that actually scores," plus the connective chain **capacity → prior → penalized likelihood → optimizer → honest measurement**.
  - Lesson 1: Generalization — the bias–variance decomposition (with the irreducible-noise floor and double descent as its limit), regularization as variance-for-bias, and the L2 = Gaussian prior / L1 = Laplace prior derivation.
  - Lesson 2: Estimation & Optimization — cross-entropy as negative log-likelihood, MLE→MAP, the SGD→momentum→adaptive→Adam→AdamW genealogy with the problem each step fixes, and why decoupled weight decay is not cosmetic.
  - Lesson 3: Metrics & Validation Design — metric choice as a statement about cost structure, the ROC-vs-PR imbalance mechanism, and split design as deployment rehearsal (grouped/temporal splits, the leakage audit rule).
  - Lesson 4: Rapid-fire rehearsal plus a 12-question **mock breadth round** delivered turn-by-turn, an honest self-assessment table, and the synthesis "chain question."
  - Chapter quiz (12 questions + 2 reflections) and four interview suites (junior/mid/senior, 3 each) totalling **36 questions**.
  - Armory notebook `core_theory_lab.ipynb`: bias–variance measured by Monte-Carlo resampling, L1/L2 shrinkage paths, numerical verification that Ridge equals Gaussian-prior MAP, optimizer trajectories on an ill-conditioned quadratic, the Adam-vs-AdamW decay sweep, and the ROC/PR imbalance trap.
- Rehearsal drills embedded in every Session 3 lesson, using the iterative coaching loop (coach evaluates → you refine → re-evaluate until an interviewer would move on).

### Changed
- Session 3 uses the **simplified, de-gamed format** — no XP, tiers, or assessment gating — matching the updated `course-generation-guidelines/interview_prep` rules. (Sessions 1–2 and the course-level docs were brought to the same format in 1.6.)

### Fixed
- Core Theory Lab optimizer demo: the initial configuration compared SGD, momentum, and Adam at learning rates where **momentum and Adam both finished worse than plain SGD**, contradicting the lesson they were built to demonstrate. SGD and momentum now share a matched learning rate (the fair comparison, where momentum wins ~6×), and the adaptive methods use their own. The table now reports loss at three checkpoints so momentum's characteristic early overshoot is visible rather than looking like a failure.
- Core Theory Lab AdamW demo: the original two-coordinate version showed both coordinates landing on identical values, so it did not actually demonstrate the decoupling claim. Replaced with a decay-coefficient sweep, which shows the sharper and more useful result — under L2-in-Adam the final weight is **unchanged to five decimal places as λ sweeps 0 → 1.0** (the knob normalizes itself away), while AdamW's weight shrinks monotonically.

## [1.3] — 2026-07-14

### Added
- **Session 2 — Project Deep-Dives: Systems & Evaluation** (`tutorial/02_systems_and_evaluation/`), replacing its locked stub:
  - Session README with scope brief (every interrogation surface for both projects) and armory table.
  - Lesson 1: Defending the RAG Failure-Mode Analysis — operational failure taxonomy as a decision tree, the recall/error dissociation, the oracle-context isolation, chunking mechanisms (dilution / fragmentation / position), judge validity, plus 6 hardest follow-ups with model answers.
  - Lesson 2: Defending Calibration Under Distribution Shift — ECE estimator pathologies, temperature scaling derived, the shift mechanism (off-manifold features → linear-head extrapolation → softmax saturation), the oracle-T exhibit, plus 6 hardest follow-ups.
  - Lesson 3: Tiered challenges across 8 core topics + the Systems Bar Raiser assessment (10-item rubric).
  - Chapter quiz (12 questions + 2 reflections) and per-lesson interview suites (junior/mid/senior, 3 each).
  - Armory notebooks: `01_rag_failure_triage_lab.ipynb` (triage decision tree, chunk×k sweep, oracle-context decomposition, leakage confound) and `02_calibration_and_temperature_lab.ipynb` (ECE from scratch both binning schemes, reliability diagrams, NLL-vs-ECE objective, oracle-T sweep, risk–coverage).
- Session 2 Metric Vault rows in PROGRESS.md; Session 2 mastery-tracker nodes expanded to all 8 core topics.

### Fixed
- Calibration lab: retuned the shift simulation so the clean model is **over**confident (fitted T ≈ 2.0, matching the range Guo et al. 2017 report for ResNets). The initial parameters produced T ≈ 0.66 — an *under*confident model, contradicting the lesson it was built to demonstrate. Temperature fit bounds widened to 25.0 so the oracle-T sweep is not silently capped.
- RAG triage lab: added 20 hard distractors sharing question vocabulary. Without them the toy corpus reached recall 1.0 at k=2, so the recall/accuracy dissociation never appeared, the retrieval-miss bucket never populated, and the leakage knob had no observable effect. Leakage demo moved to k=1 where misses actually occur; oracle decomposition now reported at two operating points to show attribution depends on the operating point.

### Verified
- Both notebooks executed end-to-end on CPU. Calibration lab reproduces the phenomenon (accuracy 0.91→0.17 while confidence 0.96→0.78; oracle T* rising 1.97→12.56). RAG lab reproduces the dissociation (recall 0.40→1.00 with a widening recall–accuracy gap) and the leakage confound (identical recall, accuracy 0.23→0.53).

## [1.2] — 2026-07-14

### Added
- Sessions 2–9 shown as visible **locked mastery-tracker nodes**, each backed by a real stub file with scope preview and unlock conditions, so the whole 9-session map is visible from day one without dead links.
- Metadata curriculum expanded to list every reachable chapter (Start Here module: overview/progress/playbook; Session 1 module: overview, 3 lessons, quiz).

### Fixed
- Restored chapters lost when the curriculum was over-trimmed: the portal renders only what the curriculum lists, so quiz, playbook, progress tracker, and session overview had silently disappeared from the sidebar.
- Colab launch links converted from image badges to plain text links — the portal's markdown renderer emitted `href="#"` for image-wrapped links.

## [1.1] — 2026-07-14

### Added
- **Armory notebooks** (interactive mode) for Session 1, paired under `modes.notebook` in metadata.json with Colab launch badges:
  - `01_memory_budget_verifier.ipynb` — paper memory budget vs. live `torch.cuda` allocator readings for Qwen2.5-1.5B in NF4; real trainable-param count via PEFT (Colab T4).
  - `02_architecture_unit_tests.ipynb` — five executable architecture claims: RoPE shift invariance, value-rotation counterfactual, padding-mask equivalence, last-token indexing bug, 1/√d_k variance measurement (CPU).
  - `03_metric_vault_extractor.ipynb` — per-class F1, confusion matrix, alignment/uniformity diagnostics from the learner's own artifacts; explicit SYNTHETIC-DEMO vs. REAL mode with a record-nothing rule for demo outputs.
- Module 1 README "Armory" section: notebooks produce evidence, not mastery on their own — defense in drills remains the only measure.

### Fixed
- Lesson 2: corrected GQA citation from "Agarwal et al." to **Ainslie et al. 2023** (arXiv:2305.13245).
- Notebook 2: RoPE phase tables computed in float64 then cast down — fp32 phases lose ~`angle × eps` precision and visibly break shift invariance at large positions (shift=5000). The trap is documented in the notebook as interview material.

### Verified
- All three notebooks' CPU code paths executed end-to-end (GPU cells validated for structure; require Colab T4). Dependency pins last verified 2026-07-13.

## [1.0] — 2026-07-13

### Added
- Course scaffold: README (course structure, mastery levels, constraints), PROGRESS.md (mastery tracker, practice log, consistency tracker, gap log, metric vault), round playbook covering all four AS round types.
- **Session 1 — Project Deep-Dives: Fine-Tuning & Architecture** (`tutorial/01_finetuning_and_architecture/`):
  - Lesson 1: QLoRA-on-ESCI defense (scope map, 6 hardest follow-ups with model answers, concept checks).
  - Lesson 2: RoPE transformer defense (scope map, 6 hardest follow-ups with model answers, concept checks).
  - Lesson 3: Tiered challenges + Architecture Bar Raiser assessment.
  - Module quiz bank and per-lesson interview question suites (JSON).
- metadata.json with full 9-session curriculum manifest. Sessions 2–9 are registered as planned; their files land as each session is delivered on request ("next").

### Maintenance
- Quarterly review cycle established. Volatile references carry "last verified: 2026-07-13".
