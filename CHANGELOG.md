# Changelog

All notable changes to this course are documented here. Versioning follows `MAJOR.MINOR`.

## [1.3] — 2026-07-14

### Added
- **Session 2 — Project Deep-Dives: Systems & Evaluation** (`tutorial/02_systems_and_evaluation/`), replacing its locked stub:
  - Session README with scope brief (every interrogation surface for both projects) and armory table.
  - Lesson 1: Defending the RAG Failure-Mode Analysis — operational failure taxonomy as a decision tree, the recall/error dissociation, the oracle-context isolation, chunking mechanisms (dilution / fragmentation / position), judge validity, plus 6 hardest follow-ups with model answers.
  - Lesson 2: Defending Calibration Under Distribution Shift — ECE estimator pathologies, temperature scaling derived, the shift mechanism (off-manifold features → linear-head extrapolation → softmax saturation), the oracle-T exhibit, plus 6 hardest follow-ups.
  - Lesson 3: Tiered challenges across 8 core topics + the Systems Bar Raiser boss fight (10-item rubric).
  - Chapter quiz (12 questions + 2 reflections) and per-lesson interview suites (junior/mid/senior, 3 each).
  - Armory notebooks: `01_rag_failure_triage_lab.ipynb` (triage decision tree, chunk×k sweep, oracle-context decomposition, leakage confound) and `02_calibration_and_temperature_lab.ipynb` (ECE from scratch both binning schemes, reliability diagrams, NLL-vs-ECE objective, oracle-T sweep, risk–coverage).
- Session 2 Metric Vault rows in PROGRESS.md; Session 2 skill-tree nodes expanded to all 8 core topics.

### Fixed
- Calibration lab: retuned the shift simulation so the clean model is **over**confident (fitted T ≈ 2.0, matching the range Guo et al. 2017 report for ResNets). The initial parameters produced T ≈ 0.66 — an *under*confident model, contradicting the lesson it was built to demonstrate. Temperature fit bounds widened to 25.0 so the oracle-T sweep is not silently capped.
- RAG triage lab: added 20 hard distractors sharing question vocabulary. Without them the toy corpus reached recall 1.0 at k=2, so the recall/accuracy dissociation never appeared, the retrieval-miss bucket never populated, and the leakage knob had no observable effect. Leakage demo moved to k=1 where misses actually occur; oracle decomposition now reported at two operating points to show attribution depends on the operating point.

### Verified
- Both notebooks executed end-to-end on CPU. Calibration lab reproduces the phenomenon (accuracy 0.91→0.17 while confidence 0.96→0.78; oracle T* rising 1.97→12.56). RAG lab reproduces the dissociation (recall 0.40→1.00 with a widening recall–accuracy gap) and the leakage confound (identical recall, accuracy 0.23→0.53).

## [1.2] — 2026-07-14

### Added
- Sessions 2–9 shown as visible **locked skill-tree nodes**, each backed by a real stub file with scope preview and unlock conditions, so the whole 9-session map is visible from day one without dead links.
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
- Module 1 README "Armory" section: notebooks produce evidence, award 0 XP — defense remains the only paid currency.

### Fixed
- Lesson 2: corrected GQA citation from "Agarwal et al." to **Ainslie et al. 2023** (arXiv:2305.13245).
- Notebook 2: RoPE phase tables computed in float64 then cast down — fp32 phases lose ~`angle × eps` precision and visibly break shift invariance at large positions (shift=5000). The trap is documented in the notebook as interview material.

### Verified
- All three notebooks' CPU code paths executed end-to-end (GPU cells validated for structure; require Colab T4). Dependency pins last verified 2026-07-13.

## [1.0] — 2026-07-13

### Added
- Course scaffold: README (game rules, course map, constraints), PROGRESS.md (skill tree, XP ledger, streak tracker, floor-hit map, metric vault), round playbook covering all four AS round types.
- **Session 1 — Project Deep-Dives: Fine-Tuning & Architecture** (`tutorial/01_finetuning_and_architecture/`):
  - Lesson 1: QLoRA-on-ESCI defense (scope map, 6 hardest follow-ups with model answers, concept checks).
  - Lesson 2: RoPE transformer defense (scope map, 6 hardest follow-ups with model answers, concept checks).
  - Lesson 3: Tiered challenges + Architecture Bar Raiser boss fight.
  - Module quiz bank and per-lesson interview question suites (JSON).
- metadata.json with full 9-session curriculum manifest. Sessions 2–9 are registered as planned; their files land as each session is delivered on request ("next").

### Maintenance
- Quarterly review cycle established. Volatile references carry "last verified: 2026-07-13".
