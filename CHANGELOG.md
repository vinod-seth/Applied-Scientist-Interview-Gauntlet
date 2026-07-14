# Changelog

All notable changes to this course are documented here. Versioning follows `MAJOR.MINOR`.

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
