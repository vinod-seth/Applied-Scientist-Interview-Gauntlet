# Player Progress — The Gauntlet

> Update this file after every drill session. It is your save file. XP and tiers here must trace back to an actual scored drill attempt — the game never pays for a made-up number, including here.

---

## 🟢 Skill Tree

Legend: `[ ]` untouched · `[B]` Bronze · `[S]` Silver · `[G]` Gold (boss-fight bar)

```text
SESSION 1 — FINE-TUNING & ARCHITECTURE          SESSION 2 — SYSTEMS & EVALUATION
├─ [ ] NF4 / quantile quantization              ├─ [ ] RAG failure taxonomy
├─ [ ] LoRA math (rank, alpha, placement)       ├─ [ ] Retrieval knobs (chunking, k)
├─ [ ] GPU memory accounting (16GB budget)      ├─ [ ] ECE & reliability diagrams
├─ [ ] Causal-LM-as-classifier design           ├─ [ ] Temperature scaling & its failure
├─ [ ] Macro-F1 & ESCI imbalance                └─ [ ] BOSS: Systems Bar Raiser
├─ [ ] RoPE derivation
├─ [ ] Pre-LN vs Post-LN                        SESSION 3 — CORE THEORY
├─ [ ] Contrastive loss & negatives             ├─ [ ] Bias–variance / regularization
└─ [ ] BOSS: Architecture Bar Raiser            ├─ [ ] Probabilistic foundations
                                                ├─ [ ] Optimization
SESSION 4 — DL & TRANSFORMERS                   ├─ [ ] Metrics & validation
├─ [ ] Backprop mechanics                       └─ [ ] BOSS: Theory Bar Raiser
├─ [ ] Normalization family
├─ [ ] Attention variants                       SESSION 5 — LLMs & RETRIEVAL
├─ [ ] Training pathologies                     ├─ [ ] PEFT landscape
└─ [ ] BOSS: DL Bar Raiser                      ├─ [ ] Decoding & sampling
                                                ├─ [ ] RAG design space
SESSION 6 — DSA                                 ├─ [ ] LLM evaluation
├─ [ ] Pattern fluency                          └─ [ ] BOSS: LLM Bar Raiser
├─ [ ] Interview protocol
└─ [ ] BOSS: Coding Bar Raiser                  SESSION 8 — LP & STAR
                                                ├─ [ ] Story bank (12 stories)
SESSION 7 — ML FROM SCRATCH                     ├─ [ ] STAR with quantified results
├─ [ ] Attention from blank editor              ├─ [ ] Failure/conflict sourcing
├─ [ ] Classic ML from blank editor             └─ [ ] BOSS: Behavioral Bar Raiser
└─ [ ] BOSS: Whiteboard Bar Raiser
                                                SESSION 9 — MOCK LOOP
                                                └─ [ ] FINAL BOSS: Full Loop
```

---

## 🟢 XP Ledger

| Date | Drill / chain | Depth survived | XP | Running total |
|---|---|---|---|---|
| | | | | 0 |

Scoring reminder: 10 XP per depth level survived (max 50/chain) · 5 XP for an honest floor-hit · 0 XP and a tier drop for hand-waving or unbacked numbers.

---

## 🟢 Streak Tracker

| Week | Mon | Tue | Wed | Thu | Fri | Sat | Sun |
|---|---|---|---|---|---|---|---|
| W1 | | | | | | | |
| W2 | | | | | | | |
| W3 | | | | | | | |
| W4 | | | | | | | |

Mark ✅ for any day with at least one scored drill attempt. A failed attempt keeps the streak; a blank day breaks it.

---

## 🟢 Floor-Hit Map

Log every point where you ran out of depth. This is next session's study map, not a failure record.

| Date | Session | Question chain | Depth reached (1–5) | Exact point the floor hit | Follow-up action |
|---|---|---|---|---|---|
| | | | | | |

---

## 🟢 [FILL] Metric Vault

Before Session 1's boss fight, pull these from your actual run logs. Any slot you cannot fill becomes a "qualitative-only" topic — you may describe direction and method, never a number.

| Slot | Value from run logs | Source (log/notebook path) |
|---|---|---|
| QLoRA: macro-F1 (test) | [FILL] | |
| QLoRA: per-class F1 (E/S/C/I) | [FILL] | |
| DeBERTa-v3 baseline: macro-F1 | [FILL] | |
| QLoRA: trainable-param % and count | [FILL] | |
| QLoRA: LoRA rank, alpha, target modules | [FILL] | |
| QLoRA: peak GPU memory observed | [FILL] | |
| ROPE model: layers/heads/d_model/params | [FILL] | |
| ROPE model: QQP eval metric + value | [FILL] | |
| ROPE model: baseline compared against (if any) | [FILL] | |
| Contrastive loss: exact formulation + temperature/margin | [FILL] | |
