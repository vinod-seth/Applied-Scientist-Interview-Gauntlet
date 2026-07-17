# Lesson 1 — Defending the RAG Failure-Mode Analysis

| | |
|---|---|
| **Prerequisites** | Session 2 README; your RAG pipeline code and eval artifacts open |
| **Reading time** | ~25 min + drills |
| **Domain tag** | Retrieval / LLM Systems / Experimental Methodology |
| **Resume line under fire** | "Built a RAG pipeline (bge-small + FAISS) over a labeled QA corpus and instrumented it to trace every wrong answer to its root cause… isolating generation rather than retrieval as the dominant error source" |

🔬 **Interactive companion:** [▶ Open the RAG Failure Triage Lab in Colab](https://colab.research.google.com/github/vinod-seth/Applied-Scientist-Interview-Gauntlet/blob/main/tutorial/02_systems_and_evaluation/01_rag_failure_triage_lab.ipynb)

> 📍 **Roadmap:** All technical track. This bullet's strongest word is "every" — *trace every wrong answer to its root cause* is a methodology claim, and methodology claims get audited, not admired. The lesson arms you for that audit.

## 🟢 Learning Objectives

- State operational (instrumentation-level) definitions of retrieval-miss, retrieved-but-ignored, and hallucination, including the boundary cases where they overlap.
- Defend the "generation is the dominant error source" conclusion with sweep evidence and an oracle-context experiment — and state what would falsify it.
- Answer the six hardest follow-ups at AS depth, including attacks on your judge and your labels.

---

## 🟢 The Interrogation Logic of This Bullet

Your other projects claim you built something; this one claims you *proved* something. An interviewer reads "isolating generation rather than retrieval as the dominant error source" and hears a causal claim extracted from observational sweeps — which is precisely the kind of claim an Applied Scientist is paid to make carefully and paid more to attack. Expect every question to take one of three shapes: **definition attacks** (your failure taxonomy has fuzzy boundaries), **measurement attacks** (your correctness judge is itself a model with an error rate), and **inference attacks** (your conclusion has confounds you didn't control).

The good news: if your instrumentation was honest, this is the easiest project to defend, because your answers are experiments, not opinions.

Background citation for the architecture itself: Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (https://arxiv.org/abs/2005.11401).

---

## 🔷 Peer-Level Refresher 1: The Taxonomy, Operationalized

"Retrieval-miss vs. retrieved-but-ignored vs. <abbr title="The model states a claim supported by nothing in the retrieved context — as opposed to grounding on the wrong retrieved evidence.">hallucination</abbr>" sounds clean until you write the classifier. The <abbr title="Operational definition: a rule stated as an executable check on observable signals, rather than a conceptual description.">operational definitions</abbr> require **<abbr title="Ground-truth annotations marking which passage actually contains the answer to each question. Without them, retrieval quality cannot be measured.">gold-passage labels</abbr>** — for each question, which passage(s) contain the answer:

| Failure mode | Instrumentation test | Requires |
|---|---|---|
| **Retrieval-miss** | No gold passage appears in the top-k retrieved chunks | Gold-passage IDs, chunk→passage mapping |
| **Retrieved-but-ignored** | Gold passage present in context, but the answer is wrong and the correct answer *is derivable* from the provided context | Gold IDs + answer-correctness judge |
| **Hallucination** | Answer contains claims supported by *neither* the retrieved context *nor* the gold answer | Claim-level support check against context |

Three boundary problems you must volunteer before the interviewer finds them:

1. **Ignored vs. hallucinated overlap.** A wrong answer with the gold passage in context could be the generator ignoring the context *or* overriding it with parametric memory. Distinguishing them needs a support test on the wrong answer itself: if the wrong answer is traceable to another retrieved chunk, it's a grounding/selection failure; if it's traceable to nothing, it's hallucination. What your pipeline did: `[FILL: your disambiguation rule]`.
2. **Chunking breaks gold labels.** Gold labels live at passage level; retrieval happens at chunk level. A chunk containing *half* the gold passage may or may not suffice to answer — so "gold in context" needs a definition (any overlap? full answer span present?). Yours: `[FILL]`.
3. **Partially correct answers.** A three-fact answer with two facts right defeats binary classification. Whether you scored at answer level or claim level changes every downstream number: `[FILL]`.

The honest framing that survives five follow-ups: *"The taxonomy is a decision tree over instrumented checks, each check has an error rate, and I can tell you both the rules and their failure cases."*

---

## 🔷 Peer-Level Refresher 2: The Sweep, and What Licenses Your Conclusion

The resume says chunk size and retrieval depth were swept and generation emerged as the dominant error source. The reasoning chain that makes this defensible:

**Step 1 — retrieval error is directly measurable.** Retrieval-miss rate = 1 − <abbr title="The fraction of questions whose gold passage appears somewhere in the top-k retrieved results. Measures retrieval quantity, not whether the answer was used.">recall@k</abbr> against gold labels. As k grows and chunking is tuned, miss rate falls; you can drive the retrieval side of the taxonomy toward zero and *watch what's left*. If total error stays high while miss rate goes low, the residual belongs to generation. Your curve: `[FILL: recall@k at your operating points]`.

**Step 2 — the oracle-context experiment.** The clean <abbr title="A condition that removes one stage as a possible explanation, so any error that survives must belong to the remaining stages.">isolation test</abbr>: bypass retrieval entirely and hand the generator the gold passage. The remaining error rate is the **<abbr title="The error rate that survives perfect retrieval. It is the part of the loss no retrieval improvement can ever fix.">generation floor</abbr>** — no retrieval explanation available. If you ran it: `[FILL: oracle accuracy vs. end-to-end accuracy]`. If you didn't, say so and name it as the missing experiment *before being asked* — it's the single most likely follow-up on this project.

**Step 3 — <abbr title="Knowing in advance which observation would prove your claim wrong. A claim nothing could refute is not a scientific claim.">falsifiability</abbr>.** Know what would overturn the conclusion: a stronger generator closing most of the gap (then the finding was generator-specific, not architectural); a judge audit revealing "ignored" was mostly judge error; or gold-label noise inflating the miss rate. Naming these unprompted is the difference between a finding and a belief.

**The chunk-size mechanism** (know it as mechanism, not folklore): small chunks sharpen embedding specificity but fragment evidence across chunks — answers needing two sentences now need two retrieval slots, aggravating *retrieved-but-insufficient* cases. Large chunks <abbr title="Dilution: averaging many topics into one embedding vector, weakening its similarity to any single specific query.">dilute</abbr> the embedding (mean-pooled signal over more topics → retrieval-miss) and push relevant sentences into the middle of long contexts, where generators demonstrably use them worst — Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts* (https://arxiv.org/abs/2307.03172). Chunk size and k also interact through the context budget: doubling both can *hurt*, because context grows past what the generator handles well. What was held fixed in your sweep: `[FILL]`.

## 🟢 Concept Check

Your pipeline's total error rate barely moves as you sweep k from 3 to 20, while measured recall@k climbs from 0.61 to 0.94. What's the strongest inference?

* [ ] The retriever is broken
* [ ] The corpus has label noise
* [x] The binding error source is downstream of retrieval — most residual failures are generation-side (ignored context or hallucination)
* [ ] k=3 was already optimal

An answer is wrong; the gold passage IS in the retrieved context; the wrong answer's content matches a *different* retrieved chunk. Best classification?

* [ ] Retrieval-miss
* [x] Retrieved-but-ignored (a grounding/selection failure among retrieved chunks), not hallucination — the content is supported by context, just the wrong part of it
* [ ] Hallucination
* [ ] Judge error, discard the example

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Q1: option 3.** Recall rising while error stays flat is exactly the dissociation that licenses "generation-dominant": you improved the retrieval quantity and the system didn't care. Options 1 and 4 contradict the recall curve; label noise (option 2) would suppress *measured* recall, not decouple it from error.

**Q2: option 2.** The distinguishing test is support: the claim is traceable to retrieved text, so the generator grounded — on the wrong evidence. Hallucination is reserved for claims supported by nothing retrieved. This distinction matters practically: selection failures respond to reranking and prompt-level source discipline; true hallucination responds to neither.
</details>

---

## 🔷 Peer-Level Refresher 3: The Stack — bge-small, FAISS, and the Judge

**bge-small.** A ~33M-parameter English embedding model from the BGE family — Xiao et al. (2023), *C-Pack: Packaged Resources To Advance General Chinese Embedding* (https://arxiv.org/abs/2309.07597) — strong on retrieval benchmarks for its size. Two facts you must know cold: (1) BGE v1 models expect a **query-side instruction prefix** ("Represent this sentence for searching relevant passages: …") — omitting it measurably degrades retrieval, and whether you used it is a guaranteed check-question: `[FILL]`. (2) Embeddings should be L2-normalized so FAISS inner product equals cosine. Index type at your corpus scale (likely `IndexFlatIP` — exact search, no recall loss to explain): `[FILL: index + corpus size]`.

**The judge.** Whatever decided "wrong answer" — exact match, token-F1, an LLM judge — is itself a measurement instrument with bias. Exact match punishes paraphrase (inflates "ignored"); LLM judges are lenient toward fluent wrong answers and biased toward their own phrasings. The defensible position: name your judge, name its known bias direction, and report a human spot-check: `[FILL: judge type + sample audit agreement, or an honest "not audited — first thing I'd add"]`.

**<abbr title="The generator already memorized the corpus facts during pretraining, so answering correctly no longer proves retrieval worked.">Leakage</abbr>.** The generator's pretraining may contain your QA corpus's facts. A <abbr title="Run the generator with no retrieved context at all. Whatever it answers correctly, it knew from memory — the control that quantifies leakage.">closed-book baseline</abbr> (no retrieval at all) measures this: if closed-book accuracy is already high, "retrieval-miss" cases can still be answered right and your taxonomy's denominators shift. Closed-book number: `[FILL or QUALITATIVE-ONLY]`.

---

## 🔷 The Six Hardest Follow-Ups (With Model Answers)

### 🔷 Follow-up 1: "Walk me through your classifier for a single wrong answer. Exact rules, in order."

**Why they ask:** "traced every wrong answer" is an instrumentation claim; they want the decision tree, not the taxonomy names.

**Model answer:** the Refresher 1 table as an ordered procedure — check gold-in-context first (by `[FILL: your overlap rule]`), then answer-support against context, then support against any retrieved chunk — ending with the two boundary rules you chose and the case counts per bucket: `[FILL: failure-mode distribution]`. Close by naming the taxonomy's own error rate (from your spot-check or as the acknowledged gap).

**The chain:** "what fraction of 'ignored' cases did a human confirm?" → "how does a multi-hop question classify?" → "show me one real example from each bucket" (have three memorized — real ones).

### 🔷 Follow-up 2: "Your headline conclusion — generation dominates. Prove it to me, then tell me what would falsify it."

**Why they ask:** this is the resume's boldest claim; the answer separates measurement from inference.

**Model answer:** Refresher 2's three steps in order: the recall-vs-error dissociation from the sweep (`[FILL]`), the oracle-context result (`[FILL or the honest gap]`), then falsifiers unprompted: stronger generator, judge audit, label noise. If you have only the sweep and not the oracle test, scope the claim aloud: "dominant *at my operating points, for my generator* — the oracle experiment is how I'd make it unconditional."

**The chain:** "which generator, and would GPT-class models flip the result?" → "how many eval questions — is your dominant-mode gap even significant?" (`[FILL: n]`; binomial CI on bucket proportions) → "you tuned retrieval on the same set you diagnosed on — is that a problem?" (yes: adaptive overfitting of the sweep; hold-out or fresh-split defense).

### 🔷 Follow-up 3: "Mechanistically — not empirically — why would chunk size change your failure distribution?"

**Why they ask:** sweep results without mechanism are pattern-matching; the mechanism predicts what happens *off* the sweep grid.

**Model answer:** the three mechanisms from Refresher 2 — embedding dilution (large chunks average topics → miss), evidence fragmentation (small chunks split multi-sentence answers → insufficient context), and position effects (large chunks + high k push evidence mid-context, where models use it worst; cite Liu et al. 2023). Then the interaction: chunk×k jointly determine context length, so the sweep axes aren't independent — say what you held fixed (`[FILL]`).

**The chain:** "predict: chunk 128 → 512 at fixed k=10 — which bucket grows?" → "does overlap between chunks fix fragmentation, and what does it cost?" (recall gain vs. duplicated context and index size) → "why not sentence-boundary chunking?" (have a position, either way).

### 🔷 Follow-up 4: "Your judge is a measurement device. What's ITS error rate, and how does judge error masquerade as each failure mode?"

**Why they ask:** the whole taxonomy sits on top of the judge; scientists are expected to calibrate their instruments.

**Model answer:** name the judge (`[FILL]`), its bias direction — exact-match/F1 over-flags paraphrases as wrong (inflating "ignored"), LLM judges under-flag fluent hallucinations (deflating "hallucination") — and the audit: `[FILL: human agreement on a sample, or "unaudited" + the audit design: stratified sample per bucket, two annotators, report agreement]`. The Gold move is showing you know which *direction* each bucket is biased, not just that bias exists.

**The chain:** "if judge precision on 'wrong' is 85%, rebuild your headline numbers" (propagate: some 'ignored' were actually right) → "would you trust an LLM judging its own family's output?" → "design the cheapest audit that would change your conclusion."

### 🔷 Follow-up 5: "Why bge-small? And what retrieval quality did you leave on the table?"

**Why they ask:** tests whether component choices were decisions or defaults.

**Model answer:** "Deliberate floor-setting: small, fast, well-benchmarked for its size — and since my goal was failure *attribution*, an imperfect retriever generates the retrieval-miss cases I needed to study; a perfect retriever would have made half the taxonomy unobservable. The known costs: I used `[FILL: prefix yes/no]` — BGE v1 expects the query instruction prefix, and that alone moves recall. Upgrades I deliberately didn't take: a larger embedder, a cross-encoder reranker on the top-k (usually the single biggest recall@small-k gain), hybrid BM25+dense. Each attacks retrieval-miss — which my analysis showed was `[FILL: the smaller bucket / whatever your data says]`, so the upgrade priority follows the taxonomy, not fashion."

**The chain:** "what's the reranker's latency cost and where does it sit?" ⚠️ JD-DEPENDENT (production teams) → "normalized embeddings? metric? verify cosine=IP" → "how would you detect that the *corpus*, not the retriever, is the ceiling?" (answerability audit: can a human answer from the corpus at all).

### 🔷 Follow-up 6: "You've diagnosed the system. Now fix it: you get one week and must halve end-to-end error. Spend it."

**Why they ask:** diagnosis without a treatment plan is incomplete science; also tests cost-awareness.

**Model answer:** "The taxonomy is the budget allocator. My distribution was `[FILL]`; generation-side dominated, so the week goes there: first, grounding-hardened prompting — require quoted evidence spans and permit abstention ('not in context'), which converts silent hallucinations into visible abstentions; second, a claim-support check at generation time (cheap NLI or overlap filter) gating unsupported sentences. I would *not* buy a reranker first — it attacks the smaller bucket. Measured checkpoints daily against the same eval set, holding the judge fixed so the numbers stay comparable — and a fresh held-out slice at the end because a week of iterating on one set overfits it."

**The chain:** "abstention trades coverage for correctness — how do you report that?" (risk–coverage curve; ties directly into your calibration project — a deliberate bridge to Lesson 2) → "your fix changed the answer style; does your judge still work?" → "what do you monitor in production?" ⚠️ JD-DEPENDENT.

---

## 🔷 Hands-On Lab: Rebuild the Decision Tree From Memory

30 minutes.

1. On paper, write your failure classifier as pseudocode — every check, every threshold, in order. Then open your actual instrumentation code and diff. Every mismatch is a floor-hit; log it.
2. Fill the Session 2 RAG rows of the Metric Vault in [PROGRESS.md](../../PROGRESS.md): failure distribution, recall@k curve points, oracle result, judge audit, closed-book baseline. Empty slots → `QUALITATIVE-ONLY`.
3. Run the [RAG Failure Triage Lab](01_rag_failure_triage_lab.ipynb) to see the same machinery on a corpus where *you* control ground truth — then write one sentence on what your real pipeline's instrumentation would misclassify that the lab's doesn't.
4. Memorize one real example per failure bucket from your own logs.

## 🟢 Concept Check

Closed-book (no retrieval), your generator already answers 40% of eval questions correctly. What does this most directly threaten?

* [ ] The FAISS index configuration
* [x] The interpretation of failure buckets — parametric knowledge means "retrieval-miss" cases can still be answered and "grounded" answers may come from memory, so attribution needs the closed-book baseline subtracted
* [ ] The chunk-size sweep
* [ ] Nothing — closed-book performance is irrelevant to a RAG evaluation

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** With leakage, correctness stops implying retrieval worked and wrongness stops implying it failed — both taxonomy denominators shift. The closed-book baseline is the control condition that anchors every attribution claim; without it, "generation vs. retrieval" is confounded by "memory vs. pipeline." This is also an honest-limitation answer that *raises* your score when volunteered.
</details>

---

## 🟢 Summary & Resources

- The taxonomy is a decision tree of instrumented checks with known boundary cases — defend the rules, volunteer the overlaps.
- The conclusion rests on a dissociation (recall up, error flat) plus, ideally, an oracle-context experiment; know your falsifiers by name.
- Chunk size works through three mechanisms — dilution, fragmentation, position (Lost in the Middle) — and interacts with k via the context budget.
- Your judge has an error rate with a *direction* per bucket; audited or not, you must know which way each number leans.
- The taxonomy is also the treatment plan: spend fixes where the buckets are, not where the tooling is shiniest.

**References:** Lewis et al. 2020 (RAG, arXiv:2005.11401) · Liu et al. 2023 (Lost in the Middle, arXiv:2307.03172) · Xiao et al. 2023 (BGE/C-Pack, arXiv:2309.07597) · Johnson et al. 2017 (FAISS, arXiv:1702.08734)

**Next:** [Lesson 2 — Defending Calibration Under Distribution Shift](02_defending_calibration_under_shift.md)
