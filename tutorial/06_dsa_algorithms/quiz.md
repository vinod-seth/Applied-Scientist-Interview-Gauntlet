# Session 6 — Chapter Quiz Bank

| | |
|---|---|
| **Prerequisites** | Lessons 1–5 |
| **Time** | ~35 min |
| **Rules** | Closed notes. Say the answer **out loud** before selecting — this session is about what you can produce audibly, and recognising the right option is not the same as being able to say it under a clock. |

12 quiz questions plus 2 reflection prompts. Nothing here is scored; the bank exists to find the protocol habits you *think* you have before an interviewer finds them.

---

## 📝 Chapter Quiz

**Q1.** Your approach statement should end with:

* [ ] The first line of code
* [x] A question — "does that sound right before I code it?" — because coding an approach the interviewer has not agreed to is the most expensive mistake available to a fluent coder
* [ ] A restatement of the problem
* [ ] The list of test cases

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The pause costs four seconds and protects twenty minutes. If the interviewer was steering you toward a different structure, you now find out before writing rather than after — and their steering attempt, ignored, would itself have become negative evidence. Option 4 is not wrong to *include*, but it belongs at the end of the approach, before the question, not in place of it.
</details>

**Q2.** The right number of clarifying questions at the start of a coding round is:

* [ ] As many as possible, to demonstrate thoroughness
* [x] Two or three, chosen because their answers would change your design — covering input domain and size, output contract, and edge semantics
* [ ] None; the constraints are always in the problem statement
* [ ] One, to save time for coding

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The trap runs in both directions: zero questions reads as assuming rather than checking — the exact reflex that solving hundreds of practice problems installs, since practice sites *print* the constraints — while twenty reads as stalling. The selection rule is what makes it calibrated: ask only what would change the design. Precede them with a one-sentence restatement, which catches a misread before it costs twenty minutes.
</details>

**Q3.** A problem states $n \le 5{,}000$. The most likely intended complexity is:

* [ ] $O(2^n)$
* [x] $O(n^2)$ — roughly $2.5 \times 10^7$ operations, comfortably inside a time limit, and typical of two-dimensional DP or all-pairs comparison
* [ ] $O(\log n)$
* [ ] $O(n!)$

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Bounds are design information, and reading them first is a beat you can perform out loud in ten seconds. The ladder worth memorising: $n \le 20$ admits exponential; $n \le 5{,}000$ admits $O(n^2)$; $n \le 10^6$ admits $O(n \log n)$; $n \le 10^{18}$ admits only $O(\log n)$ or $O(\sqrt n)$. Option 1 would be $2^{5000}$ — worth being able to reject instantly.
</details>

**Q4.** Maximum of every fixed-size window of length $w$ over an array. The right structure is:

* [ ] A max-heap of the window's elements
* [x] A monotonic deque of indices in decreasing value — each index is pushed and popped at most once, giving $O(n)$; a heap gives $O(n \log w)$ because it cannot cheaply evict the element that just left the window
* [ ] Re-scanning each window, which is $O(nw)$ but simple
* [ ] Prefix sums

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The heap's specific weakness is the reason to name: removing an *arbitrary* element (the one that just expired) is not a heap operation, so you end up lazily deleting stale entries — which works, and is strictly more complicated than the deque. Option 4 fails because maximum, unlike sum, has no inverse operation, so nothing can be subtracted off the back.
</details>

**Q5.** In Python, `grid = [[0] * n] * m` then `grid[0][0] = 1` results in:

* [ ] A single cell set to 1, as intended
* [x] Column 0 set to 1 in **every** row — the outer multiplication copies the *reference* to one list, so all $m$ rows are the same object
* [ ] A `TypeError`
* [ ] Undefined behaviour that varies by version

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** This bug deserves special fear: it produces a *wrong answer* rather than an error, in code that looks idiomatic, on grid problems — which are common. The fix is a comprehension: `[[0] * n for _ in range(m)]`. Inner `[0] * n` is safe because integers are immutable; it is the outer repetition of a mutable list that aliases.
</details>

**Q6.** Building a heap from $n$ unsorted elements costs:

* [ ] $O(n \log n)$ — one insertion per element
* [x] $O(n)$ with the bottom-up heapify, because most nodes sit near the leaves and sift down only a short distance; the sum $\sum_h h \cdot n/2^h$ converges
* [ ] $O(n^2)$
* [ ] $O(\log n)$

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Option 1 is the bound for $n$ successive insertions, which is a different algorithm and a real distinction — heapsort's build phase is linear, and only the extraction phase costs $O(n \log n)$. Volunteering the converging-sum argument is a cheap way to show the bound was derived rather than memorised.
</details>

**Q7.** To keep the $k$ **largest** elements of a stream you use:

* [ ] A max-heap of size $k$
* [x] A **min**-heap of size $k$ — its root is the smallest of your current best $k$, which is exactly the admission threshold for an arriving element
* [ ] A sorted array, rebuilt per element
* [ ] Two heaps

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The inversion is the single most common error in this family, and it is worth over-learning: *k largest → min-heap; k smallest → max-heap*. A max-heap of size $k$ gives you instant access to the element you least want to evict. Option 4 describes the running-median structure, which is a different problem — two heaps balanced around the middle.
</details>

**Q8.** A binary search using `while lo < hi` loops forever. The most likely cause is:

* [ ] The array is not sorted
* [x] The midpoint rounds down while the update on the low side is `lo = mid`, so `lo` stops moving — either use `lo = mid + 1`, or bias the midpoint upward with `lo + (hi - lo + 1) / 2`
* [ ] The range is too large
* [ ] `hi` was initialised to `n` rather than `n - 1`

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** This is the boundary-template hazard, and it is independent of the *other* binary-search hazard — `(lo + hi) / 2` overflowing a fixed-width integer, documented by Bloch in 2006 as a bug that survived nine years in the Java standard library. Option 4 is a real off-by-one source in some templates but does not cause a non-terminating loop. Fix one template and never mix them.
</details>

**Q9.** The interviewer asks you to prove your greedy interval-scheduling solution is optimal. The right shape of argument is:

* [ ] A loop invariant
* [x] An exchange argument — show any optimal solution can have its first choice swapped for the greedy one without getting worse, then induct
* [ ] Induction on the input size alone
* [ ] A monotone-predicate argument

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Matching the shape to the algorithm family is the whole skill: **loop invariant** for scans (windows, two pointers, monotonic stacks), **exchange argument** for greedy, **monotone predicate** for binary search over answers. For interval scheduling specifically: the earliest-ending compatible interval conflicts with no more future intervals than any alternative, so swapping it in never reduces the count.
</details>

**Q10.** Top-100 most frequent items from a huge log, where the *distinct* keys no longer fit in memory. The exact solution is:

* [ ] Impossible — you must approximate
* [x] Shard by hash of the key across partitions, count each independently, take a local top-100 per shard and merge — correct because every occurrence of a key hashes to the same shard, so no count is ever split
* [ ] Sample the log and scale the counts up
* [ ] Sort the whole log on disk and scan for runs

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** Naming *why* the sharding is exact — the hash guarantees all occurrences of a key land together — is the part being scored; without it the design sounds like a guess. Option 4 is a legitimate external-sort approach and worth mentioning, but it is $O(N \log N)$ on the full log rather than $O(N)$ on shards. A count–min sketch is the right answer when memory is genuinely fixed, and it should be offered as an *approximation* with its one-sided error, not as an equal alternative.
</details>

**Q11.** MinHash is used for near-duplicate detection because:

* [ ] It compresses documents losslessly
* [x] The probability that two sets share the same minimum hash value **equals** their Jaccard similarity, so a fixed-length signature estimates similarity with standard error about $1/\sqrt{m}$ — and LSH banding then avoids comparing all pairs
* [ ] It is faster than computing a hash of the whole document
* [ ] It detects semantic similarity between paraphrases

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** The exact-equality property is what makes it a *measurement* rather than a heuristic (Broder, 1997). Option 4 is the distinction to volunteer: MinHash measures **lexical** overlap over shingles, not meaning — paraphrases with no shared shingles score near zero. For semantic dedup you would need embeddings and an approximate-nearest-neighbour index, at much higher cost. The scale argument comes first though: all-pairs on $10^8$ documents is about $5 \times 10^{15}$ comparisons.
</details>

**Q12.** Edges arrive one at a time and you must report the component count after each. Union–find is right, and re-running BFS is wrong, because:

* [ ] BFS gives incorrect component counts
* [x] BFS is $O(V+E)$ per query, so re-running it per edge is $O(E(V+E))$; union–find merges incrementally at effectively constant amortised cost — though it can never *split* a set, which is its real limitation
* [ ] Union–find also returns shortest paths
* [ ] BFS cannot handle undirected graphs

<details>
<summary>🔑 Click to Reveal Answer & Explanation</summary>

**Correct: option 2.** BFS is correct but wrongly *sized* for the query pattern, and that distinction — correct versus appropriately sized — is what the question tests. With path compression and union by rank the bound is $O(\alpha(n))$ amortised (Tarjan, 1975), at most 4 for any input that exists. Volunteering the no-deletion limitation unprompted is the senior answer: if edges could be removed, union–find is the wrong tool and the problem becomes substantially harder.
</details>

---

## 🪞 Reflection Prompts

Reflection prompt 1 — *Audit your own transcript.* Record yourself solving [Problem 1 from the mock round](05_mock_round.md) out loud on a 20-minute timer. Then listen back with the beat list in front of you and answer honestly: (a) how many seconds elapsed between hearing the problem and touching the keyboard? (b) did your approach statement contain all four parts, and did it end in a question? (c) what is the longest continuous silence in the recording? (d) did you say "done", or "let me test it"?

<details>
<summary>🔑 Evaluation Criteria</summary>

The measurements matter more than the impressions, which is why this prompt asks for four specific numbers. Typical first-attempt results for a fluent competitive coder: under 15 seconds before typing, an approach statement with the complexity but not the why-not-the-obvious, a silence of 90+ seconds during implementation, and the word "done". None of those is a knowledge failure and none of them will improve by reading more.

The gap closes with repetition, not study — this is the highest-return, lowest-effort item in your entire [Gap Map](../../dossier/05_gap_map_and_study_plan.md), which is why Gap #8 is marked closable. Re-record after a week of daily attempts and compare the same four numbers. If the silence figure has not moved, narrate *deliberately trivially* for a few sessions ("standard loop setup, nothing interesting here") until speaking while typing stops competing for attention.
</details>

Reflection prompt 2 — *The 700+ question, answered without becoming "the competitive programmer".* Your résumé claims 700+ solved problems in C++. [Dossier Ch4](../../dossier/04_the_introduction.md) advises keeping that out of your introduction, because naming it early reframes you as a coder rather than a scientist. But in *this* round it is directly relevant and an interviewer may ask what you learned from it. Write the answer: (a) three things you now do differently that are not "know more patterns"; (b) one concrete problem where a wrong assumption cost you, and what you changed afterwards; (c) how you connect the practice to Applied Scientist work without overclaiming.

<details>
<summary>🔑 Evaluation Criteria</summary>

A strong answer is causal, not a list — "I read the constraints before the problem, because the bound eliminates most of the design space before I've thought about the structure" beats "I learned constraints matter." Part (b) must be a real problem from your own practice, with the real wrong assumption; a fabricated one collapses on the first follow-up, and the honest version is more persuasive anyway. Part (c) is the one to be careful with: the defensible connection is *methodological* — designing tests against invariants, choosing structures by their cost at scale, and knowing when an exact algorithm has to become an approximate one. Claiming that competitive practice makes you a better scientist is an overclaim and will be treated as one.

Note the pairing with the introduction advice: the material is not being hidden, it is being *placed*. Volume belongs in the coding round, where it is evidence; it does not belong in the opening 90 seconds, where it is a frame you do not want. Never quote a problem count you have not checked on your own profile.
</details>

---

**Next:** [Mock Round — The 45-Minute Coding Round](05_mock_round.md) if you have not run it, then [Session 7 — Coding: ML From Scratch](../07_ml_from_scratch/README.md).
