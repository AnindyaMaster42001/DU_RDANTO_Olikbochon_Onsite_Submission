# Plan v3: from LB 0.803 (rank 8) to the top (2026-07-03 → Phase-1 close 2026-07-19)

> **v3 update (P0+P1 done):** The fallback-lane hypothesis is **falsified** — injecting the
> anti-skepticism judges (j32fs/j32sv) into ret32's UNSURE lane yields ~0 net rescues through the
> stitch (best composite 0.8337 = noise), because the meta-model already carries j32sv and extracts
> what those judges know. Both workers independently conclude: **the 16 remaining rows are a corpus
> problem, not a prompt problem.** ⇒ **P3 (native fallback kernel) is CUT; P2 (corpus) is now the
> #1 GPU priority.** One free CPU lever fell out: dropping the two dead signals nli+crosslingual
> lifts OOF 0.8327 → **0.8394** (~1σ, variance-reducing — a hedge-final candidate, not a headline).

## Where we are (post-ret32, post-P0/P1)

- **LB 0.803, rank 8.** Submitted stitch = base8 + j32sv + ret32; honest 5-seed OOF **0.8327 ± 0.008**
  (reproduced exactly this session), no-ctx **0.7775**. OOF→LB transfer held exactly (−0.03) —
  the harness is predictive; trust it.
- `signal_ret32` (32B judges answers vs reranked bnwiki evidence, UNSURE-safe three-way) is the
  strongest no-ctx signal ever (0.7530 standalone) and rescued **7 of the 23** unanimously-missed
  faithful rows. **16 hopeless rows remain, all faithful** (indices in `results/HOPELESS_AUTOPSY.md`).
- **Autopsy of the 16 (P1):** realistic rescue ceiling is **15/16** — row 7 is genuine label noise
  ("competent"→"অযোগ্য"/unqualified is the opposite; judges are *correctly* flagging it). Breakdown:
  **5 idiom/proverb** (1,155,199,284,282 — need a wiktionary/idiom dictionary; defeat *every*
  parametric judge, all ≈0), **5 literature/geography/history facts** (9,122,141,260,261 — bnwiki
  retrieval depth/freshness), **4 parametric** (55,150,204,207 — read correctly by j32fs/j32sv but
  the meta-model already gets these, so not free lift), **1 near-unfixable** (171, opinion MCQ).
- **Corpus status — being verified, NOT assumed.** Earlier draft called the corpus "stale 2023-11";
  that conflated two pipelines. The **Kaggle/Phase-2** path (`kaggle_wiki_retrieve.py`) pulls the HF
  `wikimedia/wikipedia` snapshot (team notes flag it 2023-11). The **VPS** path (`prep_wiki.py`
  reads `~/bnwiki_raw/extracted/`; RESULTS.md says "dump 20260701") points to a *fresh* live dump.
  **A read-only VPS recon is confirming which dump actually sits on the box and whether passages +
  embeddings are prebuilt.** Fill this in before committing P2 effort.
- Phase-2 offline package exists but its **verification run isn't recorded yet** (README promises it;
  last commit was a vLLM-wheels fix — it's been flaky).
- Signal count is saturating: adding j32fs/j32lp on top of the current stitch *hurts* (0.827/0.820);
  and the fallback composite doesn't help either. Only genuinely new *knowledge* (corpus) moves it.

## Guiding principle (unchanged, thresholds updated)

299 rows → overall-OOF deltas under ~0.02 are noise. Every experiment is judged on two axes:
5-seed OOF via `experiments_meta.py`, **and net rescues on the 16 remaining hopeless rows**
(rescued minus new regressions on the other no-ctx rows). New submit gate: **OOF ≥ 0.845** or
**net rescue ≥ 5/16 with OOF not worse**, then burn one LB submission to confirm.

---

## P0 — CPU tiered composite: ret32-with-fallback ✅ DONE — hypothesis FALSIFIED

Tested every fallback variant (fs/sv/mean/max) × τ sweep, replacing/keeping j32sv, scored on the
honest 5-seed harness + net-rescues on the 16. Result: **~0 net rescues, best 0.8337 (noise).** The
meta-model already carries j32sv, so a targeted UNSURE-lane patch adds nothing the stack didn't have.
**Conclusion: the 16 are a corpus problem.** Scripts in `scratchpad/p0_composite.py`.
Side finding: **pruning nli+crosslingual → OOF 0.8394** (free, GPU-free, variance-reducing);
submission at `scratchpad/submission_prune_nli_xling.csv` — hold as a hedge-final candidate, do not
burn an LB check on a +0.007 move alone.

## P1 — Autopsy the 16 remaining hopeless rows ✅ DONE

Full taxonomy in `results/HOPELESS_AUTOPSY.md`. Ceiling **15/16** (drop row 7, label noise).
Rows needing `retrieved_evidence.json` to split retrieval-miss vs grounding-miss:
**[9, 122, 141, 260, 261, 282]** — pull from VPS/Kaggle when GPU access is granted.

## P2 — Corpus upgrades — ⭐ #1 PRIORITY (now evidence-backed)

**VPS recon settled the corpus facts (read-only):** the box holds a **fresh 2026 bnwiki**
(`bnwiki-latest.xml.bz2`, pulled 2026-07-02, 438,788 articles → 335,628 passages, embeddings built,
`retrieved_evidence.json` present) — NOT stale. Passage pollution is negligible (242/335,628 = 0.07%).
So the "swap the stale dump" idea is moot on the VPS side; freshness is only a *Kaggle/Phase-2* gap
(that path pulls the HF 2023-11 snapshot — worth aligning Phase-2 to the fresh dump, but it's not
where the misses come from). **Caveat:** `ret32` (our LB-0.803 signal) was grounded on the *Kaggle
stale+reranked* evidence, while the VPS fresh evidence only fed the weaker 14B `signal_retrieval`.

**The real lever is the WRONG BOOK, not a stale one.** Pulling the fresh `retrieved_evidence.json`
(read-only) and inspecting the 16 rows: for every idiom/proverb row bnwiki retrieves pure noise
(grammar/physics/fruit articles) — Wikipedia simply does not contain idiom meanings. And this is a
**big slice of the test set, not a corner case**: of the 1155 no-context test rows, **162 (14%) are
wiktionary-type** word/idiom/meaning questions (75 `ভাবার্থ` idioms + 80 word-meaning + synonyms/
antonyms/proverbs). Grammar-type (no corpus helps) is only 19 (1.6%).

### P2a — bn.wiktionary corpus ✅ DONE → OOF 0.8327 → 0.8460 (clears gate)

**Result:** built end-to-end on the VPS 32B (~1 min grounding). The signal is high-precision on
**idioms (ভাবার্থ) 12/13 = 0.92** (all 75 idiom test rows grounded) but noisy on word-meaning (0.50,
label noise + polysemy). Applied as a **targeted idiom-only override** (not a meta-model feature —
too sparse to weight): baseline + idiom-override = **0.8460 ± 0.008**, rescues ~4/16 hopeless.
`results/submission_wikt_override.csv` (flips 26/75 idiom test preds) — awaits LB confirmation.
Repro: `build_wikt.py` → `wikt_retrieve.py` → `wikt_ground32.py`. Original design notes below:


1. Fetch `bnwiktionary-latest-pages-articles.xml.bz2` (~29 MB). **Custom parser** (wikiextractor is
   built for article prose, not Wiktionary definition/POS structure) → passages keyed by headword
   with its Bengali definition/`অর্থ`. Validate FIRST on the 5 sample idiom rows
   (1,155,199,282,284) that the correct meanings are actually present — cheap CPU check before any
   GPU spend.
2. Embed with bge-m3; add to the retrieval union (source-tagged `wikt`). For a `ভাবার্থ/অর্থ` query,
   retrieve the headword entry, then 32B judges whether the given meaning matches the dictionary
   definition (same ret32 grounding pattern, right corpus). Export `signal_ret32wikt.json`.
3. **Where it runs — settled 2026-07-04.** Corpus parse is done locally (7 s → `wikt_passages.jsonl`,
   73,444 entries: 15,182 idioms + 58,262 words; idiom sample coverage 9/13 by exact lookup, a floor).
   **DEV grounding runs on the VPS 32B** (tested viable: ~57 tok/s steady-state, ~15 min for a full
   ~1300-query run; the earlier "days" reading was a one-time marlin compile artifact on the first
   inference). Run at **util ≈ 0.21, max_model_len ≈ 1536** (short wiktionary prompts) to leave a
   ~2.5 GB card-wide buffer for the other bare (no-auto-restart) jobs; risk window ~15 min, owner
   accepts it. **Phase-2 FINAL still grounds on Kaggle** (competition requirement) — same GPTQ weights,
   so signals match. Keep both VPS models (32B now used for dev; 14B-AWQ also runs fine here).

### P2b — retrieval quality on the fact rows (secondary, ~1–2 sample rows)
Freshness isn't the bottleneck for the literature/geography rows; retrieval *precision* is (row 9
matched name-homonyms: modern novelist Humayun, not the Mughal emperor). Re-grounding `ret32` on
fresh+reranked evidence may rescue ~1–2 (row 122 has the right article retrieved). Low priority vs P2a.
The stripped-MCQ rows (260 jhum, 261 newspaper, 171) are near-unfixable by retrieval — don't chase.

- Update `Phase_2/snapshots/wiki_index_v2.py` + the shipped corpus to include wiktionary if P2a wins.

## P3 — Native fallback lane in the grounding kernel — ❌ CUT (P0 falsified the premise)

The fallback lane adds nothing through the stitch, so a native-prompt version would too. Redirect
these days into **P2 depth**: larger top-k into the reranker, full-article chunking (beyond the
first 6 chunks), and the wiktionary corpus. If a *new knowledge source* still can't reach the
literature/geography rows after P2, only then revisit prompt-level ideas.

## P4 — C1 probe set (days 6–10, parallel; still our biggest blind spot)

Unchanged from plan v1 and still not built. ~120–150 Bangladesh-specific rows: true facts from
bnwiki (districts, rivers, liberation war, literature) as faithful rows in the test set's style;
corrupt half (dates/names/numbers) into hallucinated counterparts. **Second validation axis and
finals tiebreaker only — never a tuning target.** With rank 8 and everyone fighting over the same
public split, private-band visibility is worth more than another micro-signal.

## P5 — Phase-2 verification run (days 3–7, parallel; it's a submission requirement)

The package (`Phase_2/phase2_pipeline.ipynb`, 7 portable signals, OOF 0.8311, ~37 GB) exists but
the promised offline verification isn't recorded in RESULTS.md. Run it end-to-end on Kaggle
(T4×2, internet OFF) now, while there's slack to fix breakage: confirm completion time, and that
it reproduces the submitted predictions. Record both in `results/RESULTS.md`. Re-verify once if
P2/P3 change the shipped corpus or signals — freeze the package by **Jul 17**.

## Explicitly NOT doing

- Another generic parametric judge (agreement 0.74–0.86; ret32's win came from *evidence*, not scale).
- Stitching j32fs/j32lp as standalone signals (measured: hurts). They only re-enter via the
  fallback lane (P0/P3).
- Meta-model tinkering, threshold sweeps, chasing ≤0.01 LB deltas — all noise at 299 rows.

---

## Submission strategy

- **Final #1 = the 0.803 submission** (base8 + j32sv + ret32) unless something passes the gate
  (OOF ≥ 0.845 or net rescue ≥ 5/16 with OOF held) *and* confirms on the LB.
- **Final #2 = hedge:** current best hedge is the **nli+crosslingual-pruned** stitch (OOF 0.8394,
  `scratchpad/submission_prune_nli_xling.csv`); replace with a gated corpus candidate as they land,
  using the C1 probe set to arbitrate ties.
- Modeling freeze **Jul 17**; Jul 17–19 is Phase-2 re-verification and finals selection only.

## Timeline

| Days (Jul) | Work |
|---|---|
| 3 | ✅ P0 composite (falsified) + P1 autopsy (done); read-only VPS corpus recon |
| 3–7 | P5 in parallel: Phase-2 offline verification run |
| 4–8 | **P2 (now #1): wiktionary index + fresh-dump retrieval → ret32v2** (needs VPS/GPU) |
| 6–10 | P4 in parallel: C1 probe set |
| 10–14 | Stitch/gate/submit best corpus candidates; iterate on whichever rescues rows |
| 15–17 | Finals selection, modeling freeze, Phase-2 package freeze |
| 17–19 | Buffer: re-verification, submit finals |
