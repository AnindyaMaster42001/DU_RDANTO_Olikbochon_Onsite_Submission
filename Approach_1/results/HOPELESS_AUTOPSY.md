# P1 autopsy — the 16 remaining hopeless rows

All 16 are no-context, labeled faithful (label=1), and missed by every base8 signal **and**
ret32. Goal: decide where GPU budget can actually move the needle. Sample indices (0-based, aligned
with the `samples` arrays and `Approach_0/dataset samples.json`):

`[1, 7, 9, 55, 122, 141, 150, 155, 171, 199, 204, 207, 260, 261, 282, 284]`

Signal columns below: `ret32 / j32fs / j32sv / j32lp` (P(faithful); >0.5 = decisively correct).

---

## Row-by-row

### IDX 1 — `IDIOM/PROVERB`
- **Q:** "ধান্ধা" এর ভাবার্থ কী? — *What is the figurative meaning of "dhandha"?*
- **A:** কোন অসৎ উদ্দেশ্য — *"some dishonest purpose/scheme"*
- sig: 0.496 / 0.000 / 0.000 / 0.000 — every judge says hallucinated.
- **Correct?** Yes — "ধান্ধা" colloquially = a shady hustle/scheme; the gloss is defensible.
- **Fix:** P2 bn.wiktionary index. Wikipedia has no entry for a slang word's connotation.

### IDX 7 — `LABEL-NOISE` (drop)
- **Q:** ইংরেজি ভাষায় "competent" শব্দের অর্থ কী? — *Meaning of English "competent"?*
- **A:** অযোগ্য — *"unqualified / incompetent"*
- sig: 0.013 / 0.000 / 0.000 / 0.000 — unanimous hallucinated, **and the judges are right**.
- **Correct?** **NO.** "competent" = যোগ্য/দক্ষ (qualified). The answer is the exact opposite. This row
  is mislabeled faithful. **Chasing it would require teaching the model a falsehood — exclude it.**
- **Fix:** none — drop from rescue targets.

### IDX 9 — `LITERATURE-FACT` (history)
- **Q:** কোন মুঘল সম্রাট বাংলার নাম দেন 'জান্নাতাবাদ'? — *Which Mughal emperor named Bengal 'Jannatabad'?*
- **A:** হুমায়ুন — *Humayun*
- sig: 0.497 / 0.297 / 0.000 / 0.029
- **Correct?** Yes — Humayun renamed Gaur "Jannatabad" after taking it in 1538. Real, verifiable fact.
- **Fix:** P2 bnwiki retrieval (Gaur / Humayun / Bengal-history articles). Needs evidence-file check.

### IDX 55 — `GRAMMAR/PHONETICS`
- **Q:** উচ্চারণের রীতি অনুযায়ী নিচের কোনটি উচ্চমধ্য-সম্মুখ স্বরধ্বনি? — *Which is a close-mid front vowel?* (MCQ, options stripped)
- **A:** এ — *"e"*
- sig: 0.494 / **0.508** / 0.000 / **0.909**
- **Correct?** Yes — এ /e/ is the close-mid (উচ্চমধ্য) front vowel in Bengali phonology.
- **Fix:** P3 parametric fallback — j32fs/j32lp already get it; no corpus needed. Note MCQ options
  were stripped from the prompt, turning it into open recall (harder for retrieval).

### IDX 122 — `LITERATURE-FACT`
- **Q:** দীনবন্ধু মিত্রের প্রহসন কোনটি? — *Which is Dinabandhu Mitra's farce (prohoshon)?*
- **A:** বিয়ে পাগলা বুড়ো — *"Biye Pagla Buro"*
- sig: 0.155 / 0.356 / 0.000 / 0.523
- **Correct?** Yes — "বিয়ে পাগলা বুড়ো" is a well-known farce by Dinabandhu Mitra.
- **Fix:** P2 bnwiki (Dinabandhu Mitra article lists his works). Needs evidence-file check.

### IDX 141 — `LITERATURE-FACT` (yes/no format)
- **Q:** ভবিষ্যতের বাঙালি গ্রন্থটির রচয়িতা এস ওয়াজেদ আলী? — *Is 'Bhabishyater Bangali' authored by S. Wajed Ali?*
- **A:** হ্যাঁ — *"Yes"*
- sig: 0.343 / 0.416 / **1.000** / 0.120
- **Correct?** Yes — "ভবিষ্যতের বাঙালি" is indeed by এস ওয়াজেদ আলী. The yes/no framing (not a malformed
  row) trips judges that expect a factual span. j32sv handles it.
- **Fix:** P2 bnwiki + P3 fallback. The yes/no verification format is a prompt-design issue worth
  handling explicitly in the grounding kernel.

### IDX 150 — `DEFINITION` (general knowledge, parametric)
- **Q:** যাদের বুদ্ধ্যঙ্ক ১৪০ বা তার ঊর্ধ্বে তাদের বলা হয়— — *Those with IQ ≥140 are called—*
- **A:** অতিশয় প্রতিভাশালী — *"extremely gifted / genius"*
- sig: 0.035 / **0.700** / **1.000** / **0.834**
- **Correct?** Yes — 140+ is the "genius/near-genius" band in standard IQ classification. Not
  Bangladesh-specific; a general fact.
- **Fix:** P3 parametric fallback — three judge variants already get it; ret32 uniquely wrong
  (evidence pass overrode a correct parametric read → a caution for the P3 merge).

### IDX 155 — `IDIOM/PROVERB`
- **Q:** "উজানের কই" এর ভাবার্থ কী? — *Figurative meaning of "ujaner koi"?*
- **A:** সহজলভ্য বস্তু — *"an easily-obtained thing"*
- sig: 0.500 / 0.001 / 0.000 / 0.002 — ret32 pure UNSURE (0.5), every judge confidently wrong.
- **Correct?** Yes — the idiom means something easily available.
- **Fix:** P2 bn.wiktionary. Parametric judges are hopeless here (all ≈0), so P3 fallback will NOT
  help — this row needs the right corpus or nothing.

### IDX 171 — `REASONING / CURRENT-FACT` (weakest target)
- **Q:** বাংলাদেশের ব্লু-ইকোনমির চ্যালেঞ্জ নয় কোনটি? — *Which is NOT a challenge of Bangladesh's blue economy?* (negation MCQ, options stripped)
- **A:** ঘন ঘন বন্যা — *"frequent floods"*
- sig: 0.349 / 0.119 / 0.000 / 0.029
- **Correct?** Debatable — a "which is NOT" MCQ whose distractor set is missing; floods are arguably
  a challenge too. Underspecified once options are stripped.
- **Fix:** Low-value. Negation + opinion + missing options ⇒ neither corpus nor fallback reliably
  helps. Treat as near-unfixable.

### IDX 199 — `IDIOM/PROVERB`
- **Q:** "কথা ফেলা" এর ভাবার্থ কী? — *Figurative meaning of "kotha fela"?*
- **A:** কথা অগ্রাহ্য করা, অবহেলা করা — *"to disregard/ignore someone's words"*
- sig: 0.492 / **0.991** / 0.000 / **0.835**
- **Correct?** Yes.
- **Fix:** P2 wiktionary is ideal, but j32fs *already* nails it (0.991) → also rescuable by the P3
  fallback lane. Double-covered.

### IDX 204 — `REASONING/ANALOGY`
- **Q:** ব্রেক : রিপেয়ার :: উন্ড : ? — *Break : Repair :: Wound : ?*
- **A:** হিল — *"Heal"*
- sig: 0.492 / **0.511** / 0.000 / 0.116
- **Correct?** Yes — you repair a break, you heal a wound.
- **Fix:** P3 parametric fallback only. Retrieval cannot help an analogy. j32fs barely gets it →
  the P3 few-shot/anti-skepticism framing should stabilize it.

### IDX 207 — `GRAMMAR`
- **Q:** "মরি মরি! কী সুন্দর প্রভাতের রূপ!" — এখানে কোন অব্যয়? — *Which indeclinable (obboy) is used here?*
- **A:** অনন্বয়ী অব্যয় — *"interjectional indeclinable"*
- sig: 0.431 / **0.637** / 0.000 / 0.000
- **Correct?** Yes — "মরি মরি" is an exclamation ⇒ অনন্বয়ী অব্যয়.
- **Fix:** P3 parametric fallback (grammar knowledge); j32fs gets it. bnwiki Bengali-grammar article
  a weak secondary.

### IDX 260 — `GEOGRAPHY-FACT`
- **Q:** জুম চাষ হয়— ক) খাগড়াছড়ি খ) টাঙাইল গ) উত্তরা ঘ) বগুড়া — *Where does jhum cultivation happen?* (options intact)
- **A:** খাগড়াছড়িতে — *"in Khagrachari"*
- sig: 0.459 / **0.605** / 0.000 / 0.487
- **Correct?** Yes — jhum (shifting) cultivation is in the Chittagong Hill Tracts, incl. Khagrachari.
- **Fix:** P2 bnwiki (jhum / Chittagong Hill Tracts). Options are present here, so retrieval should
  work if the corpus has it. Needs evidence-file check.

### IDX 261 — `CURRENT-FACT` (underspecified)
- **Q:** ঢাকা থেকে প্রকাশিত হয় কোন পত্রিকাটি? — *Which newspaper is published from Dhaka?* (MCQ, options stripped)
- **A:** ক্রান্তি — *"Kranti"*
- sig: 0.003 / **0.505** / 0.000 / **0.792**
- **Correct?** Plausibly — but without the option set, "which newspaper from Dhaka" has thousands of
  valid answers; the intended answer is only pinnable with the original distractors.
- **Fix:** P2 bnwiki weak; j32fs/j32lp already lean correct so P3 fallback is the realistic route.
  Low confidence due to stripped options. Needs evidence-file check.

### IDX 282 — `LITERATURE-FACT` (Charyapada / Old Bengali)
- **Q:** 'রুখের তেন্তুলি কুমীরে খাই'–এর অর্থ কী? — *Meaning of this line?*
- **A:** গাছের তেঁতুল কুমিরে খায় — *"the crocodile eats the tree's tamarind"* (modern-Bengali gloss)
- sig: 0.390 / 0.000 / 0.000 / 0.000 — every judge confidently wrong.
- **Correct?** Yes — it's a Charyapada line; the answer is the standard literal translation.
- **Fix:** P2 bnwiki (Charyapada article carries these lines + glosses) OR a literature reference.
  Parametric judges are hopeless (all ≈0) → corpus is the only route. Needs evidence-file check.

### IDX 284 — `IDIOM/PROVERB`
- **Q:** "লাভের গাঁতি" এর ভাবার্থ কী? — *Figurative meaning of "laabher gãti"?*
- **A:** লাভের বিষয় — *"a matter of profit"*
- sig: 0.500 / 0.022 / 0.000 / 0.001 — ret32 pure UNSURE, judges confidently wrong.
- **Correct?** Yes (defensible idiom gloss).
- **Fix:** P2 bn.wiktionary. Parametric hopeless → corpus or nothing.

---

## Summary

### Category tally (primary bucket per row)

| Bucket | Rows | Count |
|---|---|---|
| (a) bn.wiktionary / idiom dictionary | 1, 155, 199, 284, 282* | 5 |
| (b) fresh/deeper bnwiki retrieval (literature/geography/history) | 9, 122, 141, 260, 261 | 5 |
| (d) P3 anti-skepticism parametric fallback (no corpus) | 55, 150, 204, 207 | 4 |
| (e) near-unfixable (opinion/negation, stripped options) | 171 | 1 |
| (f) label noise — drop | 7 | 1 |

`*` 282 (Charyapada line) straddles (a)/(b): it's a phrase-meaning like the idioms but the source is
the bnwiki Charyapada article, so it's reachable by bucket (b) retrieval too.

(c) "deeper retrieval/reranking" is not a separate row set — it's the *mechanism* that rescues bucket
(b) if the fact is in-corpus but currently below top-5. The evidence-file check decides (b) vs (c).

Overlap worth noting: several bucket-(d) rows are **double-covered** — j32fs/j32sv/j32lp already read
them correctly (55, 150, 199, 204, 207, 260, 261), which is exactly the P0 fallback-lane thesis:
where ret32 sits at ≈0.5, defer to the anti-skepticism judge. The idiom rows (1, 155, 284) and the
Charyapada row (282) are the ones where **every** parametric judge fails (all ≈0), so P0/P3 cannot
touch them — only a wiktionary/literature corpus can.

### Realistic rescue ceiling

- Hard ceiling: **15 of 16** (drop only row 7, the mislabeled one).
- Realistic near-term: **~10–12**.
  - **~7 via the P0/P3 fallback lane** with zero new GPU (the double-covered set above) — the
    cheapest points on the board and the reason to run P0 first.
  - **+3–5 via corpus work**: idioms (1, 155, 284) need bn.wiktionary; 282 needs the Charyapada
    article; the literature/geography facts (9, 122, 141, 260) need bnwiki retrieval that actually
    surfaces the right passage.
  - Written off: 7 (label noise), 171 (underspecified negation MCQ), 261 (stripped-options ambiguity,
    keep only as a fallback-lane freebie).

### Rows that need `retrieved_evidence.json` (retrieval-miss vs grounding-miss split)

Currently only on Kaggle/VPS, not local. Pull it for these **6 bnwiki-plausible facts** to decide
whether the fix is "index the right corpus / rerank deeper" (retrieval-miss) or "the passage was
there and the 32B still judged wrong" (grounding-miss → P3 prompt work):

`[9, 122, 141, 260, 261, 282]`

The idiom/grammar/reasoning rows (1, 55, 150, 155, 199, 204, 207, 284) do not need the evidence file —
bnwiki was never going to contain them; they route to wiktionary (idioms) or the parametric fallback
(grammar/reasoning), independent of what retrieval returned.
