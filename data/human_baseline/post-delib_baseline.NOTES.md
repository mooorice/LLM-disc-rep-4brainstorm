# The post-deliberative (six discourse) benchmark

Source: *Genome Editing: Formulating an Australian Community Response*, Occasional
Paper No 12, Centre for Law and Genetics, 2022 (`background/OP12-final-report.pdf`).
Built by `scripts/build_post_delib_baseline.py`, which refuses to write unless the
transcription reproduces the report's own published correlation tables.

Two files:

| File | Source | Content |
|---|---|---|
| `post-delib_discourses_clean.txt` | Appendix C.5.2, pp. 142–145 | Six discourse narratives |
| `post-delib_factor_array.csv` | Table 6, p. 142 | 45 statements × 6 discourses, z-scores |

## Why this is now the primary benchmark

The four mapping-study discourses describe *pre*-deliberative opinion in the
Australian population. The six describe the ways of reasoning that existed *after*
the Australian Citizens' Jury had worked through the issue. They are not a finer
partition of the same four — deliberation clarified and differentiated the space,
so the six are configurations of reasons that only became visible once people had
argued about it.

That is the space a brainstorming tool should make available. The question is not
whether an LLM reproduces the distribution of raw opinion; it is whether it can
surface the range of socially grounded ways of reasoning that substantive human
deliberation brings out. The four-discourse map stays in the repository as a
secondary comparison and as evidence about social grounding.

## Statement 46 is absent, and that is correct

The mapping study used 46 statements; Table 6 has 45. OP12 (p. 124, note to
Table 1) records that **"Statement 46 was dropped following the Mapping Study and
not used in subsequent surveys"**. Item 46 is *"If you can have strong genes, you
will have wellbeing."*

The NLI stance matrices were computed against all 46 statements. Downstream code
aligns on `statement_id`, and item 46 carries a z-score of zero under every
post-deliberative discourse, so it contributes nothing to any post-deliberative
alignment. Nothing needs re-scoring.

## Validation of the factor array

Transcribing a numeric table out of a PDF is the kind of step that fails silently,
and a column swap in particular would invert conclusions while looking entirely
plausible. Three independent checks run before the file is written, all comparing
Pearson correlations computed across the 45 statements against figures the report
publishes separately. All 25 comparisons agreed within 0.08.

1. **Control — Table 3** (p. 134, correlations among the four mapping-study
   discourses), recomputed from the *pre*-deliberative array transcribed in an
   earlier session. This validates the checking method itself on data not
   transcribed here. 6 comparisons.
2. **Table 5** (p. 140, correlations among the six post-deliberative discourses),
   recomputed from the new array. This is the column-order check: permuting two
   discourse columns permutes this matrix, and would not survive 15 comparisons.
3. **Table 4** (p. 140, six post-deliberative against four mapping-study
   discourses), recomputed by correlating the new array against the old one. An
   independent anchor tying the six columns to four whose identity is already
   established. 24 comparisons.

The tolerance of 0.08 exists because Table 6 is rounded to one decimal place while
the report's correlations were computed on unrounded scores.

### A trap that was caught

`pdftotext` extracts the *Mapping Study* half of Table 6 with its last two columns
reversed: item 1 reads `0.6 −0.6 −0.3 1.4` where Table 2 gives
`0.62 −0.61 1.35 −0.31`. The swap is systematic across all 45 rows. It affects only
that half of the table, and the six-discourse half was confirmed correct by the
checks above plus five narrative cross-references (statements 10, 11, 21, 38 and
39, each singled out in the report's own commentary). Nothing from the Mapping
Study half of Table 6 is used — those z-scores are taken from Table 2, which gives
them at two decimal places.

## Narrative cleaning

Same policy as `pre-delib_discourses_clean.CHANGES.md`: **delete or repair, never
paraphrase**. Hard line wraps from the PDF unwrapped, curly quotes normalised to
straight, headers rewritten from `Position X:` to `Discourse X:` so both baselines
parse with the same loader. Position names untouched.

One sentence deleted, from Discourse F:

> Only one Australian Citizens' Jury participant was strongly associated with this
> position.

This is a fact about the sample, not about the way of reasoning, and it injects
sample-size vocabulary into a text used as an embedding target. The equivalent
sentences were removed from the pre-deliberative descriptions for the same reason.
No other edits: no typographic repairs were needed and no substantive corrections
were required.

Note that the deleted sentence records something important — **F rests on a single
jury participant** — which is why it is reproduced here. See the reliability
caveat below.

## Why there are no population weights

There is a distribution: **Figure 10, p. 50, "Distribution of Australian Public
Positions—Six Discourse Map"**, a Venn diagram over the six positions with
percentages in each region (largest single regions: C 17%, no significant loading
16%, D 11%).

It is not used, for three reasons.

1. **The report disclaims it.** Section 7.3 introduces it as a "provisional
   analysis" and the text under the figure opens "Although the results are yet to
   be verified". The report's own authors do not treat it as established.
2. **It is not machine-readable as marginals.** The figure gives percentages for
   overlapping Venn regions, several of which are unlabelled or shared between
   two or three positions (`BF 4%`, `ABF 2%`). Recovering a marginal share per
   discourse from it would require attributing regions by eye.
3. **It is the wrong target anyway.** On the discursive-representation account
   being tested here, a brainstorming tool is not supposed to mirror population
   prevalence. Its job is to make the relevant ways of reasoning available. The
   balance criterion therefore asks whether any discourse is systematically
   marginalised, against a **uniform reference of 1/6**, not against prevalence.

The pre-deliberative baseline keeps its population weights, which is what makes it
useful as the secondary comparison: it is the one place a prevalence claim can be
made at all.

## Reliability caveats to carry into the analysis

- **Discourse F rests on one participant.** Its factor array is a single person's
  sort, more or less. Findings about F are about a discourse the report identified,
  not about a robustly populated position, and should be reported as such.
- **Discourse C is the outlier.** Table 5 shows C correlating −0.10 with A and
  −0.04 with F, while the remaining pairs sit between 0.21 and 0.56. The report
  notes C "stands apart" and is "the most distinguishable". Expect C to be the
  easiest of the six to detect and the other five to be harder to separate from
  one another — the mirror of the problem the four-discourse baseline had.
- **A, B, D, E and F all correlate moderately to strongly.** With five of six
  baselines that similar, winner-takes-all assignment between them will be
  unstable. This is a stronger version of the problem already documented at
  paragraph level in the four-discourse run, and it is the reason coverage and
  presence matter more here than argmax share.
