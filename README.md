# Testing LLMs as discursive agents in deliberative processes

Can an LLM, used as a brainstorming tool, make available the relevant ways of
reasoning that emerge through substantive human deliberation?

This repository runs one exploratory case: **human genome editing**. Three
open-weight models are each asked ten times to write a brainstorming report for
a citizens' assembly. The reports are then scored against the discourses
identified in the Australian Citizens' Jury on Genome Editing.

The test is not whether the models reproduce the distribution of raw opinion in
the population. On the discursive-representation account, what should be
represented are the socially grounded ways of understanding an issue — coherent
configurations of reasons, values and assumptions. A brainstorming tool earns
its place by making that space available for citizens to engage, contest or
develop, including perspectives that would otherwise stay implicit. So the
questions are:

| | Question |
|---|---|
| **Existence** | Are the human discourses identifiable in the LLM output at all? |
| **Coverage** | How many of them does a single essay make available? |
| **Reliability** | Does the same discourse appear run after run, or by luck? |
| **Balance** | Does each get enough space to be engaged, or is it present in name only? |
| **Social grounding** | Do the models generate discourses with no human counterpart? *(open)* |

### Two benchmarks

**Primary: the six post-deliberative discourses.** These emerged *after* the
citizens' jury had worked through the issue. They are not a finer partition of
the pre-deliberative four — deliberation clarified and differentiated the space,
so the six are configurations of reasons that only became visible once people
had argued about it. If deliberation itself produced this richer space, a useful
brainstorming tool should be able to surface it.

**Secondary: the four pre-deliberative discourses.** These describe raw opinion
in the Australian population, and are the only place a prevalence claim can be
grounded. Retained as a comparison and as evidence about social grounding.

Both are selected with `--baseline post` / `--baseline pre`; see
`data/human_baseline/post-delib_baseline.NOTES.md`.

## Running it

```bash
python src/generate.py --smoke    # one call, to check credentials first
./scripts/run_experiment.sh       # everything: both conditions, all stages
```

That is the whole experiment — 60 essays (3 models × 10 repetitions × 2 prompt
conditions), then segmentation, embedding and analysis for each condition, then
the cross-condition comparison. Expect roughly 20 minutes, nearly all of it
generation.

It is safe to re-run. Essays already on disk are skipped, so an interrupted run
resumes where it stopped and costs nothing to restart. To regenerate a single
bad essay, delete its JSON file and run again.

### Stage by stage

Each stage writes to disk, so any one can be re-run alone. All of them take
`--prompt NAME` for the condition and `--baseline pre|post` for the discourse map.

```bash
python src/generate.py         # 1.  essays from the LLMs       -> data/generated/
python src/segment.py          # 2a. essays into paragraphs     -> data/processed/
python src/sentences.py        # 2b. paragraphs into sentences  -> data/processed/
python src/embed.py            # 3a. embed paragraphs+baseline  -> data/processed/
python src/embed_sentences.py  # 3b. embed sentences            -> data/processed/
python src/stance.py           # 3c. NLI against the statements -> data/processed/
python src/analyse.py          # 4a. paragraph-level result     -> results/<prompt>/<baseline>/
python src/analyse_sentences.py# 4b. sentence-level result      -> results/<prompt>/<baseline>/
python src/compare.py          # 5.  conditions side by side    -> results/comparison.md
```

Generation, segmentation and stance scoring are baseline-independent: the same
essays and the same NLI matrices serve both discourse maps. Only embedding
against the baselines and the analyses need re-running per baseline.

`src/config.py` holds every tunable choice; the scripts hard-code nothing.

### Output

Results are namespaced `results/<prompt>/<baseline>/`, so the two discourse maps
never overwrite each other.

| File | What it holds |
|---|---|
| `sentence_summary_<form>.md` | **The main report.** Airtime, stance, then existence / coverage / reliability / balance |
| `summary.md` | The paragraph-level result, kept for comparison |
| `attribution.md` | Directness: how often each discourse is asserted vs put in someone else's mouth |
| `essay_coverage_<form>.csv` | Per essay: which discourses were available, coverage, evenness |
| `attribution.csv` | Every sentence with its attribution category and discourse |
| `sentence_scores_<form>.csv` | Every sentence, its similarity and alignment to every discourse, its verdict |
| `paragraph_similarities.csv` | Every paragraph, its similarity to every baseline, its assignment |
| `proportions_by_model.csv` | Headline shares and TVD per model |
| `proportions_by_run.csv` | Per-essay shares, for the spread across repetitions |
| `sensitivity_margin.csv` | Shares at each assignment margin |

### Validating the instrument

Two scripts test whether the measures do what they are being asked to do. Both
are part of the run, not optional extras.

```bash
python src/validate_stance.py    # can cosine tell endorsement from rejection? (no)
python src/validate_baseline.py  # does the embedding recover the discourse structure? (no)
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-v1-..." > .env
```

Generation goes through OpenRouter; embedding runs locally and wants a GPU
(about 1.5 GB of VRAM at bfloat16).

## Data

| File | What it is |
|---|---|
| `post-delib_discourses.txt` | The six post-deliberative positions, verbatim from the report |
| `post-delib_discourses_clean.txt` | **The primary baseline.** Same text, unwrapped, mechanics removed |
| `post-delib_factor_array.csv` | Table 6: 45 statements × six discourses, z-scores |
| `post-delib_baseline.NOTES.md` | How it was built and validated; why there are no weights |
| `pre-delib_discourses.txt` | The four mapping-study discourses, verbatim |
| `pre-delib_discourses_clean.txt` | **The secondary baseline.** Same text, mechanics removed |
| `pre-delib_discourses_clean.CHANGES.md` | Every edit made in cleaning, and why |
| `pre-delib_discourse_mapping_factor_array.csv` | Table 2: 46 Q-statements × four discourses |
| `pre-delib_q_statements_depersonalised.csv` | Both statement phrasings side by side |
| `proportional_representation.csv` | Table 9: discourse distribution in Australia |
| `pre-delib_discourse_weights_australia.csv` | The prevalence target, for the secondary baseline |
| `pre-delib_discourse_weights_australia.NOTES.md` | How the weights were derived, and their problems |

Source for all of it is the Australian Citizens' Jury on Genome Editing
(Nicol, Paxton & Niemeyer et al., *Genome Editing: Formulating an Australian
Community Response*, Occasional Paper No 12, 2022).

**The six post-deliberative discourses** (primary benchmark, no population
weights — the reference is an even split):

| | Discourse |
|---|---|
| A | Beneficial Scientific Progress |
| B | Social Benefits / Precautionary Risks |
| C | Principled Constraints |
| D | Revolutionary Medicine |
| E | Profound Social Risks |
| F | Libertarian Revolutionary Medicine |

**The four pre-deliberative discourses** and their shares of the
discourse-loaded Australian population (secondary benchmark):

| | Discourse | Expected | |
|---|---|---|---|
| A | Scientific Progress | 41.7% | |
| B | Principled Concern | 38.9% | |
| C | Profound Concern | 19.4% | |
| D | Agnosticism | 0.0% | excluded from the analysis |

## Decisions worth knowing about

Each of these is a fork in the road that changes the results. They are recorded
here rather than buried in the code.

**Both prompt conditions are run.** `prompts/brainstorm_australian.txt` names
the deliberating public as Australian, matching the population the weights
describe; `prompts/brainstorm_generic.txt` is the identical prompt with the
country removed. The Australian weights are only strictly the right target for
the first, but the second is the more general claim about LLMs as brainstorming
interfaces. `results/comparison.md` reports the distance between them.

**The six post-deliberative discourses are the primary benchmark.** They are the
ways of reasoning that deliberation itself brought out, which is the space a
brainstorming tool should be able to surface. The four pre-deliberative
discourses describe raw opinion and are kept as the secondary comparison. See
`post-delib_baseline.NOTES.md` for the argument.

**The six are judged against an even split, not against prevalence.** There is a
published distribution over them (OP12 Figure 10) but the report disclaims it as
provisional, it is a Venn diagram whose regions cannot be read as marginals, and
on the brainstorming criterion prevalence is the wrong target anyway. The
question is whether any relevant way of reasoning is systematically
marginalised. The four-discourse baseline keeps its population weights, which is
what makes it useful as the comparison.

**Discourse D of the four is excluded.** No Australian in the population survey
loaded on Agnosticism at the reported precision, so its expected share is zero —
not a proportion anything can be scored against. A, B and C already carry the
whole expected distribution, so dropping D costs nothing on the target side.

It does change assignment, though: a paragraph that most resembles Agnosticism
is now forced onto its nearest surviving discourse rather than set aside. Each
summary reports what share of paragraphs that affects, and
`paragraph_similarities.csv` keeps the similarity to D for every paragraph, so
the effect stays measurable. Restoring D is one line in
`config.ACTIVE_DISCOURSES`, with no re-embedding.

**Only 35% of Australians load on any discourse** (53% no significant loading,
12% confounded). The weights, and therefore the proportions, are conditional on
being discourse-loaded.

**Assignment is winner-takes-all** over four cosine similarities, which turns
small differences into whole paragraphs. `config.SENSITIVITY_MARGINS` re-runs
the proportions requiring the winner to lead by a margin, leaving close cases
unassigned — loosely the analogue of a confounded loading.

**Embedding is symmetric.** Jasper has a retrieval query prompt; we do not use
it, because a paragraph and a discourse description are two texts being compared,
not a query against a corpus. `config.USE_ASYMMETRIC_PROMPTS` switches this.

**Reasoning traces are stored but never analysed.** What matters is the text an
assembly participant would actually read.

## What the validation found

Both instruments were tested rather than assumed, and both failed in ways that
constrain what can be claimed.

**Cosine similarity is stance-blind, and mildly stance-inverted.** Across six
matched endorse/reject pairs, the *rejecting* passage scored closer to the
statement in four of six. Rejecting a claim means restating it and then arguing,
which makes rejections lexically denser on the statement. Cosine is therefore
used only as an *airtime* measure — which themes are engaged — and never for
stance.

**The embedding does not recover the structure of the human discourse space.**
Correlating the report's published inter-discourse correlations against the
cosine similarities between the same descriptions gives r = +0.15 across the six
discourses' fifteen pairs (and −0.28 across the four's six pairs). The pair the
report calls most distinct, A–C, is one of the embedding's closest. Cosine to a
discourse description measures shared topic, not shared reasoning.

**Argmax assignment is partly a keyword router.** Description length correlates
with argmax share at r = +0.73, and unique terms carry large lifts — a sentence
containing "passed" is 7× more likely to be assigned to E, one containing
"parents" 7× more likely to go to F. `src/validate_baseline.py` prints all of
this for whichever baseline is active.

**NLI works on assertive text but abstains on reported speech.** It detects
denial far more readily than assertion (7.8% of pairs vs 0.9%), because "many
Australians worry that X" genuinely does not entail X. It is therefore used as a
*dismissal* detector only, and the results distinguish "aligned" from "voiced".

**The dismissal result did not replicate.** NLI put dismissal at 31.5% for
discourse A and 35.8% for F. Two independent judges returned **zero** dismissals
across 1,079 judgements. Part of the NLI signal traced to a rewriting template
in the depersonalised statements; the earlier dismissal percentages should not
be reported.

## Open questions

- **The judge rubric is the open problem.** Two blinded open-weight judges rate
  presence with Cohen's kappa of **+0.027** — chance agreement. Coverage is
  5.47/6 under `gemma-4-31b-it` and 0.60/6 under `gpt-oss-120b`. The
  disagreement is perfectly one-directional (gemma rates higher 152 times,
  gpt-oss zero), so they share an ordering but not a threshold: gpt-oss requires
  a discourse's policy preferences to be articulated, gemma accepts its
  reasoning core. Fixing where the bar sits is now the bottleneck, not adding
  instruments.
- **Social grounding.** Do the models produce coherent ways of reasoning with no
  counterpart in the human map? More perspectives are not better if they are not
  socially meaningful. This needs engagement with meta-consensus before it can
  be operationalised.
- **Is a sentence the right unit?** Sentence-level assignment changed the
  headline substantially against paragraph-level, because long text embeds
  toward a generic topic centroid. Neither unit is obviously correct.
- **Anaphora needs coreference resolution, not concatenation.** The context pass
  has been run and analysed (`src/analyse_context.py`). It confirms the main
  results are robust — 3.7% of verdicts change, headline TVD moves 0.001 — but
  it should not be adopted as the primary scoring: the combined premise
  correlates more with the *context* sentence's own score (0.79) than with the
  target sentence's (0.64), so it double-counts the predecessor's stance.
  Rewriting the pronoun in place would keep the premise about the sentence.
