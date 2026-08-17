# Testing LLMs as discursive agents in deliberative processes

Do LLMs, used as neutral brainstorming interfaces for a deliberative process,
represent the range of public discourses on an issue *proportionally*?

This repository runs one exploratory case: **human genome editing**. Three
open-weight models are each asked ten times to write a brainstorming report for
a citizens' assembly. Every paragraph of every report is embedded and compared
by cosine similarity against the four discourses identified in the Australian
Citizens' Jury on Genome Editing. The share of paragraphs falling to each
discourse is then set against the share of Australians who actually hold it.

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
`--prompt NAME` to select the condition.

```bash
python src/generate.py    # 1. essays from the LLMs      -> data/generated/
python src/segment.py     # 2. essays into paragraphs    -> data/processed/
python src/embed.py       # 3. embed paragraphs+baseline -> data/processed/
python src/analyse.py     # 4. similarity + proportions  -> results/<prompt>/
python src/compare.py     # 5. conditions side by side   -> results/comparison.md
```

`src/config.py` holds every tunable choice; the scripts hard-code nothing.

### Output

| File | What it holds |
|---|---|
| `results/<prompt>/summary.md` | The readable report for one condition |
| `results/comparison.md` | Both conditions side by side |
| `results/<prompt>/paragraph_similarities.csv` | Every paragraph, its similarity to all four baselines, its assignment |
| `results/<prompt>/proportions_by_model.csv` | Headline shares and TVD per model |
| `results/<prompt>/proportions_by_run.csv` | Per-essay shares, for the spread across repetitions |
| `results/<prompt>/sensitivity_margin.csv` | Shares at each assignment margin |

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
| `pre-delib_discourses.txt` | The four mapping-study discourses, verbatim from the report |
| `pre-delib_discourses_clean.txt` | **The baseline.** Same text with report mechanics removed |
| `pre-delib_discourses_clean.CHANGES.md` | Every edit made in cleaning, and why |
| `pre-delib_discourse_mapping_factor_array.csv` | Table 2: 46 Q-statements × factor scores |
| `proportional_representation.csv` | Table 9: discourse distribution in Australia |
| `pre-delib_discourse_weights_australia.csv` | **The target.** Weights derived from the above |
| `pre-delib_discourse_weights_australia.NOTES.md` | How the weights were derived, and their problems |
| `post-delib_discourses.txt` | The six post-deliberation positions — not used yet |

Source for all of it is the Australian Citizens' Jury on Genome Editing
(Nicol, Paxton & Niemeyer et al., *Genome Editing: Formulating an Australian
Community Response*, Occasional Paper No 12, 2022).

The four discourses and their expected shares of the discourse-loaded
Australian population:

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

**Discourse D is excluded.** No Australian in the population survey loaded on
Agnosticism at the reported precision, so its expected share is exactly zero —
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

## Open questions

- Are the four baselines distinct enough to separate? `src/embed.py` prints the
  baseline-to-baseline similarity matrix; the human correlations (OP12 Table 3)
  put B–C closest at 0.48 and A–C furthest at 0.22, so that ordering is the
  thing to check against.
- Is a paragraph the right unit? Report paragraphs often canvass several views
  at once, which winner-takes-all assignment flattens.
- Should the six post-deliberation positions be used instead? They are richer,
  but no population weights exist for them.
