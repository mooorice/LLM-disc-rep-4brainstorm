# Discourse weights from the Australian population survey

Derived from `proportional_representation.csv`, which transcribes **Table 9: Selection
Stratification: Discursive Criteria** (OP12 Appendix D.2.2, pp. 149–150). Output:
`pre-delib_discourse_weights_australia.csv`.

## Provenance of the "Australia %" column

The report's note to Table 9 is explicit: these percentages "were obtained from the
**population survey in stage 9** of the project, after recruitment" — the national
survey of **n = 1008** (project flow diagram, p. 11). They are not census figures and
not the recruitment pool; the other columns in the CSV are recruitment and jury
figures and play no part in these weights.

## Derivation

Each discourse appears twice in Table 9, as a moderate loading (>0.5) and a high
loading (>0.7). These are disjoint categories, so a discourse's population share is
their sum. The remaining categories — "no significant loading" (53%) and "confounded
loading" (12%) — belong to no single discourse and are excluded.

| Discourse | Moderate | High | Total | Weight (all loadings) | Weight (high only) |
|---|---|---|---|---|---|
| A Scientific Progress | 11% | 4% | 15% | **0.417** | 0.571 |
| B Principled Concern | 12% | 2% | 14% | **0.389** | 0.286 |
| C Profound Concern | 6% | 1% | 7% | **0.194** | 0.143 |
| D Agnosticism | 0% | 0% | 0% | **0.000** | 0.000 |

Weights normalise the totals over the discourse-loaded population only (36% of
respondents; the published percentages sum to 101% through rounding). The
`weight_all_loadings` column is the one to use by default: 0.5 is the report's own
"significant loading" threshold, and the high-only variant rests on just 7 percentage
points, which visibly changes the answer (A jumps from 0.42 to 0.57).

## Three things this does to the design

**1. Discourse D has an expected share of zero.** Taken literally, proportional
representation demands the LLMs never voice Agnosticism, and *any* D-similar paragraph
counts as over-representation. Note the percentages are rounded integers, so with
n = 1008 a "0%" means fewer than about 5 respondents — possibly, but not necessarily,
zero. D is also the discourse the mapping study itself described as marginal ("just 5%
of the study variance"), and the recruitment pool found only 3 people out of 228 who
loaded on it. So this is a real finding about D, not an artifact.

Options, in rough order of defensibility: report D's target as 0 and treat observed
D-similarity as over-representation (honest, and arguably the more interesting result);
apply a continuity correction, treating "0%" as 0.25% for a target near 0.7%; or drop D
from the proportionality statistic and handle it descriptively. This needs deciding
before the metric is written, since it changes what the headline number means.

**2. Only 35% of Australians load on any single discourse.** 53% show no significant
loading and 12% are confounded across discourses. The weights are therefore conditional
on being discourse-loaded, which is the right conditioning for this design — an LLM
paragraph cannot represent "no significant loading" — but it should be stated, because
it means the baseline describes a *minority* of the population, and the largest single
group in Australia holds none of these four positions cleanly.

**3. The population-survey loadings used a reduced statement set.** Appendix D.2.2
records that some project stages used a sub-sample of the 46 statements with "indexed"
loadings rather than full factor loadings, validated at r > 0.9 against full-set
loadings. Good enough, worth a footnote.

## Cross-check available

Table 3 (p. 134) gives inter-discourse correlations: A–B 0.32, A–C 0.22, A–D 0.36,
B–C 0.48, B–D 0.28, C–D 0.34. B and C are the closest pair, so their cosine-similarity
profiles should be expected to correlate too — useful as a sanity check that the
embedding baselines behave, and a caution against reading small A-vs-B or B-vs-C gaps
as meaningful.
