# Sentence-level representation: airtime, stance and availability

Baseline: **Post-deliberative jury map (six discourses)**

Prompt: `brainstorm_australian`  |  Statement form: `depersonalised`  |  NLI: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

1610 sentences from 30 essays, 3 models.

| Code | Discourse |
|---|---|
| A | Beneficial Scientific Progress |
| B | Social Benefits/Precautionary Risks |
| C | Principled Constraints |
| D | Revolutionary Medicine |
| E | Profound Social Risks |
| F | Libertarian Revolutionary Medicine |

## 1. Airtime — whose themes get engaged

Cosine similarity, sentence against discourse description. This measures topical engagement only; it is deliberately blind to whether the sentence endorses or rejects what it engages with.

| Discourse | Observed | Reference | Deviation (pp) |
|---|---|---|---|
| A | 8.9% | 16.7% | -7.8 |
| B | 15.8% | 16.7% | -0.8 |
| C | 36.3% | 16.7% | +19.6 |
| D | 24.4% | 16.7% | +7.7 |
| E | 7.8% | 16.7% | -8.8 |
| F | 6.8% | 16.7% | -9.9 |

**Total variation distance: 0.273**  |  Evenness: 0.893

| Model | A | B | C | D | E | F | TVD |
|---|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 9.8% | 16.3% | 39.4% | 20.9% | 8.4% | 5.2% | 0.269 |
| `moonshotai/kimi-k3` | 6.8% | 12.7% | 38.2% | 24.5% | 6.3% | 11.5% | 0.293 |
| `z-ai/glm-5.2` | 9.9% | 18.4% | 30.7% | 28.5% | 8.7% | 3.8% | 0.276 |

## 2. Stance — left standing, or raised and defused

Of the sentences that engage each discourse's themes, how many go on to align with its position, contradict it, or merely report it without commitment. `reported` is the expected majority: a briefing that attributes a view to others neither asserts nor denies it, and NLI correctly abstains.

Note the column name. `aligned` is **not** the same as voiced: because the NLI stage detects denial far more readily than assertion, most of what lands in that column is the text denying something the discourse also denies. That is a real form of agreement, but it is not the essay speaking in the discourse's voice.

| Discourse | Sentences | Aligned | Reported | Dismissed | Net alignment |
|---|---|---|---|---|---|
| A | 143 | 37.1% | 31.5% | 31.5% | +0.028 |
| B | 255 | 62.0% | 26.3% | 11.8% | +1.108 |
| C | 584 | 86.0% | 12.0% | 2.1% | +4.822 |
| D | 393 | 86.0% | 13.0% | 1.0% | +2.205 |
| E | 126 | 35.7% | 34.1% | 30.2% | +0.527 |
| F | 109 | 25.7% | 38.5% | 35.8% | -0.114 |

Net alignment is the mean of `alignment(sentence, its own discourse)`. A negative value means the corpus engages that discourse's themes more to deny them than to affirm them.

### Dismissal asymmetry, by discourse and model

| Model | A net | B net | C net | D net | E net | F net |
|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | -0.134 | +0.868 | +4.371 | +2.112 | +0.417 | +0.133 |
| `moonshotai/kimi-k3` | +0.050 | +1.515 | +4.664 | +2.329 | +0.939 | -0.228 |
| `z-ai/glm-5.2` | +0.201 | +1.074 | +5.703 | +2.178 | +0.352 | -0.163 |

## 3. Representation = engaged and not dismissed

Airtime shares recomputed after removing sentences that engage a discourse in order to deny it. A discourse is represented when its themes are raised and its position is left standing.

| Discourse | Airtime | Not dismissed | Reference | Deviation (pp) |
|---|---|---|---|---|
| A | 8.9% | 6.8% | 16.7% | -9.9 |
| B | 15.8% | 15.6% | 16.7% | -1.1 |
| C | 36.3% | 39.7% | 16.7% | +23.0 |
| D | 24.4% | 27.0% | 16.7% | +10.3 |
| E | 7.8% | 6.1% | 16.7% | -10.6 |
| F | 6.8% | 4.9% | 16.7% | -11.8 |

**TVD after removing dismissals: 0.333** (airtime alone: 0.273)

## 4. Existence — are the human discourses identifiable in the output?

Before asking how much space each discourse gets, ask whether the measurement can tell them apart at all. Two things have to hold: each discourse must claim some sentences, and the claim must be more than a coin flip between near-identical baselines.

| Discourse | Sentences engaged | Represented | Mean cosine | Mean winning margin |
|---|---|---|---|---|
| A | 143 | 98 | 0.5965 | 0.0131 |
| B | 255 | 225 | 0.6039 | 0.0105 |
| C | 584 | 572 | 0.6158 | 0.0109 |
| D | 393 | 389 | 0.6094 | 0.0146 |
| E | 126 | 88 | 0.5991 | 0.0076 |
| F | 109 | 70 | 0.5957 | 0.0149 |

**6 of 6 discourses are recovered somewhere in the corpus** — every one appears in at least one essay.

Corpus-level existence is a weak test: with 1610 sentences and 6 discourses, argmax assignment will hand every discourse something whether or not the essays genuinely articulate it. The informative unit is the individual essay, which is what a participant would actually read — sections 5 and 6.

## 5. Coverage and reliability — how much of the space does one essay open?

A discourse counts as available in an essay when at least 2 sentences, and at least 2% of the essay, engage it without dismissing it. Coverage is how many of the 6 it makes available. This is the central brainstorming criterion: a participant reads one essay, not thirty.

| Model | Mean coverage | Worst essay | Best essay | Essays at full coverage |
|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 5.4 / 6 | 4 | 6 | 5 / 10 |
| `moonshotai/kimi-k3` | 5.1 / 6 | 4 | 6 | 2 / 10 |
| `z-ai/glm-5.2` | 5.0 / 6 | 4 | 6 | 1 / 10 |
| **pooled** | **5.2 / 6** | 4 | 6 | **8 / 30** |

### Reliability — how often each discourse turns up

Share of that model's ten essays in which the discourse is available. A discourse at 10/10 can be counted on; one at 3/10 means a participant using the tool once is unlikely to encounter it at all.

| Model | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 9/10 | 10/10 | 10/10 | 10/10 | 8/10 | 7/10 |
| `moonshotai/kimi-k3` | 7/10 | 10/10 | 10/10 | 10/10 | 6/10 | 8/10 |
| `z-ai/glm-5.2` | 8/10 | 10/10 | 10/10 | 10/10 | 10/10 | 2/10 |
| **pooled** | 24/30 | 30/30 | 30/30 | 30/30 | 24/30 | 17/30 |

Distribution of coverage across all 30 essays:

| Discourses available | Essays |
|---|---|
| 6 / 6 | 8 |
| 5 / 6 | 19 |
| 4 / 6 | 3 |

### Sensitivity — how much of the above is the threshold

Presence turns on very small counts: in the median essay the weaker discourses get two or three non-dismissed sentences each. The floor is therefore doing a lot of the work, and moving it moves the headline.

| Min. sentences | Mean coverage | Full coverage | A | B | C | D | E | F |
|---|---|---|---|---|---|---|---|---|
| 1 | 5.73 | 22/30 | 29/30 | 30/30 | 30/30 | 30/30 | 26/30 | 27/30 |
| 2 ← | 5.17 | 8/30 | 24/30 | 30/30 | 30/30 | 30/30 | 24/30 | 17/30 |
| 3 | 4.63 | 3/30 | 16/30 | 30/30 | 30/30 | 30/30 | 21/30 | 12/30 |
| 4 | 4.23 | 2/30 | 13/30 | 30/30 | 30/30 | 30/30 | 14/30 | 10/30 |
| 5 | 3.40 | 1/30 | 6/30 | 29/30 | 30/30 | 30/30 | 3/30 | 4/30 |

The *ordering* is stable — B, C and D are available in every essay at every threshold, and A, E and F are the weak ones throughout. The *levels* are not: read them as threshold-dependent, not as counts.

### Null model — how much of this is arithmetic

Coverage is bounded by balance. A discourse holding a small share of the ~48 scored sentences in an essay will drop below the floor by chance alone some of the time, whatever the essay is doing. This reallocates each essay's non-dismissed sentences at random in the corpus-wide proportions and recomputes coverage, 5,000 times.

Note what the reference is: the pooled shares come from this corpus, not from the human study. The null therefore asks whether any single essay is unusually concentrated *given how often the models write each discourse overall* -- it cannot ask whether those overall rates are themselves adequate. That question belongs to the uniform reference in section 6.

| | Mean coverage |
|---|---|
| Observed | **5.17** / 6 |
| Random allocation, same pooled shares | 5.31 (95% 5.07–5.53) |

Observed coverage sits inside the null interval, so it adds nothing beyond the imbalance already reported in section 6: the shortfall from 6/6 is what the pooled shares produce on their own.

## 6. Balance — is each discourse given room, or present in name only?

Coverage asks whether a discourse appears. Balance asks whether it appears with enough space to be engaged, contested or developed. A discourse is counted as **marginalised** when it takes less than 50% of an even share (under 8.3% of represented sentences).

| Model | A | B | C | D | E | F | Evenness | TVD |
|---|---|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 7.1% ⚠ | 16.6% | 42.9% | 23.1% | 6.1% ⚠ | 4.2% ⚠ | 0.832 | 0.327 |
| `moonshotai/kimi-k3` | 5.0% ⚠ | 13.2% | 41.8% | 27.1% | 5.0% ⚠ | 8.0% ⚠ | 0.830 | 0.355 |
| `z-ai/glm-5.2` | 8.3% | 16.9% | 33.8% | 31.4% | 7.2% ⚠ | 2.4% ⚠ | 0.847 | 0.320 |
| **pooled** | 6.8% ⚠ | 15.6% | 39.7% | 27.0% | 6.1% ⚠ | 4.9% ⚠ | 0.843 | 0.333 |

**3 of 6 discourses are marginalised pooled across the corpus**: A (Beneficial Scientific Progress), E (Profound Social Risks), F (Libertarian Revolutionary Medicine).

Evenness is normalised entropy: 1.0 is a perfectly even split across the 6 discourses, 0.0 is one discourse taking everything. It is reported alongside TVD because the two differ in what they punish — TVD is distance from the reference, evenness is concentration regardless of which discourse dominates.

Within-essay balance (mean over essays, so a model that covers everything by averaging lopsided essays does not score well here):

| Model | Mean evenness | Mean TVD |
|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0.803 | 0.351 |
| `moonshotai/kimi-k3` | 0.803 | 0.360 |
| `z-ai/glm-5.2` | 0.823 | 0.340 |

## Caveats

- The NLI stage detects denial far more readily than assertion (7.84% of pairs vs 0.92%). This asymmetry was predicted before the run and is a property of the instrument, not a finding about the essays: reported speech does not entail the proposition reported, so the model abstains on voicing while still catching explicit rejection. This is why the stance column is labelled `aligned` rather than `voiced`, and why the dismissal rate is the only part of section 2 that should be read as a result.
- Assignment is winner-takes-all over baselines that are all about the same topic. Mean gap between the winning and runner-up discourse is 0.0119 on a cosine scale where the discourses sit at 0.603 from the average sentence. Small differences therefore decide whole sentences. Section 4 reports the margin per discourse; where it is thin, the split between that discourse and its nearest neighbour is not reliable.
- Airtime is stance-blind by construction. A sentence that engages a discourse's themes counts towards it whatever it says about them; that is what section 3 corrects for.
- 14.8% of sentences open with an unresolved reference. Run `python src/stance.py --context` for the robustness pass that gives those sentences their predecessor.
- The null model in section 5 assumes sentences fall independently. They do not: a paragraph tends to stay with one discourse, so real essays are more clustered than a multinomial draw. Clustering makes low counts for a rare discourse *more* likely than the null implies, so the true mechanical baseline probably sits a little below the figure reported there, and the essay-level effect a little above. Resampling paragraphs rather than sentences would settle it and is not run.
- **Discourse F rests on a single jury participant** (OP12 p. 145). Its factor array is close to one person's Q-sort. Findings about F concern a discourse the report identified, not a robustly populated position.
- **The six are not equally distinguishable.** The report's own Table 5 puts C at −0.10 with A and −0.04 with F while every other pair sits between 0.21 and 0.56. Expect C to separate cleanly and A, B, D, E and F to be harder to tell apart — which inflates the apparent instability of the split among those five.
- The reference distribution is uniform by choice, not by measurement. It encodes the brainstorming criterion (no relevant discourse systematically marginalised), not a claim about how common these positions are. See `post-delib_baseline.NOTES.md`.
