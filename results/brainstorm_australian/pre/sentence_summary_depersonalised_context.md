# Sentence-level representation: airtime, stance and availability

> **Status: diagnostic only — not used for scoring.** This file is output from the anaphora context pass, which failed its own validity check (the combined premise tracks the prepended sentence's score more closely than the target's). Report `sentence_summary_depersonalised.md` instead.

Baseline: **Pre-deliberative mapping study (four discourses)**

Prompt: `brainstorm_australian`  |  Statement form: `depersonalised`  |  NLI: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`  |  **anaphora context pass**

1610 sentences from 30 essays, 3 models. Discourse D excluded (expected share zero).

| Code | Discourse |
|---|---|
| A | Scientific Progress |
| B | Principled Concern |
| C | Profound Concern |

## 1. Airtime — whose themes get engaged

Cosine similarity, sentence against discourse description. This measures topical engagement only; it is deliberately blind to whether the sentence endorses or rejects what it engages with.

| Discourse | Observed | Reference | Deviation (pp) |
|---|---|---|---|
| A | 44.6% | 41.7% | +2.9 |
| B | 39.3% | 38.9% | +0.4 |
| C | 16.1% | 19.4% | -3.3 |

**Total variation distance: 0.033**  (a uniform split over 3 discourses would score 0.139)

| Model | A | B | C | TVD |
|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 40.7% | 42.8% | 16.5% | 0.039 |
| `moonshotai/kimi-k3` | 46.4% | 33.5% | 20.2% | 0.054 |
| `z-ai/glm-5.2` | 47.3% | 41.0% | 11.7% | 0.078 |

## 2. Stance — left standing, or raised and defused

Of the sentences that engage each discourse's themes, how many go on to align with its position, contradict it, or merely report it without commitment. `reported` is the expected majority: a briefing that attributes a view to others neither asserts nor denies it, and NLI correctly abstains.

Note the column name. `aligned` is **not** the same as voiced: because the NLI stage detects denial far more readily than assertion, most of what lands in that column is the text denying something the discourse also denies. That is a real form of agreement, but it is not the essay speaking in the discourse's voice.

| Discourse | Sentences | Aligned | Reported | Dismissed | Net alignment |
|---|---|---|---|---|---|
| A | 718 | 28.4% | 30.9% | 40.7% | +0.010 |
| B | 632 | 85.1% | 14.4% | 0.5% | +3.457 |
| C | 260 | 62.3% | 25.8% | 11.9% | +1.594 |

Net alignment is the mean of `alignment(sentence, its own discourse)`. A negative value means the corpus engages that discourse's themes more to deny them than to affirm them.

### Dismissal asymmetry, by discourse and model

| Model | A net | B net | C net |
|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | +0.241 | +3.079 | +1.235 |
| `moonshotai/kimi-k3` | -0.132 | +3.545 | +1.526 |
| `z-ai/glm-5.2` | -0.082 | +3.850 | +2.312 |

## 3. Representation = engaged and not dismissed

Airtime shares recomputed after removing sentences that engage a discourse in order to deny it. A discourse is represented when its themes are raised and its position is left standing.

| Discourse | Airtime | Not dismissed | Reference | Deviation (pp) |
|---|---|---|---|---|
| A | 44.6% | 33.2% | 41.7% | -8.5 |
| B | 39.3% | 49.0% | 38.9% | +10.1 |
| C | 16.1% | 17.8% | 19.4% | -1.6 |

**TVD after removing dismissals: 0.101** (airtime alone: 0.033)

## 4. Existence — are the human discourses identifiable in the output?

Before asking how much space each discourse gets, ask whether the measurement can tell them apart at all. Two things have to hold: each discourse must claim some sentences, and the claim must be more than a coin flip between near-identical baselines.

| Discourse | Sentences engaged | Represented | Mean cosine | Mean winning margin |
|---|---|---|---|---|
| A | 718 | 426 | 0.6129 | 0.0181 |
| B | 632 | 629 | 0.6062 | 0.0205 |
| C | 260 | 229 | 0.6043 | 0.0112 |

**3 of 3 discourses are recovered somewhere in the corpus** — every one appears in at least one essay.

Corpus-level existence is a weak test: with 1610 sentences and 3 discourses, argmax assignment will hand every discourse something whether or not the essays genuinely articulate it. The informative unit is the individual essay, which is what a participant would actually read — sections 5 and 6.

## 5. Coverage and reliability — how much of the space does one essay open?

A discourse counts as available in an essay when at least 2 sentences, and at least 2% of the essay, engage it without dismissing it. Coverage is how many of the 3 it makes available. This is the central brainstorming criterion: a participant reads one essay, not thirty.

| Model | Mean coverage | Worst essay | Best essay | Essays at full coverage |
|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 3.0 / 3 | 3 | 3 | 10 / 10 |
| `moonshotai/kimi-k3` | 3.0 / 3 | 3 | 3 | 10 / 10 |
| `z-ai/glm-5.2` | 3.0 / 3 | 3 | 3 | 10 / 10 |
| **pooled** | **3.0 / 3** | 3 | 3 | **30 / 30** |

### Reliability — how often each discourse turns up

Share of that model's ten essays in which the discourse is available. A discourse at 10/10 can be counted on; one at 3/10 means a participant using the tool once is unlikely to encounter it at all.

| Model | A | B | C |
|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 10/10 | 10/10 | 10/10 |
| `moonshotai/kimi-k3` | 10/10 | 10/10 | 10/10 |
| `z-ai/glm-5.2` | 10/10 | 10/10 | 10/10 |
| **pooled** | 30/30 | 30/30 | 30/30 |

Distribution of coverage across all 30 essays:

| Discourses available | Essays |
|---|---|
| 3 / 3 | 30 |

### Sensitivity — how much of the above is the threshold

Presence turns on very small counts: in the median essay the weaker discourses get two or three non-dismissed sentences each. The floor is therefore doing a lot of the work, and moving it moves the headline.

| Min. sentences | Mean coverage | Full coverage | A | B | C |
|---|---|---|---|---|---|
| 1 | 3.00 | 30/30 | 30/30 | 30/30 | 30/30 |
| 2 ← | 3.00 | 30/30 | 30/30 | 30/30 | 30/30 |
| 3 | 2.97 | 29/30 | 30/30 | 30/30 | 29/30 |
| 4 | 2.93 | 28/30 | 30/30 | 30/30 | 28/30 |
| 5 | 2.83 | 25/30 | 30/30 | 30/30 | 25/30 |

The *ordering* is stable — B, C and D are available in every essay at every threshold, and A, E and F are the weak ones throughout. The *levels* are not: read them as threshold-dependent, not as counts.

### Null model — how much of this is arithmetic

Coverage is bounded by balance. A discourse holding a small share of the ~41 scored sentences in an essay will drop below the floor by chance alone some of the time, whatever the essay is doing. This reallocates each essay's non-dismissed sentences at random in the corpus-wide proportions and recomputes coverage, 5,000 times.

Note what the reference is: the pooled shares come from this corpus, not from the human study. The null therefore asks whether any single essay is unusually concentrated *given how often the models write each discourse overall* -- it cannot ask whether those overall rates are themselves adequate. That question belongs to the uniform reference in section 6.

| | Mean coverage |
|---|---|
| Observed | **3.00** / 3 |
| Random allocation, same pooled shares | 3.00 (95% 2.97–3.00) |

Observed coverage sits inside the null interval, so it adds nothing beyond the imbalance already reported in section 6: the shortfall from 3/3 is what the pooled shares produce on their own.

## 6. Balance — is each discourse given room, or present in name only?

Coverage asks whether a discourse appears. Balance asks whether it appears with enough space to be engaged, contested or developed. A discourse is counted as **marginalised** when it takes less than 50% of an even share (under 16.7% of represented sentences).

| Model | A | B | C | Evenness | TVD |
|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 32.7% | 50.6% | 16.7% | 0.919 | 0.117 |
| `moonshotai/kimi-k3` | 33.8% | 42.9% | 23.2% | 0.973 | 0.078 |
| `z-ai/glm-5.2` | 33.2% | 53.1% | 13.7% ⚠ | 0.887 | 0.142 |
| **pooled** | 33.2% | 49.0% | 17.8% | 0.931 | 0.101 |

**0 of 3 discourses are marginalised pooled across the corpus** — every discourse clears the floor.

Evenness is normalised entropy: 1.0 is a perfectly even split across the 3 discourses, 0.0 is one discourse taking everything. It is reported alongside TVD because the two differ in what they punish — TVD is distance from the reference, evenness is concentration regardless of which discourse dominates.

Within-essay balance (mean over essays, so a model that covers everything by averaging lopsided essays does not score well here):

| Model | Mean evenness | Mean TVD |
|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0.902 | 0.130 |
| `moonshotai/kimi-k3` | 0.947 | 0.111 |
| `z-ai/glm-5.2` | 0.857 | 0.155 |

## Caveats

- The NLI stage detects denial far more readily than assertion (7.80% of pairs vs 1.35%). This asymmetry was predicted before the run and is a property of the instrument, not a finding about the essays: reported speech does not entail the proposition reported, so the model abstains on voicing while still catching explicit rejection. This is why the stance column is labelled `aligned` rather than `voiced`, and why the dismissal rate is the only part of section 2 that should be read as a result.
- Assignment is winner-takes-all over baselines that are all about the same topic. Mean gap between the winning and runner-up discourse is 0.0179 on a cosine scale where the discourses sit at 0.608 from the average sentence. Small differences therefore decide whole sentences. Section 4 reports the margin per discourse; where it is thin, the split between that discourse and its nearest neighbour is not reliable.
- Airtime is stance-blind by construction. A sentence that engages a discourse's themes counts towards it whatever it says about them; that is what section 3 corrects for.
- 14.8% of sentences open with an unresolved reference. Run `python src/stance.py --context` for the robustness pass that gives those sentences their predecessor.
- The null model in section 5 assumes sentences fall independently. They do not: a paragraph tends to stay with one discourse, so real essays are more clustered than a multinomial draw. Clustering makes low counts for a rare discourse *more* likely than the null implies, so the true mechanical baseline probably sits a little below the figure reported there, and the essay-level effect a little above. Resampling paragraphs rather than sentences would settle it and is not run.
