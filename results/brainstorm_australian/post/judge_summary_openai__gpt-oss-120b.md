# LLM-as-judge: is each way of reasoning available?

Baseline: **Post-deliberative jury map (six discourses)**

Judge: `openai/gpt-oss-120b` @ temperature 1.0  |  Prompt: `brainstorm_australian`  |  539 judgements over 30 essays x 6 discourses x 3 replicates  |  **1 unreadable responses dropped** of 540

The judge sees one unlabelled discourse description and one essay, and rates how developed that way of reasoning is (**presence**), how the essay positions it (**treatment**) and how much room it gets (**extent**). Descriptions are blinded — position letters and names are stripped — so the judge reads the reasoning rather than a label. Replicates are collapsed by majority vote, ties broken toward the more conservative rating.

A discourse counts as **available** when presence reaches `articulated` and treatment is not `dismissed`.

## 1. Existence — what the judge finds, corpus-wide

| Discourse | Articulated | Mentioned only | Absent | Dismissed | Available |
|---|---|---|---|---|---|
| A Beneficial Scientific Progress | 3/30 | 12/30 | 15/30 | 0/30 | **3/30** |
| B Social Benefits/Precautionary Risks | 3/30 | 18/30 | 9/30 | 0/30 | **3/30** |
| C Principled Constraints | 2/30 | 2/30 | 26/30 | 0/30 | **2/30** |
| D Revolutionary Medicine | 8/30 | 17/30 | 5/30 | 0/30 | **8/30** |
| E Profound Social Risks | 1/30 | 6/30 | 23/30 | 0/30 | **1/30** |
| F Libertarian Revolutionary Medicine | 1/30 | 3/30 | 26/30 | 0/30 | **1/30** |

**6 of 6 discourses are available in at least one essay** — every one is recovered somewhere.

## 2. Coverage — how much of the space one essay opens

| Model | Mean coverage | Worst | Best | Full coverage |
|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0.3 / 6 | 0 | 1 | 0 / 10 |
| `moonshotai/kimi-k3` | 0.8 / 6 | 0 | 2 | 0 / 10 |
| `z-ai/glm-5.2` | 0.7 / 6 | 0 | 3 | 0 / 10 |
| **pooled** | **0.6 / 6** | 0 | 3 | **0 / 30** |

### Reliability — how often each discourse is available

| Model | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 1/10 | 1/10 | 0/10 | 1/10 | 0/10 | 0/10 |
| `moonshotai/kimi-k3` | 0/10 | 1/10 | 1/10 | 5/10 | 0/10 | 1/10 |
| `z-ai/glm-5.2` | 2/10 | 1/10 | 1/10 | 2/10 | 1/10 | 0/10 |
| **pooled** | 3/30 | 3/30 | 2/30 | 8/30 | 1/30 | 1/30 |

Distribution of coverage:

| Discourses available | Essays |
|---|---|
| 3 / 6 | 1 |
| 2 / 6 | 3 |
| 1 / 6 | 9 |
| 0 / 6 | 17 |

## 3. Balance — how much room each discourse gets

Share of total *extent*, where each judgement contributes 0 for none up to 3 for several paragraphs. Unlike the cosine measure, nothing here depends on assigning each sentence to exactly one discourse, so a passage that serves two perspectives can count for both.

| Discourse | Share of extent | Reference | Deviation (pp) |
|---|---|---|---|
| A | 19.8% | 16.7% | +3.1 |
| B | 27.1% | 16.7% | +10.5 |
| C | 6.8% ⚠ | 16.7% | -9.9 |
| D | 32.2% | 16.7% | +15.5 |
| E | 10.2% | 16.7% | -6.5 |
| F | 4.0% ⚠ | 16.7% | -12.7 |

**TVD from reference: 0.291**  |  Evenness: 0.883  |  2 of 6 marginalised: C, F

### Treatment — how the essays position each discourse

| Discourse | Endorsed | Neutral | Dismissed | Not applicable |
|---|---|---|---|---|
| A | 0 | 16 | 0 | 14 |
| B | 0 | 23 | 0 | 7 |
| C | 0 | 6 | 0 | 24 |
| D | 0 | 27 | 0 | 3 |
| E | 0 | 8 | 0 | 22 |
| F | 0 | 4 | 0 | 26 |

## 4. Judge against the cosine pipeline

Two independent measurements of the same thing. Where they agree, the finding is not an artefact of either instrument.

| Discourse | Judge: available | Cosine: available |
|---|---|---|
| A | 3/30 | 24/30 |
| B | 3/30 | 30/30 |
| C | 2/30 | 30/30 |
| D | 8/30 | 30/30 |
| E | 1/30 | 24/30 |
| F | 1/30 | 17/30 |

| Discourse | Judge share of extent | Cosine share of sentences |
|---|---|---|
| A | 19.8% | 6.6% |
| B | 27.1% | 15.6% |
| C | 6.8% | 39.6% |
| D | 32.2% | 27.3% |
| E | 10.2% | 6.1% |
| F | 4.0% | 4.8% |

Correlation between the two share vectors: **r = +0.141** over 6 discourses.

The two instruments disagree about which discourses get room. Given that cosine was shown not to recover the human discourse structure at all (`src/validate_baseline.py`), the judge is the more credible of the two — but the disagreement should be reported, not resolved by preference.

## 5. Checking the judge

A judge is an instrument too. These are the checks that decide how much of the above to believe.

### Replicate agreement

Each pair was judged 3 times independently at temperature 1.0.

| | Unanimous | Mean agreement |
|---|---|---|
| Presence | 36.1% | 0.773 |
| Treatment | 51.7% | 0.838 |

| Discourse | Presence agreement | Treatment agreement |
|---|---|---|
| A | 0.800 | 0.844 |
| B | 0.700 | 0.811 |
| C | 0.828 | 0.850 |
| D | 0.700 | 0.867 |
| E | 0.756 | 0.767 |
| F | 0.856 | 0.889 |

### Quote verification

Every non-absent judgement had to supply a verbatim quote, checked against the essay automatically. Quotes joined by an ellipsis are checked fragment by fragment.

| | Value |
|---|---|
| Judgements requiring a quote | 264 |
| Fully verified | 83.3% |
| Mean share of quoted words found | 0.838 |
| Wholly unfindable (0% matched) | 15.5% |
| Unparseable responses | 0 of 539 |

A quote that cannot be found is the judge inventing support for a rating. That rate is the ceiling on how much any individual verdict can be trusted.

### Do different discourses get the same evidence?

16 cases where one passage was cited as support for two different discourses within the same essay.

Some overlap is expected and legitimate — the report's own Table 5 has several discourse pairs correlating above 0.5, so a passage can genuinely serve both. Heavy overlap would mean the judge is no better at separating the six than the embedding was.

| Discourse pair | Shared quotes | Essays |
|---|---|---|
| A – D | 12 | 12/30 |
| B – E | 2 | 2/30 |
| D – E | 1 | 1/30 |
| D – F | 1 | 1/30 |

Across the six discourses, the judge cites a mean of **4.2 genuinely distinct passages per essay**.

| Distinct passages cited | Essays |
|---|---|
| 2 | 3 |
| 3 | 5 |
| 4 | 9 |
| 5 | 9 |
| 6 | 4 |

**A and D are supported by the same passage in 12 of 30 essays.** They are the most correlated pair in the report's own factor arrays, so some overlap is expected — but at this rate the judge is not separately evidencing them, and coverage counts them twice.

Treating A and D as one discourse gives mean coverage 0.60 of 5, against 0.60 of 6 as reported above. Read the headline coverage figure with that in mind.

### The treatment scale did not vary

**Every one of the 84 applicable judgements came back `neutral`. Not one `dismissed`, not one `endorsed`.**

A constant is not a measurement. Either these briefings really are uniformly even-handed — plausible, since they were written to canvass all sides — or the judge will not apply the ends of this scale. Nothing here distinguishes the two, so the treatment column should be reported as uninformative rather than as evidence of even-handedness.

This matters because it contradicts the NLI stage, which put dismissal at 31.5% for A, 30.2% for E and 35.8% for F. Those figures were already suspect — part of the signal traced to a rewriting template in the depersonalised statements — but the disagreement between the two instruments is now total, and unresolved. The dismissal question needs an instrument that has been shown to be capable of returning both answers.

## Caveats

- One judge model. `openai/gpt-oss-120b` has its own reading of what counts as a way of reasoning, and nothing here separates that from the essays. A second judge would be the obvious control.
- Blinding removes the position labels, but the descriptions still carry the report's own vocabulary. A judge can match distinctive phrasing without engaging the reasoning, exactly as the embedding did.
- Presence, treatment and extent are the judge's ratings, not ground truth. Replicate agreement bounds their stability; it says nothing about their accuracy. A hand-labelled calibration set is still the missing piece.
- Extent is a coarse four-point scale, so the balance shares in section 3 are approximate and should not be read past the first significant figure.
