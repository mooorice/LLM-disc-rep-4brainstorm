# LLM-as-judge: is each way of reasoning available?

Baseline: **Pre-deliberative mapping study (four discourses)**

Judge: `openai/gpt-oss-120b` @ temperature 1.0  |  Prompt: `brainstorm_australian`  |  270 judgements over 30 essays x 3 discourses x 3 replicates

The judge sees one unlabelled discourse description and one essay, and rates how developed that way of reasoning is (**presence**), how the essay positions it (**treatment**) and how much room it gets (**extent**). Descriptions are blinded — position letters and names are stripped — so the judge reads the reasoning rather than a label. Replicates are collapsed by majority vote, ties broken toward the more conservative rating.

A discourse counts as **available** when presence reaches `articulated` and treatment is not `dismissed`.

## 1. Existence — what the judge finds, corpus-wide

| Discourse | Articulated | Mentioned only | Absent | Dismissed | Available |
|---|---|---|---|---|---|
| A Scientific Progress | 2/30 | 9/30 | 19/30 | 0/30 | **2/30** |
| B Principled Concern | 29/30 | 0/30 | 1/30 | 0/30 | **29/30** |
| C Profound Concern | 3/30 | 13/30 | 14/30 | 0/30 | **3/30** |

**3 of 3 discourses are available in at least one essay** — every one is recovered somewhere.

## 2. Coverage — how much of the space one essay opens

| Model | Mean coverage | Worst | Best | Full coverage |
|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 1.1 / 3 | 1 | 2 | 0 / 10 |
| `moonshotai/kimi-k3` | 1.1 / 3 | 0 | 2 | 0 / 10 |
| `z-ai/glm-5.2` | 1.2 / 3 | 1 | 2 | 0 / 10 |
| **pooled** | **1.1 / 3** | 0 | 2 | **0 / 30** |

### Reliability — how often each discourse is available

| Model | A | B | C |
|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0/10 | 10/10 | 1/10 |
| `moonshotai/kimi-k3` | 0/10 | 9/10 | 2/10 |
| `z-ai/glm-5.2` | 2/10 | 10/10 | 0/10 |
| **pooled** | 2/30 | 29/30 | 3/30 |

Distribution of coverage:

| Discourses available | Essays |
|---|---|
| 2 / 3 | 5 |
| 1 / 3 | 24 |
| 0 / 3 | 1 |

## 3. Balance — how much room each discourse gets

Share of total *extent*, where each judgement contributes 0 for none up to 3 for several paragraphs. Unlike the cosine measure, nothing here depends on assigning each sentence to exactly one discourse, so a passage that serves two perspectives can count for both.

| Discourse | Share of extent | Reference | Deviation (pp) |
|---|---|---|---|
| A | 15.2% ⚠ | 41.7% | -26.5 |
| B | 62.1% | 38.9% | +23.2 |
| C | 22.8% | 19.4% | +3.3 |

**TVD from reference: 0.265**  |  Evenness: 0.837  |  1 of 3 marginalised: A

### Treatment — how the essays position each discourse

| Discourse | Endorsed | Neutral | Dismissed | Not applicable |
|---|---|---|---|---|
| A | 0 | 14 | 0 | 16 |
| B | 0 | 30 | 0 | 0 |
| C | 0 | 19 | 0 | 11 |

## 4. Judge against the cosine pipeline

Two independent measurements of the same thing. Where they agree, the finding is not an artefact of either instrument.

| Discourse | Judge: available | Cosine: available |
|---|---|---|
| A | 2/30 | 30/30 |
| B | 29/30 | 30/30 |
| C | 3/30 | 30/30 |

| Discourse | Judge share of extent | Cosine share of sentences |
|---|---|---|
| A | 15.2% | 33.4% |
| B | 62.1% | 49.2% |
| C | 22.8% | 17.3% |

Correlation between the two share vectors: **r = +0.777** over 3 discourses.

The two instruments broadly agree on which discourses get room, which is reassuring for both.

## 5. Checking the judge

A judge is an instrument too. These are the checks that decide how much of the above to believe.

### Replicate agreement

Each pair was judged 3 times independently at temperature 1.0.

| | Unanimous | Mean agreement |
|---|---|---|
| Presence | 63.3% | 0.852 |
| Treatment | 68.9% | 0.896 |

| Discourse | Presence agreement | Treatment agreement |
|---|---|---|
| A | 0.800 | 0.844 |
| B | 0.978 | 0.989 |
| C | 0.778 | 0.856 |

### Quote verification

Every non-absent judgement had to supply a verbatim quote, checked against the essay automatically. Quotes joined by an ellipsis are checked fragment by fragment.

| | Value |
|---|---|
| Judgements requiring a quote | 175 |
| Fully verified | 95.4% |
| Mean share of quoted words found | 0.960 |
| Wholly unfindable (0% matched) | 2.9% |
| Unparseable responses | 0 of 270 |

A quote that cannot be found is the judge inventing support for a rating. That rate is the ceiling on how much any individual verdict can be trusted.

### Do different discourses get the same evidence?

1 cases where one passage was cited as support for two different discourses within the same essay.

Some overlap is expected and legitimate — the report's own Table 5 has several discourse pairs correlating above 0.5, so a passage can genuinely serve both. Heavy overlap would mean the judge is no better at separating the six than the embedding was.

| Discourse pair | Shared quotes | Essays |
|---|---|---|
| B – C | 1 | 1/30 |

Across the six discourses, the judge cites a mean of **2.3 genuinely distinct passages per essay**.

| Distinct passages cited | Essays |
|---|---|
| 1 | 4 |
| 2 | 12 |
| 3 | 14 |

### The treatment scale did not vary

**Every one of the 63 applicable judgements came back `neutral`. Not one `dismissed`, not one `endorsed`.**

A constant is not a measurement. Either these briefings really are uniformly even-handed — plausible, since they were written to canvass all sides — or the judge will not apply the ends of this scale. Nothing here distinguishes the two, so the treatment column should be reported as uninformative rather than as evidence of even-handedness.

This matters because it contradicts the NLI stage, which put dismissal at 31.5% for A, 30.2% for E and 35.8% for F. Those figures were already suspect — part of the signal traced to a rewriting template in the depersonalised statements — but the disagreement between the two instruments is now total, and unresolved. The dismissal question needs an instrument that has been shown to be capable of returning both answers.

## Caveats

- One judge model. `openai/gpt-oss-120b` has its own reading of what counts as a way of reasoning, and nothing here separates that from the essays. A second judge would be the obvious control.
- Blinding removes the position labels, but the descriptions still carry the report's own vocabulary. A judge can match distinctive phrasing without engaging the reasoning, exactly as the embedding did.
- Presence, treatment and extent are the judge's ratings, not ground truth. Replicate agreement bounds their stability; it says nothing about their accuracy. A hand-labelled calibration set is still the missing piece.
- Extent is a coarse four-point scale, so the balance shares in section 3 are approximate and should not be read past the first significant figure.
