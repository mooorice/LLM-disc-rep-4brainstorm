# Two judges, same question

Baseline: **Post-deliberative jury map (six discourses)**  |  Prompt: `brainstorm_australian`

- Judge 0: `google/gemma-4-31b-it`
- Judge 1: `openai/gpt-oss-120b`

180 matched judgements over 30 essays x 6 discourses, each the majority of 3 replicates.

## 1. How much do they agree?

Raw agreement flatters a rater that always says the same thing, so Cohen's kappa is reported alongside: it subtracts the agreement two judges would reach by chance given how often each uses each label.

| Variable | Raw agreement | Cohen's kappa |
|---|---|---|
| presence | 15.6% | +0.027 |
| treatment | 51.7% | +0.088 |
| extent | 29.4% | +0.092 |
| **available** | **18.9%** | **+0.021** |

An em dash means one judge used only a single label for that variable, so there is no variation for kappa to correct against.

## 2. How each judge uses the scale

| Presence | `gemma-4-31b-it` | `gpt-oss-120b` |
|---|---|---|
| absent | 5.0% | 57.8% |
| mentioned | 3.9% | 32.2% |
| articulated | 91.1% | 10.0% |

| Treatment | `gemma-4-31b-it` | `gpt-oss-120b` |
|---|---|---|
| endorsed | 0.0% | 0.0% |
| neutral | 95.0% | 46.7% |
| dismissed | 0.0% | 0.0% |
| not_applicable | 5.0% | 53.3% |

Where they differ on presence, `gemma-4-31b-it` rates it higher 152 times and `gpt-oss-120b` rates it higher 0 times. Disagreement is **one-directional**: one judge is simply more generous than the other, rather than the two disagreeing case by case.

## 3. Does the conclusion survive the change of judge?

### Availability per discourse

| Discourse | `gemma-4-31b-it` | `gpt-oss-120b` | Agreement | Kappa |
|---|---|---|---|---|
| A Beneficial Scientific Progress | 30/30 | 3/30 | 10.0% | +0.000 |
| B Social Benefits/Precautionary Risks | 30/30 | 3/30 | 10.0% | +0.000 |
| C Principled Constraints | 30/30 | 2/30 | 6.7% | +0.000 |
| D Revolutionary Medicine | 30/30 | 8/30 | 26.7% | +0.000 |
| E Profound Social Risks | 27/30 | 1/30 | 13.3% | -0.002 |
| F Libertarian Revolutionary Medicine | 17/30 | 1/30 | 46.7% | +0.091 |

### Coverage

| Judge | Mean coverage | Full coverage | Range |
|---|---|---|---|
| `google/gemma-4-31b-it` | 5.47 / 6 | 16/30 | 4–6 |
| `openai/gpt-oss-120b` | 0.60 / 6 | 0/30 | 0–3 |

**Coverage ranges from 0.60 to 5.47 of 6 depending on which model is asked.**

That spread is wider than any difference between the three models under test, which means coverage as measured here is at least as much a property of the judge as of the essays. It has to be reported as a range, with the rubric that produced it.

### Do the judges rank the three models the same way?

This is the comparison the experiment is actually for. Absolute levels can move with the rater as long as the ordering holds.

| Model under test | `gemma-4-31b-it` | `gpt-oss-120b` |
|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 5.60 | 0.30 |
| `moonshotai/kimi-k3` | 6.00 | 0.80 |
| `z-ai/glm-5.2` | 4.80 | 0.70 |

**The two judges rank the models differently.** `gemma-4-31b-it`: kimi-k3 > deepseek-v4-pro-0813 > glm-5.2  `gpt-oss-120b`: kimi-k3 > glm-5.2 > deepseek-v4-pro-0813. Nothing comparative survives here without a third opinion.

### Balance

| Discourse | `gemma-4-31b-it` | `gpt-oss-120b` | Reference |
|---|---|---|---|
| A | 17.0% | 19.8% | 16.7% |
| B | 18.1% | 27.1% | 16.7% |
| C | 17.8% | 6.8% | 16.7% |
| D | 17.3% | 32.2% | 16.7% |
| E | 18.4% | 10.2% | 16.7% |
| F | 11.4% | 4.0% | 16.7% |

| Judge | TVD from reference | Evenness |
|---|---|---|
| `google/gemma-4-31b-it` | 0.053 | 0.994 |
| `openai/gpt-oss-120b` | 0.291 | 0.883 |

Correlation between the two judges' share vectors: **r = +0.464**.

## Caveats

- Two judges is the minimum for this comparison, not a sufficient number. Where they disagree, nothing here identifies which is right.
- Both judges saw an identical prompt and identical blinded descriptions, so any shared bias from the rubric affects both and is invisible to this comparison. Agreement is evidence against rater idiosyncrasy, not against a badly framed question.
- Kappa is undefined where a judge used a single label throughout. That is itself the finding for that variable.
