# Two judges, same question

Baseline: **Pre-deliberative mapping study (four discourses)**  |  Prompt: `brainstorm_australian`

- Judge 0: `google/gemma-4-31b-it`
- Judge 1: `openai/gpt-oss-120b`

90 matched judgements over 30 essays x 3 discourses, each the majority of 3 replicates.

## 1. How much do they agree?

Raw agreement flatters a rater that always says the same thing, so Cohen's kappa is reported alongside: it subtracts the agreement two judges would reach by chance given how often each uses each label.

| Variable | Raw agreement | Cohen's kappa |
|---|---|---|
| presence | 44.4% | +0.138 |
| treatment | 71.1% | +0.051 |
| extent | 48.9% | +0.318 |
| **available** | **56.7%** | **+0.248** |

An em dash means one judge used only a single label for that variable, so there is no variation for kappa to correct against.

## 2. How each judge uses the scale

| Presence | `gemma-4-31b-it` | `gpt-oss-120b` |
|---|---|---|
| absent | 2.2% | 37.8% |
| mentioned | 16.7% | 24.4% |
| articulated | 81.1% | 37.8% |

| Treatment | `gemma-4-31b-it` | `gpt-oss-120b` |
|---|---|---|
| endorsed | 0.0% | 0.0% |
| neutral | 98.9% | 70.0% |
| dismissed | 0.0% | 0.0% |
| not_applicable | 1.1% | 30.0% |

Where they differ on presence, `gemma-4-31b-it` rates it higher 50 times and `gpt-oss-120b` rates it higher 0 times. Disagreement is **one-directional**: one judge is simply more generous than the other, rather than the two disagreeing case by case.

## 3. Does the conclusion survive the change of judge?

### Availability per discourse

| Discourse | `gemma-4-31b-it` | `gpt-oss-120b` | Agreement | Kappa |
|---|---|---|---|---|
| A Scientific Progress | 24/30 | 2/30 | 26.7% | -0.015 |
| B Principled Concern | 30/30 | 29/30 | 96.7% | +0.000 |
| C Profound Concern | 19/30 | 3/30 | 46.7% | +0.054 |

### Coverage

| Judge | Mean coverage | Full coverage | Range |
|---|---|---|---|
| `google/gemma-4-31b-it` | 2.43 / 3 | 16/30 | 1–3 |
| `openai/gpt-oss-120b` | 1.13 / 3 | 0/30 | 0–2 |

**Coverage ranges from 1.13 to 2.43 of 3 depending on which model is asked.**

That spread is wider than any difference between the three models under test, which means coverage as measured here is at least as much a property of the judge as of the essays. It has to be reported as a range, with the rubric that produced it.

### Do the judges rank the three models the same way?

This is the comparison the experiment is actually for. Absolute levels can move with the rater as long as the ordering holds.

| Model under test | `gemma-4-31b-it` | `gpt-oss-120b` |
|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 2.80 | 1.10 |
| `moonshotai/kimi-k3` | 2.80 | 1.10 |
| `z-ai/glm-5.2` | 1.70 | 1.20 |

**The two judges rank the models differently.** `gemma-4-31b-it`: deepseek-v4-pro-0813 > kimi-k3 > glm-5.2  `gpt-oss-120b`: glm-5.2 > deepseek-v4-pro-0813 > kimi-k3. Nothing comparative survives here without a third opinion.

### Balance

| Discourse | `gemma-4-31b-it` | `gpt-oss-120b` | Reference |
|---|---|---|---|
| A | 29.4% | 15.2% | 41.7% |
| B | 44.2% | 62.1% | 38.9% |
| C | 26.4% | 22.8% | 19.4% |

| Judge | TVD from reference | Evenness |
|---|---|---|
| `google/gemma-4-31b-it` | 0.122 | 0.976 |
| `openai/gpt-oss-120b` | 0.265 | 0.837 |

Correlation between the two judges' share vectors: **r = +0.952**.

## Caveats

- Two judges is the minimum for this comparison, not a sufficient number. Where they disagree, nothing here identifies which is right.
- Both judges saw an identical prompt and identical blinded descriptions, so any shared bias from the rubric affects both and is invisible to this comparison. Agreement is evidence against rater idiosyncrasy, not against a badly framed question.
- Kappa is undefined where a judge used a single label throughout. That is itself the finding for that variable.
