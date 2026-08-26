# Discursive representation results (paragraph level — SECONDARY)

> **Status: secondary.** The reported analysis is sentence-level and lives in `sentence_summary_*.md`. This file assigns whole *paragraphs* to a single discourse, which is a coarser unit and gives materially different numbers: argmax on a ~5-sentence block erases every minority discourse inside a mixed paragraph, so the distribution here is far more concentrated (TVD ~0.42 against ~0.27 at sentence level, with one model assigning no paragraphs at all to discourse F). These essays mix discourses *within* paragraphs, which is why the paragraph is the wrong unit for this question. Kept as a robustness check on the choice of unit — the ordering of discourses survives it — and not as a result to quote. Do not take figures from this file.

Baseline: **Post-deliberative jury map (six discourses)**

Prompt: `brainstorm_australian`  |  Embedding model: `infgrad/Jasper-Token-Compression-600M`  |  Assignment margin: 0.0

297 paragraphs from 30 essays across 3 models.

## Reference distribution (even split)

The six post-deliberative discourses have no usable population weights, and on the brainstorming criterion they should not be judged by prevalence anyway. The reference is an even split: a yardstick for whether any way of reasoning is systematically marginalised, not a claim that the six are equally common.

| Discourse | Name | Expected share |
|---|---|---|
| A | Beneficial Scientific Progress | 16.7% |
| B | Social Benefits/Precautionary Risks | 16.7% |
| C | Principled Constraints | 16.7% |
| D | Revolutionary Medicine | 16.7% |
| E | Profound Social Risks | 16.7% |
| F | Libertarian Revolutionary Medicine | 16.7% |

## Observed representation, by model

| Model | A | B | C | D | E | F | TVD from target |
|---|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0.9% | 30.2% | 35.8% | 24.5% | 7.5% | 0.9% | 0.406 |
| `moonshotai/kimi-k3` | 1.0% | 30.4% | 41.2% | 18.6% | 2.0% | 6.9% | 0.402 |
| `z-ai/glm-5.2` | 2.2% | 41.6% | 23.6% | 31.5% | 1.1% | 0.0% | 0.466 |

Deviation from the expected share, in percentage points (positive = over-represented):

| Model | A (pp) | B (pp) | C (pp) | D (pp) | E (pp) | F (pp) |
|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | -15.7 | +13.5 | +19.2 | +7.9 | -9.1 | -15.7 |
| `moonshotai/kimi-k3` | -15.7 | +13.7 | +24.5 | +2.0 | -14.7 | -9.8 |
| `z-ai/glm-5.2` | -14.4 | +24.9 | +6.9 | +14.8 | -15.5 | -16.7 |

## Spread across the ten repetitions

Standard deviation of each model's per-essay shares. Large values mean the model's representation is unstable from one essay to the next, which matters as much as the average for a brainstorming interface.

| Model | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0.029 | 0.093 | 0.097 | 0.076 | 0.106 | 0.024 |
| `moonshotai/kimi-k3` | 0.029 | 0.056 | 0.126 | 0.082 | 0.040 | 0.063 |
| `z-ai/glm-5.2` | 0.042 | 0.140 | 0.180 | 0.100 | 0.040 | 0.000 |

## Mean cosine similarity, by model

Assignment is winner-takes-all, so it hides how close the contest was. These are the raw mean similarities to each baseline.

| Model | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0.700 | 0.734 | 0.736 | 0.724 | 0.722 | 0.704 |
| `moonshotai/kimi-k3` | 0.666 | 0.693 | 0.699 | 0.689 | 0.681 | 0.679 |
| `z-ai/glm-5.2` | 0.690 | 0.720 | 0.723 | 0.717 | 0.708 | 0.683 |

## Robustness: centred similarities

The same proportions after each discourse's similarity column is centred on its corpus mean, removing the advantage of a baseline that is generically close to everything. If these differ sharply from the headline table, the headline is being driven by baseline attractiveness rather than by discourse content.

| Model | A | B | C | D | E | F | TVD from target |
|---|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 12.3% | 27.4% | 13.2% | 18.9% | 16.0% | 12.3% | 0.129 |
| `moonshotai/kimi-k3` | 10.8% | 15.7% | 11.8% | 12.7% | 13.7% | 35.3% | 0.186 |
| `z-ai/glm-5.2` | 9.0% | 33.7% | 11.2% | 31.5% | 13.5% | 1.1% | 0.318 |

## Sensitivity to the assignment margin

| Margin | Unassigned | A | B | C | D | E | F |
|---|---|---|---|---|---|---|---|
| 0.000 | 0.0% | 1.3% | 33.7% | 34.0% | 24.6% | 3.7% | 2.7% |
| 0.002 | 12.8% | 1.2% | 34.0% | 32.4% | 25.9% | 3.5% | 3.1% |
| 0.005 | 29.0% | 0.9% | 36.5% | 31.3% | 25.1% | 2.4% | 3.8% |
| 0.010 | 49.5% | 0.7% | 37.3% | 28.7% | 26.7% | 3.3% | 3.3% |
| 0.020 | 82.2% | 0.0% | 47.2% | 22.6% | 24.5% | 0.0% | 5.7% |

## Caveats

- The baselines are not equally distinctive. Check the baseline-to-baseline similarities printed by `src/embed.py`: if two baselines are very close, the split between them is not reliable.
- Winner-takes-all assignment turns small similarity differences into whole paragraphs. The margin sensitivity table above is the check on that.
