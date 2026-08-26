# Discursive representation results (paragraph level — SECONDARY)

> **Status: secondary.** The reported analysis is sentence-level and lives in `sentence_summary_*.md`. This file assigns whole *paragraphs* to a single discourse, which is a coarser unit and gives materially different numbers: argmax on a ~5-sentence block erases every minority discourse inside a mixed paragraph, so the distribution here is far more concentrated (TVD ~0.42 against ~0.27 at sentence level, with one model assigning no paragraphs at all to discourse F). These essays mix discourses *within* paragraphs, which is why the paragraph is the wrong unit for this question. Kept as a robustness check on the choice of unit — the ordering of discourses survives it — and not as a result to quote. Do not take figures from this file.

Baseline: **Pre-deliberative mapping study (four discourses)**

Prompt: `brainstorm_australian`  |  Embedding model: `infgrad/Jasper-Token-Compression-600M`  |  Assignment margin: 0.0

297 paragraphs from 30 essays across 3 models.

## Expected representation (Australian population)

Observed prevalence of each discourse in the Australian population, renormalised over the active discourses.

| Discourse | Name | Expected share |
|---|---|---|
| A | Scientific Progress | 41.7% |
| B | Principled Concern | 38.9% |
| C | Profound Concern | 19.4% |

Discourse D excluded from the analysis: expected share of zero in the Australian population. Paragraphs that most resemble an excluded discourse are assigned to their nearest surviving one rather than being set aside.

**0.3% of paragraphs** would have gone to D had it stayed in the running. That is how much of the distribution below is redistributed by the exclusion.

## Observed representation, by model

| Model | A | B | C | TVD from target |
|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 24.5% | 61.3% | 14.2% | 0.224 |
| `moonshotai/kimi-k3` | 33.3% | 43.1% | 23.5% | 0.083 |
| `z-ai/glm-5.2` | 20.2% | 78.7% | 1.1% | 0.398 |

Deviation from the expected share, in percentage points (positive = over-represented):

| Model | A (pp) | B (pp) | C (pp) |
|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | -17.1 | +22.4 | -5.3 |
| `moonshotai/kimi-k3` | -8.3 | +4.2 | +4.1 |
| `z-ai/glm-5.2` | -21.4 | +39.8 | -18.3 |

## Spread across the ten repetitions

Standard deviation of each model's per-essay shares. Large values mean the model's representation is unstable from one essay to the next, which matters as much as the average for a brainstorming interface.

| Model | A | B | C |
|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0.131 | 0.125 | 0.125 |
| `moonshotai/kimi-k3` | 0.092 | 0.116 | 0.125 |
| `z-ai/glm-5.2` | 0.109 | 0.132 | 0.032 |

## Mean cosine similarity, by model

Assignment is winner-takes-all, so it hides how close the contest was. These are the raw mean similarities to each baseline.

| Model | A | B | C |
|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 0.736 | 0.750 | 0.731 |
| `moonshotai/kimi-k3` | 0.695 | 0.699 | 0.693 |
| `z-ai/glm-5.2` | 0.731 | 0.758 | 0.728 |

## Robustness: centred similarities

The same proportions after each discourse's similarity column is centred on its corpus mean, removing the advantage of a baseline that is generically close to everything. If these differ sharply from the headline table, the headline is being driven by baseline attractiveness rather than by discourse content.

| Model | A | B | C | TVD from target |
|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 36.8% | 35.8% | 27.4% | 0.079 |
| `moonshotai/kimi-k3` | 37.3% | 28.4% | 34.3% | 0.149 |
| `z-ai/glm-5.2` | 32.6% | 51.7% | 15.7% | 0.128 |

## Sensitivity to the assignment margin

| Margin | Unassigned | A | B | C |
|---|---|---|---|---|
| 0.000 | 0.0% | 26.3% | 60.3% | 13.5% |
| 0.002 | 5.1% | 26.2% | 60.3% | 13.5% |
| 0.005 | 16.5% | 24.2% | 63.3% | 12.5% |
| 0.010 | 30.0% | 21.6% | 69.2% | 9.1% |
| 0.020 | 57.9% | 15.2% | 80.8% | 4.0% |

## Caveats

- Excluding a discourse redistributes rather than removes its paragraphs: anything closest to D now counts towards A, B, C. `paragraph_similarities.csv` keeps the raw similarity to every baseline, so the size of that effect is recoverable.
- The baselines are not equally distinctive. Check the baseline-to-baseline similarities printed by `src/embed.py`: if two baselines are very close, the split between them is not reliable.
- Winner-takes-all assignment turns small similarity differences into whole paragraphs. The margin sensitivity table above is the check on that.
