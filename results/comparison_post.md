# Prompt conditions compared

`brainstorm_australian` names the deliberating public as Australian; `brainstorm_generic` is the identical prompt with the country removed. The Australian population weights are the target in both cases, but they are only strictly the right target for the first.

Not yet run, and omitted below: brainstorm_generic.

## Pooled across all models

| Condition | A | B | C | D | E | F | TVD from target |
|---|---|---|---|---|---|---|---|
| *target* | 16.7% | 16.7% | 16.7% | 16.7% | 16.7% | 16.7% | — |
| `brainstorm_australian` | 1.3% | 33.7% | 34.0% | 24.6% | 3.7% | 2.7% | 0.423 |

## By model

Whether the prompt effect is a property of the setup or of a particular model.

| Model | Condition | A | B | C | D | E | F | TVD from target |
|---|---|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | `brainstorm_australian` | 0.9% | 30.2% | 35.8% | 24.5% | 7.5% | 0.9% | 0.406 |
| `moonshotai/kimi-k3` | `brainstorm_australian` | 1.0% | 30.4% | 41.2% | 18.6% | 2.0% | 6.9% | 0.402 |
| `z-ai/glm-5.2` | `brainstorm_australian` | 2.2% | 41.6% | 23.6% | 31.5% | 1.1% | 0.0% | 0.466 |
