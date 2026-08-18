# First pass against the six post-deliberative discourses

What the existing 30 essays (`brainstorm_australian`, 1,610 sentences, 3 models ×
10 repetitions) look like when scored against the post-deliberative jury map
rather than the pre-deliberative population map.

Generated results: `results/brainstorm_australian/post/`. Instrument checks:
`python src/validate_baseline.py --baseline post`.

---

## The headline

**Switching benchmarks changes the finding, because the old benchmark could not
discriminate.** Against three pre-deliberative discourses, every one of the 30
essays covered all three — 30/30, no variation to explain. Against the six, only
8 of 30 make the whole space available.

Part of that is mechanical: three categories spread over a ~54-sentence essay
leaves each one around 9–24 sentences, so any small floor is cleared
automatically. Six categories put the weaker ones at two or three sentences,
where a floor can bite. But that *is* the point about resolution — the coarse map
returns "all present" regardless of what the essay does, so it cannot register a
perspective being crowded out. The finer map can.

The theoretical claim about deliberation differentiating the discourse space
therefore shows up here as measurement sensitivity, not yet as a result about
the models.

## 1. Existence — are the human discourses identifiable in the output?

All six are recovered somewhere in the corpus. This is close to a free pass:
with 1,610 sentences and six baselines, winner-takes-all assignment hands
everything something. The informative unit is the individual essay.

| Discourse | Sentences engaged | Represented | Mean winning margin |
|---|---|---|---|
| A Beneficial Scientific Progress | 143 | 98 | 0.0131 |
| B Social Benefits / Precautionary Risks | 255 | 225 | 0.0105 |
| C Principled Constraints | 584 | 572 | 0.0109 |
| D Revolutionary Medicine | 393 | 389 | 0.0146 |
| E Profound Social Risks | 126 | 88 | 0.0076 |
| F Libertarian Revolutionary Medicine | 109 | 70 | 0.0149 |

The margins are the thing to notice. The winning discourse beats the runner-up
by about 0.012 on a scale where the six baselines sit 0.82–0.92 from each other.
Assignment is being decided in the third decimal place.

## 2. Coverage and reliability — what one essay makes available

A discourse counts as available when at least 2 sentences, and at least 2% of the
essay, engage it without dismissing it. **In practice only the first clause
matters**: essays run 42–69 sentences, so 2% is under 1.4 sentences and never
binds. Presence is "at least two non-dismissed sentences whose nearest discourse
is this one" — see the caveat at the end of this section, which is load-bearing.

| Model | Mean coverage | Essays at 6/6 |
|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 5.4 / 6 | 5 / 10 |
| `moonshotai/kimi-k3` | 5.1 / 6 | 2 / 10 |
| `z-ai/glm-5.2` | 5.0 / 6 | 1 / 10 |
| **pooled** | **5.2 / 6 (87%)** | **8 / 30** |

No essay drops below 4/6, so the failure is never wholesale — it is the same few
discourses going missing.

**Which ones go missing, out of 30 essays:**

| A | B | C | D | E | F |
|---|---|---|---|---|---|
| 24/30 | 30/30 | 30/30 | 30/30 | 24/30 | **17/30** |

B, C and D are guaranteed. A and E appear four times in five. **F appears in
just over half of runs** — a participant using the tool once has a coin-flip
chance of encountering Libertarian Revolutionary Medicine at all.

The models differ in a way worth noting: GLM finds E in 10/10 but F in only
2/10, while Kimi finds F in 8/10 but E in 6/10. They fail on different
discourses, which means the gap is not a fixed property of the discourse.

### How much of this is real

Two checks, and both cut the finding down.

**The threshold is doing a lot of the work.** Presence turns on counts of two or
three. Moving the floor moves everything:

| Min. sentences | Mean coverage | Full coverage | A | E | F |
|---|---|---|---|---|---|
| 1 | 5.73 | 22/30 | 29/30 | 26/30 | 27/30 |
| **2** | **5.17** | **8/30** | **24/30** | **24/30** | **17/30** |
| 3 | 4.63 | 3/30 | 16/30 | 21/30 | 12/30 |
| 5 | 3.40 | 1/30 | 6/30 | 3/30 | 4/30 |

B, C and D sit at 30/30 at every threshold. So "F appears in 17 of 30 essays" is
not a fact to quote — at a floor of 1 it is 27/30 and at 3 it is 12/30. What
survives is the ordering: B, C, D always available; A, E, F always the weak ones.

**Most of the coverage shortfall is arithmetic.** Coverage is bounded by balance:
a discourse holding 5% of a 54-sentence essay averages under three sentences and
will fall below the floor by chance whatever the essay is doing. Allocating each
essay's sentences at random in the corpus-wide proportions gives mean coverage
**5.46** (95% 5.23–5.67) against an observed **5.17**.

Observed does fall below the null interval, so essays are slightly more
concentrated than the pooled shares alone predict — but the effect is small. The
distance from 6/6 is mostly a restatement of the imbalance in section 3, not an
independent finding. Coverage and balance are close to the same measurement.

## 3. Balance — room to be engaged, or present in name only

Share of represented sentences. ⚠ marks a discourse below half an even share
(under 8.3%).

| | A | B | C | D | E | F | Evenness |
|---|---|---|---|---|---|---|---|
| pooled | 6.8% ⚠ | 15.6% | **39.7%** | 27.0% | 6.1% ⚠ | 4.9% ⚠ | 0.843 |

**Three of six discourses are marginalised.** Two thirds of the represented
material goes to C (Principled Constraints) and D (Revolutionary Medicine) — the
two poles of the standard "promise versus peril" framing. What gets squeezed are
the more specific configurations: A's disease-burden-driven optimism, E's
dystopian social risk, F's libertarian reproductive autonomy.

Consistent across models: all three produce the same shape, evenness 0.83–0.85,
TVD 0.32–0.36. Within-essay evenness is slightly lower (0.80–0.82) than pooled,
so the models are not covering the space by averaging lopsided essays — the
individual essays are lopsided in the same direction.

Robust to statement phrasing: TVD after removing dismissals is 0.333
(depersonalised) and 0.307 (original).

## 4. Directness — whose voice carries each discourse

Airtime asks whether a discourse's themes appear; stance asks whether they are
contradicted. Neither catches the difference between the essay asserting
something and the essay reporting that others assert it. A perspective that only
ever arrives as somebody else's opinion is being described to the reader, not
made available as a way of reasoning they might take up.

Across the corpus, **25.4% of sentences** carry an attribution marker (61–66% of
paragraphs contain at least one). By discourse:

| Discourse | Attributed | vs corpus (pp) |
|---|---|---|
| F Libertarian Revolutionary Medicine | **36.7%** | +11.3 |
| C Principled Constraints | 29.1% | +3.7 |
| A Beneficial Scientific Progress | 28.0% | +2.6 |
| D Revolutionary Medicine | 26.2% | +0.8 |
| B Social Benefits / Precautionary Risks | 15.7% | −9.7 |
| E Profound Social Risks | **12.7%** | −12.7 |

A 24-point spread. F is held at arm's length nearly three times as often as E.

**Attribution and dismissal are separate failures**, and reading them together
is more informative than either alone:

| Discourse | Attributed | Dismissed | Direct *and* not dismissed |
|---|---|---|---|
| A | 28.0% | 31.5% | 51.0% |
| B | 15.7% | 11.8% | **75.7%** |
| C | 29.1% | 2.1% | 69.9% |
| D | 26.2% | 1.0% | 73.3% |
| E | 12.7% | 30.2% | 59.5% |
| F | 36.7% | 35.8% | **35.8%** |

The last column is the strictest reading available from the automated measures:
themes engaged, essay speaking in its own voice, nothing contradicting it.

Two distinct failure modes show up. **E is asserted directly and then
contradicted** — it gets into the essay's own voice more than any other
discourse, and is denied 30% of the time. **F is quarantined** — quoted rather
than asserted, and denied when it does appear. Only about a third of F's
sentences survive both filters.

F is therefore marginalised on every measure at once: present in 17 of 30 essays,
4.9% of represented sentences, most attributed, most dismissed. Whether that says
something about the models or about a discourse resting on a single jury
participant is not separable from this data.

Against the four pre-deliberative discourses the pattern differs — there it is C
(Profound Concern) that is most attributed at 39.2%, with A and B close together
around 22–23%. The finer map redistributes which perspectives get quoted.

## 5. Stance — who gets contradicted

| Discourse | Dismissed | Net alignment |
|---|---|---|
| A | 31.5% | +0.028 |
| B | 11.8% | +1.108 |
| C | 2.1% | +4.822 |
| D | 1.0% | +2.205 |
| E | 30.2% | +0.527 |
| F | **35.8%** | **−0.114** |

The same three discourses that are marginalised by volume are also the ones most
often contradicted when they do appear. F is the only discourse with negative net
alignment: the essays engage libertarian reproductive-autonomy themes mainly in
order to push back on them. A and E are dismissed at roughly 30%.

So the two effects compound. A, E and F get less space *and* a larger share of
that space is spent defusing them. That is why the TVD rises from 0.273 on
airtime alone to 0.333 once dismissals are removed.

---

## 6. Two judges, and what survives them

Both open-weight, neither under test, both blinded, identical prompt and rubric:
`google/gemma-4-31b-it` and `openai/gpt-oss-120b`, 3 replicates per essay-discourse
pair, 1,079 judgements. Full detail in `judge_comparison.md`.

**They do not agree.** Cohen's kappa on presence is **+0.027** — chance. Raw
agreement 15.6%; on availability, 18.9%.

| Presence rating | gemma | gpt-oss |
|---|---|---|
| absent | 5.0% | 57.8% |
| mentioned | 3.9% | 32.2% |
| articulated | **91.1%** | **10.0%** |

Coverage is **5.47/6 under one judge and 0.60/6 under the other**. That spread is
far wider than any difference between the three models being tested, so coverage
as measured here is at least as much a property of the rater as of the essays.

One thing keeps that from being pure noise: the disagreement is **perfectly
one-directional**. Gemma rates presence higher 152 times; gpt-oss rates it higher
**zero** times. The two are not disagreeing case by case — they are applying the
same ordering with the bar in wildly different places. gpt-oss requires the
discourse's *policy preferences* to be articulated, not just its reasoning core,
which is a defensible reading of descriptions that end in policy paragraphs.

**Model ranking does not survive either.** Gemma: kimi > deepseek > glm.
gpt-oss: kimi > glm > deepseek. Kimi is top under both; the other two swap.

### What survives all three instruments

| | Cosine | gemma | gpt-oss |
|---|---|---|---|
| F available | 17/30 | 17/30 | 1/30 |
| F share | 4.8% (lowest) | 11.4% (lowest) | 4.0% (lowest) |
| Best model | — | kimi | kimi |

**F (Libertarian Revolutionary Medicine) is the weakest discourse under every
instrument**, and kimi-k3 is the strongest model under both judges. Those two
claims are what the evidence currently supports.

### The dismissal finding is probably wrong

**Neither judge returned a single `dismissed` or `endorsed` verdict**, in 540 and
539 judgements respectively. The NLI stage had put dismissal at 31.5% for A,
30.2% for E and 35.8% for F.

Part of that NLI signal was already traced to a rewriting template in the
depersonalised statements. Two independent raters now finding no dismissal at
all makes the raise-and-dismiss result look like an artefact rather than a
finding. Two readings remain open — the briefings really are even-handed, or the
rubric's "dismissed" bar is too high for any of them — and nothing here
separates them. Either way the earlier dismissal percentages should not be
reported.

---

## What this cannot support, and why

The instrument was tested rather than assumed, and it failed the test.

**The embedding does not recover the human discourse structure.** Correlating the
report's fifteen published inter-discourse correlations (Table 5) against the
cosine similarities between the same six descriptions gives **r = +0.15**. The
pair the report calls most distinct — A–C at −0.10 — is one of the embedding's
*closest* at 0.868. Cosine to a discourse description measures shared topic, not
shared reasoning. (The four-discourse baseline scores r = −0.28 on the same test,
so this is not new; it is now measured across fifteen pairs instead of six.)

**Description length predicts the winner.** Words in the description correlate
with argmax share at r = +0.73 across the six. With n = 6 that is suggestive
rather than established, and it does not explain everything — B and C are both
214 words and split 16% / 36% — but it means the researchers' writing budget is
part of what is being measured.

**Unique vocabulary routes sentences.** A sentence containing "passed" is 7×
more likely to be assigned to E; "parents" 7× more likely to go to F;
"scientific" 6× to A; "cultural" is 90% predictive of C. Some of these name a
discourse's reasoning. Most name its subject matter.

**The most confidently assigned sentences are often neutral exposition.** B's
top-margin sentence is a list of framing questions for participants. E's is a
neutral description of what germline editing is. Neither expresses a position.

### What survives

The **ordinal** claims. Which discourses go missing, which get contradicted, and
how the three models differ from each other are all comparisons made under the
same bias, and the bias is roughly constant across them.

The **coverage and reliability** results are the sturdiest, because presence is a
weaker claim than proportion — but they are not clean. "F appears in 17/30
essays" partly means "the words *parents*, *wealthy* and *born* appear in 17/30
essays". Read it as a lower bound on a topical signal, not as a measurement of
whether the reasoning was made available.

The **share and balance** numbers should not be reported as point estimates. That
C takes 39.7% is not a finding about C; it is substantially a finding about C's
description.

---

## Next

1. **The judge stage is now load-bearing, not an improvement.** Cosine cannot
   measure whether a way of reasoning was made available — it measures whether
   the vocabulary was. An LLM-as-judge rating each essay against each of the six
   discourse descriptions ("is this way of reasoning present, and is it
   articulated or dismissed?") answers the coverage question directly. Needs a
   fourth open-weight model to avoid self-preference, which is a decision outside
   the three CLAUDE.md fixes.

2. **Cross-check coverage against the factor arrays.** An alternative to the
   prose descriptions: score each essay's stance against the 45 statements and
   correlate the resulting profile with each discourse's z-column. That uses the
   quantitative instrument the discourses were actually derived from, rather than
   a narrative summary written afterwards, and it is not vulnerable to
   description length. Cheap to build — the stance matrices already exist.

3. **Social grounding** stays open. The current pipeline can only ask whether the
   human discourses are present; it has no way to detect a coherent LLM discourse
   with no human counterpart. That needs clustering the essays' own structure and
   then asking whether the clusters are socially meaningful — the meta-consensus
   question.

4. **Anaphora: done, and it argues for a larger unit.** The context pass ran and
   is analysed in `context_robustness_depersonalised.md`. The main results are
   robust to it (3.7% of verdicts change, TVD moves 0.001), but the pass itself
   is invalid as a scoring method — the two-sentence premise tracks the
   *context* sentence's own score (r = 0.79) more closely than the target
   sentence's (r = 0.64), so it double-counts the predecessor. The useful
   by-product: part of the assertion deficit is sentence fragmentation, not just
   the attitude-report problem, which is a further argument for measuring
   voicing above the sentence.

5. **Not yet run:** the second prompt condition (`brainstorm_generic`).
