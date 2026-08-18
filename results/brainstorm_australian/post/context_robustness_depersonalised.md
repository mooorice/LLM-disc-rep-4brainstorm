# Anaphora robustness: does resolving references change the result?

Baseline: **Post-deliberative jury map (six discourses)**  |  Prompt: `brainstorm_australian`  |  Statement form: `depersonalised`

230 of 1610 sentences (14.3%) were re-scored with their preceding sentence prepended, because they open with a bare demonstrative or pronoun. 8 further flagged sentences were left alone, having no predecessor inside their paragraph.

## 1. Control — were the untouched sentences left alone?

These sentences were handed identical premises in both runs, so any difference is numerical rather than substantive. The size of it bounds how much of everything below can be believed.

| | Max change | Mean change | Pairs moving > 0.1 |
|---|---|---|---|
| Untouched (1380 sentences) | 0.0141 | 3.09e-05 | 0.0000% |
| Extended (230 sentences) | 1.6776 | 0.0961 | 17.55% |

The untouched row is not exactly zero because batching changes when premise lengths change, and the model runs in half precision. The magnitude is around one part in ten thousand on average, which is far below anything the analysis resolves.

## 2. What resolving the reference actually recovered

The stated motivation for this pass was denial: "But this is unfounded" is a dismissal that cannot be scored without its antecedent. That is not mainly what happened.

| Signal | Extended sentences, before | after | Whole corpus, before | after |
|---|---|---|---|---|
| asserting (> +0.5) | 0.78% | 3.75% | 0.92% | 1.35% |
| denying (< −0.5) | 7.81% | 7.52% | 7.84% | 7.80% |
| silent (abs < 0.1) | 85.90% | 82.42% | 85.09% | 84.60% |

On the extended sentences, detected **assertion rose from 0.78% to 3.75%** (x4.8), while detected denial went from 7.81% to 7.52%.

That is the opposite of the expectation. Dismissals turn out not to have needed the antecedent: a contradiction marker like "unfounded" or "overstated" sits inside the sentence that carries it. What the extra context changed was assertion — which would be good news, since assertion is the weak half of the NLI instrument, if the extra assertion belonged to the sentence being scored. Section 3 tests whether it does.

## 3. Does the recovered signal belong to the sentence?

A two-sentence premise entails whatever *either* sentence entails, and the NLI model has no way of knowing that only the second one is being measured. The prepended context is itself a scored sentence elsewhere in the corpus, so its solo score can be looked up and the three compared. If the combined score tracks the context more closely than the sentence, the pass is measuring the predecessor.

Matched 230 of 230 extended sentences to their context's own score.

| Combined (context + sentence) score compared against | Correlation | Mean absolute gap |
|---|---|---|
| The sentence scored alone | 0.635 | 0.0961 |
| The **context** scored alone | **0.787** | **0.0694** |

The combined premise tracks the context more closely than the sentence it was supposed to disambiguate. Of the 324 (sentence, statement) pairs that newly register as assertion, **43.5% were already assertions of the context sentence on its own** — and that context sentence is separately scored in its own right, so its stance is being counted twice.

A further 27.8% arise where the context was silent, so they are genuinely emergent from the combination. Some of those will be correct resolutions. The examples in section 5 suggest others are spurious entailments produced by giving the model more text to find a connection in.

## 4. Does anything downstream move?

### Dismissal rate per discourse

| Discourse | Before | After | Change (pp) |
|---|---|---|---|
| A Beneficial Scientific Progress | 31.5% | 32.9% | +1.4 |
| B Social Benefits/Precautionary Risks | 11.8% | 11.8% | +0.0 |
| C Principled Constraints | 2.1% | 2.7% | +0.7 |
| D Revolutionary Medicine | 1.0% | 1.3% | +0.3 |
| E Profound Social Risks | 30.2% | 31.7% | +1.6 |
| F Libertarian Revolutionary Medicine | 35.8% | 35.8% | +0.0 |

### Representation shares (engaged and not dismissed)

| Discourse | Before | After | Change (pp) |
|---|---|---|---|
| A | 6.8% | 6.7% | -0.1 |
| B | 15.6% | 15.7% | +0.1 |
| C | 39.7% | 39.6% | -0.0 |
| D | 27.0% | 27.1% | +0.1 |
| E | 6.1% | 6.0% | -0.1 |
| F | 4.9% | 4.9% | +0.0 |

**TVD from reference: 0.333 before, 0.334 after.**  Evenness 0.843 → 0.842.

### Coverage and availability

| | Before | After |
|---|---|---|
| Mean coverage | 5.17 / 6 | 5.20 / 6 |
| Essays at full coverage | 8/30 | 10/30 |

| Discourse | Essays available, before | after |
|---|---|---|
| A | 24/30 | 23/30 |
| B | 30/30 | 30/30 |
| C | 30/30 | 30/30 |
| D | 30/30 | 30/30 |
| E | 24/30 | 24/30 |
| F | 17/30 | 19/30 |

**60 of 1610 sentences (3.73%) change verdict**, of which 60 are among the sentences that were actually rewritten.

## 5. The sentences that moved most

Ordered by the largest single change against any statement. These are the cases the pass exists for; reading them is the check on whether the prepended context was the right context.

**glm-5.2 run z-ai__glm-5.2__rep05** — largest shift +1.678 on statement 3

> *context:* These changes are heritable, meaning they will be passed down to all future generations.

> *sentence:* Those who urge caution or outright prohibition of germline editing point to the unknown and potentially irreversible consequences.

> *statement:* Parents and guardians have a right to edit the genes of their children before they are born.

**kimi-k3 run moonshotai__kimi-k3__rep07** — largest shift +1.251 on statement 38

> *context:* Defenders of permissive approaches respond that this is an argument for public funding, not prohibition: if editing prevents lifelong illness, Medicare coverage could make it an instrument of fairness rather than division.

> *sentence:* They also note that Australia already tolerates large disparities in access to reproductive technologies, and ask why editing should be singled out.

> *statement:* It is acceptable for a person to have their own genome edited.

**deepseek-v4-pro-0813 run deepseek__deepseek-v4-pro-0813__rep07** — largest shift +1.250 on statement 38

> *context:* Many people support editing to treat or prevent disease but oppose using it to select or enhance traits such as height, intelligence, strength or appearance.

> *sentence:* They worry about designer babies and a slippery slope from medical need to parental preference.

> *statement:* It is acceptable for a person to have their own genome edited.

**deepseek-v4-pro-0813 run deepseek__deepseek-v4-pro-0813__rep02** — largest shift +1.217 on statement 38

> *context:* Many people are comfortable with editing to treat or prevent serious medical conditions but oppose using it to select or improve traits such as intelligence, height, appearance or athletic ability.

> *sentence:* They fear a slippery slope from medicine to designer babies, and a future in which parents feel pressured to genetically optimise their children.

> *statement:* It is acceptable for a person to have their own genome edited.

**kimi-k3 run moonshotai__kimi-k3__rep06** — largest shift +1.079 on statement 3

> *context:* Their caution concentrates on heritable editing, where mistakes would be permanent and transmissible.

> *sentence:* They point to the risk of unintended edits elsewhere in the genome, to embryos in which only some cells are corrected, to single genes that have multiple effects we do not fully understand, and to the simple fact that no experiment can reveal consequences appearing two generations later.

> *statement:* Parents and guardians have a right to edit the genes of their children before they are born.

**deepseek-v4-pro-0813 run deepseek__deepseek-v4-pro-0813__rep08** — largest shift +0.999 on statement 21

> *context:* Many disabled people and advocates warn that editing out genes associated with disability can send a powerful message that disabled lives are less worth living.

> *sentence:* They argue that society often exaggerates the suffering caused by disability while underestimating the barriers created by discrimination, inaccessible environments and lack of support.

> *statement:* Cultural beliefs are a reason to be cautious about genome editing.

**glm-5.2 run z-ai__glm-5.2__rep06** — largest shift -0.996 on statement 39

> *context:* Because germline modifications alter the DNA of future generations who cannot consent to the procedure, critics argue that we are treating unborn people as subjects of an irreversible experiment.

> *sentence:* There are significant worries about the long-term safety of these edits, including the possibility of off-target effects where the genetic scissors inadvertently alter the wrong parts of the genome, potentially causing new diseases or developmental disorders years or decades down the line.

> *statement:* Each individual has a right to decide for themselves whether to undergo gene editing.

**deepseek-v4-pro-0813 run deepseek__deepseek-v4-pro-0813__rep09** — largest shift +0.996 on statement 20

> *context:* Because heritable editing affects future generations, no one whose genes are changed can consent.

> *sentence:* This creates a unique ethical problem: we would be making irreversible choices on behalf of people who do not yet exist.

> *statement:* Editing genes that will be inherited is problematic because it means making decisions for future people who don't exist yet.

## Verdict

**The main report survives, and the context pass should not be adopted as the primary scoring.** Two separate conclusions.

*Robustness:* the pass changes 3.73% of verdicts and moves the headline TVD by 0.001. Every substantive claim in `sentence_summary_*.md` holds under both scorings, so anaphora is not quietly driving the result.

*Validity:* the pass does not do what it was built to do. It was meant to recover dismissals hiding behind unresolved references; dismissals turned out not to need it. What it recovered instead was assertion, and section 3 shows that assertion belongs substantially to the **prepended context rather than to the sentence being scored** — the combined premise correlates more strongly with the context's own solo score than with the sentence's. Since the context is separately scored as a sentence in its own right, adopting this pass would count a large part of that stance twice.

Concatenation is the wrong instrument for anaphora. An NLI model reads a two-sentence premise as one claim and will entail anything either half supports; there is no way to ask it about the second half only. The fix that would actually work is coreference resolution — rewriting the pronoun in place so the sentence stands alone at its original length — which keeps the premise about the sentence. Until then the base pass is the conservative choice and remains the primary scoring.

One thing worth keeping: the assertion deficit is not solely the attitude-report problem with the Q-statements. Sentence fragmentation contributes too, since a sentence cut off from its subject cannot entail anything. That is an argument for measuring voicing at a unit larger than the sentence, which is what a judge stage would do.
