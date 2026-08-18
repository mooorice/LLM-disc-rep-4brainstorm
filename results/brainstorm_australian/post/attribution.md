# Directness: whose voice carries each discourse

Baseline: **Post-deliberative jury map (six discourses)**

Prompt: `brainstorm_australian`  |  1610 sentences, 297 paragraphs

A discourse can be given airtime and left uncontradicted while still never being spoken in the essay's own voice. This report separates **direct** assertion from **attributed** reporting, and breaks the rate down by discourse.

## 1. How much of the corpus is reported speech

| Level | Attributed | Direct |
|---|---|---|
| Paragraphs, any marker | 66.3% | 33.7% |
| Paragraphs, named attribution only | 61.3% | 38.7% |
| Sentences | 25.4% | 74.6% |

The paragraph figures are higher than the sentence figure by construction: a paragraph counts as attributed if any part of it is, and paragraphs here average 5.4 sentences.

The narrow row is the one comparable to the **53.5%** recorded in `pre-delib_q_statements_depersonalised.CHANGES.md`, which came from a narrower ad-hoc pattern run before this script existed. The two are measuring the same thing at different sensitivities, so they should be in the same region rather than identical.

Breakdown of the sentence-level classification:

| Category | Sentences | Share | What it looks like |
|---|---|---|---|
| `named` | 259 | 16.1% | an attributor and a reporting verb — *critics argue that* |
| `framing` | 79 | 4.9% | ascription without a verb of saying — *on this view*, *for opponents* |
| `impersonal` | 18 | 1.1% | distanced but unascribed — *it is often argued*, *there are concerns* |
| `inherited` | 53 | 3.3% | continues an attribution from the previous sentence |
| `direct` | 1201 | 74.6% | asserted in the essay's own voice |

## 2. Directness by discourse

Of the sentences that engage each discourse's themes, the share put in somebody else's mouth. A high rate means the discourse reaches the reader as a report of what others think rather than as a claim about the world.

| Discourse | Sentences | Attributed | Direct | vs corpus (pp) |
|---|---|---|---|---|
| A Beneficial Scientific Progress | 143 | 28.0% | 72.0% | +2.6 |
| B Social Benefits/Precautionary Risks | 255 | 15.7% | 84.3% | -9.7 |
| C Principled Constraints | 584 | 29.1% | 70.9% | +3.7 |
| D Revolutionary Medicine | 393 | 26.2% | 73.8% | +0.8 |
| E Profound Social Risks | 126 | 12.7% | 87.3% | -12.7 |
| F Libertarian Revolutionary Medicine | 109 | 36.7% | 63.3% | +11.3 |

**Spread: 24.0 percentage points** between F (Libertarian Revolutionary Medicine, 36.7% attributed) and E (Profound Social Risks, 12.7%).

### By model

| Model | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4-pro-0813` | 29.3% | 20.6% | 29.5% | 24.2% | 12.0% | 38.7% |
| `moonshotai/kimi-k3` | 25.7% | 15.4% | 23.6% | 20.0% | 9.4% | 40.7% |
| `z-ai/glm-5.2` | 28.0% | 10.8% | 35.5% | 33.3% | 15.9% | 21.1% |

## 3. Two different ways of not letting a discourse speak

Attribution and dismissal are separate failures. A discourse can be reported at length and never contradicted, or asserted directly and then denied. Reading them together says more than either alone.

| Discourse | Attributed | Dismissed | Direct *and* not dismissed |
|---|---|---|---|
| A | 28.0% | 31.5% | 51.0% |
| B | 15.7% | 11.8% | 75.7% |
| C | 29.1% | 2.1% | 69.9% |
| D | 26.2% | 1.0% | 73.3% |
| E | 12.7% | 30.2% | 59.5% |
| F | 36.7% | 35.8% | 35.8% |

The last column is the strictest reading of representation available from the automated measures: the discourse's themes are engaged, the essay speaks in its own voice, and nothing contradicts it.

## 4. Samples, for checking the detector

A pattern-based classifier is exactly the kind of instrument that looks authoritative while measuring something adjacent to the intended target. These are drawn at random with a fixed seed; the question to ask of each is whether the label is right.

**`named`**

- Some participants may fear that editing will become a luxury service, reinforcing existing advantages in health, education and appearance.
- They also note that Australia already tolerates large disparities in access to reproductive technologies, and ask why editing should be singled out.
- Many Indigenous scholars therefore emphasise collective rather than purely individual consent, kinship, and sovereignty over genomic data, and caution that judgements about which traits are desirable have never been neutral.
- Others believe that a permanent prohibition is unnecessary if strict oversight and staged, transparent research can be developed.
- Many people with genetic conditions argue that a push to edit out certain genes can send a harmful message that their lives are less valuable, or that they should not exist.
- Proponents argue that human beings do not have the wisdom to safely and ethically manipulate the fundamental code of life, and that doing so exhibits dangerous hubris.

**`framing`**

- For these advocates, the push toward genome editing evokes historical traumas associated with eugenics.
- Critics of this view reply that medicine has always reshaped nature, from insulin to organ transplants; that dignity belongs to persons rather than to their DNA; and that refusing a safe intervention is as much a choice, with as much moral weight, as making one.
- For many, the human genome holds a sacred or natural status that should not be tampered with, viewing genetic manipulation as humanity overstepping its bounds.
- From this perspective, editing a gene to cure a devastating illness is simply a more precise extension of traditional medicine, much like developing a vaccine or performing surgery.
- From this viewpoint, proceeding without absolute certainty is an unacceptable gamble with the biological heritage of humanity.
- From this perspective, failing to use a safe and effective tool to prevent a child from being born with a painful, life-shortening condition is itself ethically questionable.

**`impersonal`**

- There is concern that widespread genetic editing could revive eugenic thinking, pressuring parents to select or engineer children who fit narrow norms of health and ability.
- There is also concern about genetic discrimination: if people are expected to edit their children’s genomes to avoid disease, those who cannot afford it, or who choose not to, may be blamed or penalised.
- There are concerns about new forms of social pressure, narrow standards of normalcy, and the loss of valuable human diversity.
- In this light, the technology is seen as a moral imperative: if we have the power to prevent a child from enduring a life of pain or a premature death, we arguably have a duty to use it.
- Beyond the social implications, there are significant concerns regarding safety and the limits of human knowledge.
- Others, including many disability advocates, distinguish sharply between respecting people who live with disability and reducing avoidable suffering, and they support somatic therapies while asking that their voices shape how these technologies are framed and governed.

**`inherited`**

- They ask why a couple who knows they carry a serious condition should be denied the chance to have a genetically related child who is free of that condition, particularly when other options such as prenatal testing and embryo selection may not work or may involve termination.
- They press what philosophers call the expressivist objection: systematically selecting against certain conditions sends a message that lives lived with them are worth less.
- This caution is often accompanied by a call for more basic research before any clinical use.
- It draws parallels to previous medical revolutions, such as vaccines or antibiotics, suggesting that genome editing is merely the next logical step in our quest to conquer disease and improve the human condition.
- This perspective highlights the economic benefits of biotechnological leadership and the importance of keeping Australian medical researchers at the forefront of global innovation.
- There is a deep fear that normalizing genome editing could lead to a modern form of eugenics, where society deems certain lives as less valuable or inherently flawed.

**`direct`**

- For germline editing, any mistake would not only affect the individual born from that embryo but could be inherited by their children and grandchildren.
- Some fear a "genetic divide" in which wealth purchases heritable advantage, converting social inequality into biology.
- Options for this assembly range from maintaining the prohibition, through moratoria with built-in review, to licensed pathways for defined serious conditions under strict conditions.
- Because eugenic ideas once influenced Australian policy, including policies that devastated First Nations families, state involvement in heredity carries a particular historical weight.
- Editing could, in principle, remove a disease-causing variant from a family line forever.
- Much of the public debate focuses on germline editing because its effects could permanently alter a family’s genetic inheritance and, over time, the broader human gene pool.

## Caveats

- The detector is lexical. It finds the *markers* of attribution, not attribution itself, and will miss a view ascribed by context alone.
- `direct` is a residual category, so every false negative lands in it. The direct rate is therefore an upper bound and the attributed rate a lower bound.
- Attribution scope is inherited only by sentences that open with an unresolved reference, and never across a paragraph boundary. Both choices are conservative, and both push the attributed rate down.
- Which discourse a sentence belongs to comes from the same cosine argmax used everywhere else, with the same weaknesses — see `src/validate_baseline.py`. The comparison between discourses is made under a constant bias, so the ordering is sturdier than the levels.
