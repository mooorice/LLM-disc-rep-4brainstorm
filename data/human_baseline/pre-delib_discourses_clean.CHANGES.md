# Cleaning log: `pre-delib_discourses_clean.txt`

Source: `pre-delib_discourses.txt`, itself a transcription of Appendix C.4 ("Mapping
Study (Four) Discourses") of *Genome Editing: Formulating an Australian Community
Response*, Occasional Paper No 12, Centre for Law and Genetics, 2022
(`background/OP12-final-report.pdf`, pp. 135–137).

Every sentence in the source was checked against the PDF before editing. The policy
was **delete or repair, never paraphrase**: no sentence was rewritten to say something
the report did not say, so the cleaned file adds no wording of ours to the baseline.

## 1. Deleted sentences (report mechanics, not discourse content)

**Discourse C** — trailing sentence about jury recruitment, which belongs to the
report's narrative flow and describes nothing about the discourse:

> The results of the mapping study helped to ensure diversity in the selection of
> Australian Citizens' Jury participants, as described in the next section.

**Discourse D** — two sentences of methodological commentary. They also mislabel the
discourse ("the third discourse", where D is listed fourth) and carry the only
variance figure in the document, which is preserved in §4 below:

> The third discourse accounts for just 5% of the study variance, but this doesn't
> mean it should be dismissed. It still means that this viewpoint exists in the wider
> community and we can't say anything about its wider prevalence it is at this stage.

Discourses A and B had no such sentences.

## 2. Substantive correction (one)

**Discourse B**, third paragraph. The source reads:

> They do not believe that parents should **not** have the right to edit their
> children's genes before they are born.

This double negative is verbatim in the published PDF, so it is the report's own
error rather than a transcription slip. As written it says B *endorses* a parental
right to edit — which contradicts both the surrounding paragraph (B "feel strongly
that genome editing should not be used for non-medical purposes") and the factor
array, where item 3 ("Parents and guardians have a right to edit the genes of their
children before they are born") scores **B = −1.84**, B's second-strongest
disagreement of all 46 items. The stray "not" was dropped:

> They do not believe that parents should have the right to edit their children's
> genes before they are born.

This is the only edit that changes meaning. To reproduce the report verbatim instead,
restore the "not" — the rest of the file is unaffected.

## 3. Typographic repairs (no change in meaning)

| Discourse | Source | Cleaned |
|---|---|---|
| B | "gene editing being used as quick fix" | "as a quick fix" |
| B | "when you change someone's genes you change you who they are" | "…you change who they are" |
| C | "we should remain as natural possible" | "as natural as possible" |
| D | "seems to be relatively unconcern about genome editing" | "relatively unconcerned about" |
| D | "compared to other viewpoints - unconvinced" | em dash, matching the file's other dashes |

Also applied throughout: hard line wraps from the PDF were unwrapped so that each
paragraph is a single line, and curly apostrophes/quotes were normalised to straight
ones. The `###` separators and `Discourse X: Name` headers are unchanged, so the file
parses exactly like the original.

## 4. Known issues left in place

- **Discourse D's closing sentence** refers to "ethical sociologists", a label that
  appears nowhere else in the four-discourse map. From context it means Discourse B
  (Principled Concern). Left as-is because repairing it would mean writing our own
  words into the baseline.
- **Q-methodology framing** ("Participants loading on this discourse…") is retained in
  all four descriptions. It is method vocabulary rather than discourse content, but it
  occurs in every description, so it shifts all four baselines alike. If it proves to
  drag similarity scores toward a common centre, a further de-jargoned variant is the
  next thing to try.
- **The 5% variance figure** for Discourse D is the *only* variance number the report
  gives for the mapping study; see the note below.

## 5. On discourse weights (for the proportionality measure)

Searched both PDFs. There is **no table of explained variance or of participant
loadings per discourse** for the four-discourse mapping study — no eigenvalues, no
percentage breakdown, no counts of participants loading on each factor. What exists:

- The single sentence quoted in §1, "The third discourse accounts for just 5% of the
  study variance" (OP12 p. 137), inside the Discourse D narrative but calling D "the
  third" discourse. Ambiguous, and unusable on its own.
- Method (Appendix C.3.4): varimax rotation of the four largest components from a PCA
  (inverted, Q-methodology). So per-factor variance was computed — it just was not
  reported.
- Sample: the mapping study Q-sorts were **n = 31** (Stage 2 of the project flow
  diagram, OP12 p. 11; the n = 123 alongside it is the later recruitment screening
  survey).
- Table 3 (OP12 p. 134) gives inter-discourse correlations: A–B 0.32, A–C 0.22,
  A–D 0.36, B–C 0.48, B–D 0.28, C–D 0.34.

So weights would have to come from elsewhere — the authors, the 2022 JLM article, or
an equal-weights assumption stated explicitly as a limitation.
