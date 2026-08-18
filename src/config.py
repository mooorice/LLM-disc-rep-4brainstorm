"""
Central configuration for the discursive-representation experiment.

Every tunable choice in the pipeline lives here so that a single file records the
exact configuration a given set of results was produced under. The scripts
(generate -> segment -> embed -> analyse) import from here and never hard-code
paths or parameters of their own.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Project root, i.e. the directory containing src/, data/, results/.
ROOT = Path(__file__).resolve().parent.parent

PROMPT_DIR = ROOT / "prompts"
BASELINE_DIR = ROOT / "data" / "human_baseline"

# Raw LLM output: one JSON file per generated essay.
GENERATED_DIR = ROOT / "data" / "generated"

# Intermediate artefacts: the paragraph table and the embedding matrices.
PROCESSED_DIR = ROOT / "data" / "processed"

# Final analysis outputs.
RESULTS_DIR = ROOT / "results"

# --------------------------------------------------------------------------
# Which human discourse map is the benchmark
# --------------------------------------------------------------------------
#
# Two are available, and the choice is theoretical rather than technical.
#
# "post" (the primary benchmark) is the six-discourse map produced *after* the
# Australian Citizens' Jury deliberated. These are the ways of reasoning that
# emerged once people had worked through the issue together -- not a finer
# partition of the four, but a differentiated space that deliberation itself
# clarified. On the discursive-representation account being tested, that is the
# space a brainstorming tool should make available to participants.
#
# "pre" is the four-discourse map from the pre-deliberative mapping study. It
# describes raw opinion in the Australian population, which is the only place a
# prevalence claim can be grounded, so it is retained as the secondary
# comparison and as evidence about social grounding.
#
# See post-delib_baseline.NOTES.md for the full argument and for why the six
# have no population weights.
BASELINE = "post"

BASELINES = {
    "pre": {
        "label": "Pre-deliberative mapping study (four discourses)",
        # Cleaned narrative descriptions; see pre-delib_discourses_clean.CHANGES.md.
        "discourse_file": BASELINE_DIR / "pre-delib_discourses_clean.txt",
        # The 46 Q-statements with each discourse's z-score on each, in both the
        # original and depersonalised phrasing.
        "z_file": BASELINE_DIR / "pre-delib_q_statements_depersonalised.csv",
        "codes": ["A", "B", "C", "D"],
        # Discourse D ("Agnosticism") is excluded: no respondent in the
        # Australian population survey loaded on it at the reported precision,
        # so its expected share is exactly 0%. Scoring a model against a target
        # of "never voice this" is not a proportional-representation claim, and
        # it breaks every distance measure that needs a non-zero expectation.
        # Dropping it costs nothing on the target side -- A, B and C already
        # carry the whole expected distribution -- but it does change
        # assignment: a paragraph that would have gone to D is forced onto its
        # nearest surviving discourse. That trade is recorded in the summary.
        "active": ["A", "B", "C"],
        # Target distribution: observed prevalence in the Australian population.
        "target": "population",
        "weights_file": BASELINE_DIR / "pre-delib_discourse_weights_australia.csv",
    },
    "post": {
        "label": "Post-deliberative jury map (six discourses)",
        "discourse_file": BASELINE_DIR / "post-delib_discourses_clean.txt",
        # 45 statements (item 46 was dropped after the mapping study) x 6
        # discourses. Transcribed from OP12 Table 6 and validated against the
        # report's published correlation tables.
        "z_file": BASELINE_DIR / "post-delib_factor_array.csv",
        "codes": ["A", "B", "C", "D", "E", "F"],
        # All six stay in. None has a zero target, and the whole point of this
        # benchmark is whether the full space is made available.
        "active": ["A", "B", "C", "D", "E", "F"],
        # Target distribution: uniform. The brainstorming criterion is that no
        # relevant discourse is systematically marginalised, not that population
        # prevalence is reproduced. The one published distribution over the six
        # (OP12 Figure 10) is disclaimed by the report as provisional and is not
        # recoverable as marginals; see post-delib_baseline.NOTES.md.
        "target": "uniform",
        "weights_file": None,
    },
}

# Set from BASELINES at import and re-set by apply_overrides(). Kept as
# module-level names so every downstream script can read config.DISCOURSE_FILE
# without knowing which baseline is active.
DISCOURSE_FILE = BASELINES[BASELINE]["discourse_file"]
WEIGHTS_FILE = BASELINES[BASELINE]["weights_file"]

# Table 2: the 46 Q-statements participants sorted, with each discourse's
# z-score on each. Used by the stance pipeline, which scores agreement with the
# statements themselves rather than similarity to a prose summary of them.
FACTOR_ARRAY_FILE = BASELINE_DIR / "pre-delib_discourse_mapping_factor_array.csv"

# The same 46 statements with their attitude wrappers removed ("I'm concerned
# that X" -> "X"), holding both forms side by side. Built by
# scripts/build_depersonalised_statements.py; rationale in its CHANGES.md.
DEPERSONALISED_FILE = BASELINE_DIR / "pre-delib_q_statements_depersonalised.csv"

# Which phrasing the stance and similarity scorers use. Both are scored and
# reported: the original form is the instrument as administered, the
# depersonalised form is the one NLI can actually read.
STATEMENT_FORMS = ["original", "depersonalised"]

# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

# Open-weight models only, as specified in CLAUDE.md.
MODELS = [
    "deepseek/deepseek-v4-pro-0813",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.2",
]

# Independent essays generated per model.
N_REPETITIONS = 10

# Which prompt file in prompts/ defines the brainstorming task.
#
# NOTE: this is the central experimental instrument and the choice matters.
# "australian" names the deliberating public as Australian, matching the
# population the baseline weights describe; "generic" names no country. Results
# are only strictly comparable to the Australian weights under the former, but
# the latter is the more general claim about LLMs as brainstorming interfaces.
#
# Both are run as separate conditions. PROMPT_NAME is the default for a single
# invocation; every script also takes --prompt to override it, and
# scripts/run_experiment.sh walks PROMPTS in turn.
PROMPT_NAME = "brainstorm_australian"
PROMPTS = ["brainstorm_australian", "brainstorm_generic"]

# Sampling temperature. Kept at 1.0 so that the ten repetitions per model
# reflect the model's own output distribution rather than a sharpened version
# of it -- the spread across repetitions is part of what we are measuring.
TEMPERATURE = 1.0

# Enable reasoning tokens (see CLAUDE.md). Reasoning traces are stored alongside
# each essay but are NOT part of the analysed text: we measure what the model
# presents to the user, not what it considered privately.
ENABLE_REASONING = True

# Parallel API calls. Kept modest to stay well inside OpenRouter rate limits.
MAX_WORKERS = 4

# Retries per call, with exponential backoff between attempts.
MAX_RETRIES = 4

# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

# Paragraphs shorter than this are dropped before embedding. Very short
# fragments (headings that survived cleaning, one-line transitions, list stubs)
# carry too little content for a stable embedding and would add noise to the
# proportions. Dropped fragments are logged so the loss is auditable.
MIN_PARAGRAPH_WORDS = 25

# Sentences shorter than this are not scored. Below roughly this length a
# fragment carries no proposition for an NLI model to judge, but the model will
# still return a confident-looking probability for it.
MIN_SENTENCE_WORDS = 5

# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

EMBEDDING_MODEL = "infgrad/Jasper-Token-Compression-600M"

# Token compression ratio. The model card recommends 0.3-0.8; 0.3333 is the
# value used in the reference snippet in CLAUDE.md.
COMPRESSION_RATIO = 0.3333

# Jasper exposes an asymmetric "query" prompt intended for retrieval. Our
# comparison is symmetric -- a paragraph and a discourse description are two
# pieces of text whose similarity we want -- so both sides are encoded the same
# way, without the query prompt. Set to True to encode paragraphs as queries
# and discourse descriptions as documents instead.
USE_ASYMMETRIC_PROMPTS = False

EMBEDDING_BATCH_SIZE = 8

# --------------------------------------------------------------------------
# Stance scoring (NLI)
# --------------------------------------------------------------------------

# Multi-dataset NLI checkpoint (MNLI + FEVER + ANLI + LingNLI + WANLI). Chosen
# over a plain MNLI model for its stronger zero-shot behaviour on opinionated
# hypotheses, and because it has precedent in social-science text-as-data work.
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

NLI_BATCH_SIZE = 64

# A sentence counts as voicing or dismissing a discourse only if its alignment
# with that discourse's factor array exceeds this magnitude. Alignment is a sum
# of (stance x z-score) over 46 statements, so it is not bounded at 1; the
# threshold is set low because the NLI stage abstains on most pairs and the
# signal that survives is sparse.
STANCE_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# LLM-as-judge
# --------------------------------------------------------------------------

# The judges, per CLAUDE.md. Deliberately none of the three models under test,
# so that no model rates its own output: self-preference is well documented and
# would bias precisely the between-model comparison the experiment is for.
#
# Two of them, because one judge cannot tell you whether a flat result is a
# property of the corpus or of the rater. The first run returned `neutral` for
# every applicable treatment judgement -- a constant, which is not a
# measurement -- and the only way to find out whether that is the briefings or
# the model is to ask a different model the same question.
JUDGE_MODELS = [
    "google/gemma-4-31b-it",
    "openai/gpt-oss-120b",
]

# The default when no --judge-model is given.
JUDGE_MODEL = JUDGE_MODELS[0]

# Each (essay, discourse) pair is judged this many times and the modal label is
# taken. Repetition does double duty: majority voting is more stable than a
# single call on a subjective labelling task, and the spread across replicates
# is a reliability estimate for the instrument -- the same discipline applied to
# every other measure here.
JUDGE_REPLICATES = 3

# Non-zero, so the replicates are actually independent draws. At temperature 0
# they would be near-identical and the agreement figure would be meaningless.
JUDGE_TEMPERATURE = 1.0

# The judge returns short structured JSON, but reasoning models spend hundreds
# of tokens thinking first and that counts against the same budget. At 600,
# gpt-oss-120b burned the whole allowance on reasoning and returned empty
# content -- which the parser then read as "absent", silently turning a
# truncation into a finding. Set well above what any judgement needs.
JUDGE_MAX_TOKENS = 3000

# Judging is 30 essays x n discourses x JUDGE_REPLICATES calls, so it runs wider
# than generation does.
JUDGE_MAX_WORKERS = 8

# How developed a discourse must be in an essay to count as available. The judge
# rates presence on an ordered scale; this is where the line is drawn for the
# coverage statistic. "mentioned" is a passing reference, "articulated" means a
# reader could understand why someone reasons that way -- which is what
# "making a perspective available for deliberation" has to mean.
JUDGE_PRESENCE_LEVELS = ["absent", "mentioned", "articulated"]
JUDGE_PRESENCE_MIN = "articulated"

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

# Which discourses take part in the analysis. Set from the active baseline; see
# the BASELINES table above for why each one excludes what it excludes. Every
# discourse in the file is still embedded, so restoring one is a config change
# with no re-embedding.
ACTIVE_DISCOURSES = BASELINES[BASELINE]["active"]

# --------------------------------------------------------------------------
# Brainstorming criteria (post-deliberative benchmark)
# --------------------------------------------------------------------------
#
# Against the six post-deliberative discourses the question is not
# proportionality but availability: does the tool make each relevant way of
# reasoning present, consistently, and with enough space to be engaged with.
#
# A discourse counts as PRESENT in an essay when at least this share of the
# essay's sentences engage it and are not dismissed. A single stray sentence is
# not "making a perspective available for deliberation"; the threshold is what
# separates recovery from noise.
#
# NOTE: with essays running 42-69 sentences, 2% is 0.8-1.4 sentences, so this
# rule never binds -- PRESENCE_MIN_SENTENCES decides every case. It is kept
# because it would bind on a substantially longer corpus, but for the current
# data the absolute floor is the whole rule, and the sensitivity table in the
# report is the thing to read.
PRESENCE_MIN_SHARE = 0.02

# ...and at least this many sentences in absolute terms, so that a short essay
# cannot clear the share threshold on one sentence alone.
PRESENCE_MIN_SENTENCES = 2

# Presence is decided on small counts -- the median essay gives A, E and F only
# two or three non-dismissed sentences each -- so the choice of floor above
# moves the headline substantially. Coverage is recomputed at each of these to
# show how much of the result is the threshold rather than the essays.
PRESENCE_SENSITIVITY = [1, 2, 3, 4, 5]

# Draws used for the null model that asks how much of the coverage shortfall is
# a mechanical consequence of the pooled distribution rather than a property of
# individual essays.
PRESENCE_NULL_DRAWS = 5000

# A discourse is MARGINALISED when its share falls below this fraction of the
# uniform expectation (1/n). At 0.5 with six discourses that means under 8.3%:
# present, but with half the space an even split would give it.
MARGINALISATION_RATIO = 0.5

# A paragraph is assigned to its most similar discourse only if that discourse
# leads the runner-up by at least this margin in cosine similarity; otherwise it
# is recorded as "unassigned". This mirrors the human baseline, where 12% of
# respondents had a "confounded loading" across discourses rather than being
# forced onto one.
#
# The primary analysis uses 0.0 (every paragraph is assigned). The sensitivity
# analysis re-runs the proportions at each of the other values.
# The margins are small on purpose. Because all four baselines describe the same
# topic, every paragraph sits at high cosine similarity to all of them (~0.7-0.8)
# and the winner typically leads the runner-up by only a few thousandths. A
# margin of 0.05 leaves nothing assigned at all.
ASSIGNMENT_MARGIN = 0.0
SENSITIVITY_MARGINS = [0.0, 0.002, 0.005, 0.01, 0.02]

# --------------------------------------------------------------------------
# Command-line overrides
# --------------------------------------------------------------------------


def add_prompt_argument(parser) -> None:
    """
    Give a script --prompt and --baseline flags, so every combination can be run
    without editing this file. Each stage reads the module-level names at call
    time, so apply_overrides() re-setting them here redirects the whole stage.
    """
    parser.add_argument(
        "--prompt", default=None, metavar="NAME",
        help=f"prompt file in prompts/ to run (default: {PROMPT_NAME}). "
             f"Available: {', '.join(PROMPTS)}",
    )
    parser.add_argument(
        "--baseline", default=None, choices=list(BASELINES),
        help=f"which human discourse map to score against (default: {BASELINE}). "
             "'post' = six post-deliberative discourses, 'pre' = four "
             "pre-deliberative mapping-study discourses.",
    )


def apply_overrides(args) -> str:
    """Apply parsed command-line overrides to this module. Returns the prompt name."""
    global PROMPT_NAME, BASELINE, DISCOURSE_FILE, WEIGHTS_FILE, ACTIVE_DISCOURSES

    if getattr(args, "prompt", None):
        PROMPT_NAME = args.prompt

    if getattr(args, "baseline", None):
        BASELINE = args.baseline
        settings = BASELINES[BASELINE]
        DISCOURSE_FILE = settings["discourse_file"]
        WEIGHTS_FILE = settings["weights_file"]
        ACTIVE_DISCOURSES = settings["active"]

    return PROMPT_NAME


def add_judge_argument(parser) -> None:
    """Give a script a --judge-model flag, so each judge can be run separately."""
    parser.add_argument(
        "--judge-model", default=None, metavar="MODEL",
        help=f"which model judges (default: {JUDGE_MODEL}). "
             f"Configured: {', '.join(JUDGE_MODELS)}",
    )


def apply_judge_override(args) -> str:
    """Apply --judge-model to this module. Returns the active judge model."""
    global JUDGE_MODEL
    chosen = getattr(args, "judge_model", None)
    if chosen:
        JUDGE_MODEL = chosen
    return JUDGE_MODEL


def judge_slug(model: str = None) -> str:
    """Filesystem-safe name for a judge model, used to namespace its outputs."""
    return (model or JUDGE_MODEL).replace("/", "__")


def baseline() -> dict:
    """Settings for the currently selected discourse map."""
    return BASELINES[BASELINE]


def results_dir(prompt_name: str = None):
    """
    Where a run's outputs go: results/<prompt>/<baseline>/.

    Namespaced by baseline so that scoring the same essays against the four
    pre-deliberative and the six post-deliberative discourses produces two sets
    of results rather than one overwriting the other.
    """
    return RESULTS_DIR / (prompt_name or PROMPT_NAME) / BASELINE


def discourse_embeddings_path():
    """
    Where the encoded discourse descriptions live.

    Outside the per-prompt directory, because they depend only on the baseline
    file and the encoder settings -- but suffixed by baseline, so the four and
    the six do not overwrite each other.
    """
    return PROCESSED_DIR / f"discourse_embeddings_{BASELINE}.npy"


def discourse_table_path():
    """Companion to discourse_embeddings_path(): the codes and names, in order."""
    return PROCESSED_DIR / f"discourses_{BASELINE}.csv"
