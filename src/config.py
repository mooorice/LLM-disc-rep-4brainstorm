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

# The cleaned discourse descriptions that serve as the representation baseline,
# and the Australian population weights the observed proportions are judged against.
# See pre-delib_discourses_clean.CHANGES.md and
# pre-delib_discourse_weights_australia.NOTES.md for how both were derived.
DISCOURSE_FILE = BASELINE_DIR / "pre-delib_discourses_clean.txt"
WEIGHTS_FILE = BASELINE_DIR / "pre-delib_discourse_weights_australia.csv"

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
# Analysis
# --------------------------------------------------------------------------

# Which discourses take part in the analysis.
#
# Discourse D ("Agnosticism") is excluded. No respondent in the Australian
# population survey loaded on it at the reported precision, so its expected
# share is exactly 0%. Keeping it would mean scoring the models against a target
# of "never voice this", which is not a proportional-representation claim, and
# it would break every distance measure that needs a non-zero expectation.
#
# Dropping it costs nothing on the target side -- A, B and C already carry the
# whole of the expected distribution -- but it does change assignment: a
# paragraph that would have gone to D is now forced onto its nearest surviving
# discourse. That trade is recorded in the results summary.
#
# All four baselines are still embedded, so restoring D is a one-line change
# here with no re-embedding.
ACTIVE_DISCOURSES = ["A", "B", "C"]

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
    Give a script a --prompt flag, so both prompt conditions can be run without
    editing this file. Every stage reads config.PROMPT_NAME at call time, so
    apply_overrides() setting it here is enough to redirect the whole stage.
    """
    parser.add_argument(
        "--prompt", default=None, metavar="NAME",
        help=f"prompt file in prompts/ to run (default: {PROMPT_NAME}). "
             f"Available: {', '.join(PROMPTS)}",
    )


def apply_overrides(args) -> str:
    """Apply parsed command-line overrides to this module. Returns the prompt name."""
    global PROMPT_NAME
    if getattr(args, "prompt", None):
        PROMPT_NAME = args.prompt
    return PROMPT_NAME
