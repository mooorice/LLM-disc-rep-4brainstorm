"""
Step 5: figures for the paper.

Every number plotted here is recomputed from the result CSVs written by the
analysis steps, so the figures cannot drift away from the tables they
accompany. Nothing is hard-coded except the layout.

Three figures, matching the numbering in the results section:

  Figure 1  Discourse recovery by model and coding procedure.
            (a) heatmap: how often each generator makes each discourse
                available, under the distributional (semantic similarity)
                procedure at tau = 2.
            (b) bars: pooled recovery of each discourse under all three
                procedures -- distributional, and the two LLM coders.

  Figure 2  Reliability and threshold sensitivity.
            (a) spread across the three generators, per discourse.
            (b) recovery as a function of the presence threshold tau.

  Figure 4  Visibility and rhetorical treatment.
            (a) visibility against dismissal, point size = pooled recovery.
            (b) the three rhetorical-treatment indicators side by side.

  Figure 3  Descriptive profiles of differential treatment.
            (a) dendrogram over the standardised five-indicator profiles.
            (b) the profiles themselves, as parallel coordinates.

A NOTE ON THE TWO PRESENCE RULES
--------------------------------
The headline specification counts a discourse as available in a text when it
holds at least PRESENCE_MIN_SENTENCES sentences *and* at least
PRESENCE_MIN_SHARE of the text's scored sentences. The published sensitivity
curve varies the sentence count alone and drops the share condition. The two
rules coincide everywhere from tau = 2 upwards -- the share floor never binds
on these texts once two sentences are required -- and differ only at tau = 1.
Panel 2(b) reproduces the published table, so it uses the count-only rule and
says so in the axis label.

Usage:
    python src/figures.py
    python src/figures.py --form original
    python src/figures.py --outdir paper/figures --format png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # write files; never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram, linkage

import config

# --------------------------------------------------------------------------
# Shared styling
# --------------------------------------------------------------------------

# One colour per discourse, held constant across every figure so that a reader
# who learns the palette in Figure 1 can carry it to Figure 4. Chosen from
# Okabe-Ito, which stays distinguishable in the common forms of colour
# blindness and survives greyscale conversion reasonably well.
DISCOURSE_COLOURS = {
    "A": "#0072B2",   # blue
    "B": "#E69F00",   # orange
    "C": "#009E73",   # green
    "D": "#CC79A7",   # pink
    "E": "#56B4E9",   # light blue
    "F": "#D55E00",   # vermillion
}

# Discourses that sit at the same value for every threshold would otherwise
# hide one another completely in the line panel (C and D are both at 100 per
# cent throughout). Distinct dash patterns make overlapping lines read as
# interleaved segments rather than as a single line.
DISCOURSE_LINESTYLES = {
    "A": "solid",
    "B": (0, (5, 1)),
    "C": (0, (1, 1.2)),
    "D": (0, (4, 1, 1, 1)),
    "E": (0, (3, 1.4)),
    "F": "solid",
}

# Everything below is drawn from the same Okabe-Ito palette as the discourse
# colours above, so that the whole figure set reads as one scheme: neutral grey
# for the reference or baseline series, blue and vermillion for the two
# contrasting series, green for the "representation actually achieved" measure.
# Four roles, four greys, used everywhere instead of ad-hoc hex values so that
# the same kind of mark is the same colour in every figure.
NEUTRAL_GREY = "#7F7F7F"   # a data series acting as the baseline or reference
RULE_GREY = "#BBBBBB"      # reference lines, zero lines, legend size keys
LABEL_GREY = "#555555"     # secondary text: captions, in-plot annotations
EMPHASIS_INK = "#1A1A1A"   # the one summary series that must read above colour

# One colour per coding procedure, for the panels that compare instruments.
PROCEDURE_COLOURS = {
    "Semantic similarity": NEUTRAL_GREY,
    "Coder: gemma-4-31b-it": "#0072B2",
    "Coder: gpt-oss-120b": "#D55E00",
}

# One colour per rhetorical-treatment indicator, shared by Figures 3 and 4.
INDICATOR_COLOURS = {
    "attributed": NEUTRAL_GREY,
    "dismissed": "#D55E00",
    "direct_nd": "#009E73",
}

# Sequential ramp for the heatmap, built from the palette's blue so that the
# one continuous scale in the figure set belongs to the same scheme rather
# than importing an unrelated matplotlib colormap.
# The stops are placed unevenly on purpose. Recovery counts on these texts
# cluster hard at the top of the range (most cells are 8-10), so an evenly
# spaced ramp would render the whole informative band in near-identical dark
# blue. Pushing the mid-tones up the scale spends the visible colour range
# where the data actually varies.
RECOVERY_CMAP = LinearSegmentedColormap.from_list(
    "okabe_blue",
    [(0.00, "#F4F8FC"), (0.45, "#9EC9E4"), (0.80, "#0072B2"), (1.00, "#00456B")]
)

PLOT_STYLE = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
}


def short_model_name(model_id: str) -> str:
    """'deepseek/deepseek-v4-pro-0813' -> 'deepseek-v4-pro'."""
    name = model_id.split("/")[-1]
    # Drop a trailing date stamp such as '-0813', which carries no information
    # for a reader and costs a lot of axis width.
    parts = name.split("-")
    if parts[-1].isdigit():
        parts = parts[:-1]
    return "-".join(parts)


def model_slug(model_id: str) -> str:
    """'google/gemma-4-31b-it' -> 'google__gemma-4-31b-it' (the file naming)."""
    return model_id.replace("/", "__")


def load_discourse_names(path: Path) -> dict[str, str]:
    """
    Read code -> name from the cleaned discourse file.

    Same block format that embed.load_discourses parses ('Discourse A: Name'
    headers separated by '###'), reimplemented here in five lines so that
    plotting does not have to import the embedding stack.
    """
    names = {}
    for block in path.read_text(encoding="utf-8").split("###"):
        header = block.strip().partition("\n")[0].strip()
        if not header:
            continue
        code, _, name = header.removeprefix("Discourse").strip().partition(":")
        names[code.strip()] = name.strip()
    return names


# --------------------------------------------------------------------------
# Data loading -- everything is derived from the analysis outputs
# --------------------------------------------------------------------------

def load_distributional(results: Path, form: str, codes: list[str]) -> pd.DataFrame:
    """
    Per-text availability under the distributional procedure.

    Returns the raw per-essay coverage table, which carries for each discourse
    the non-dismissed sentence count (n_X), its share of the text's scored
    sentences (share_X) and the resulting presence flag (present_X).
    """
    path = results / f"essay_coverage_{form}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run src/analyse_sentences.py --form {form} first"
        )
    return pd.read_csv(path)


def recovery_by_model(coverage: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """Texts per generator in which each discourse is available (rows = models)."""
    rows = {}
    for model, group in coverage.groupby("model"):
        rows[short_model_name(model)] = {c: int(group[f"present_{c}"].sum())
                                         for c in codes}
    return pd.DataFrame(rows).T[codes]


def tau_sensitivity(coverage: pd.DataFrame, codes: list[str],
                    taus: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Recompute availability across presence thresholds.

    Uses the sentence count alone, matching the published sensitivity table --
    see the module docstring on the two presence rules.

    Returns (availability per discourse per tau, mean coverage per tau).
    """
    per_discourse, mean_coverage = {}, {}
    for tau in taus:
        present = pd.DataFrame({c: coverage[f"n_{c}"] >= tau for c in codes})
        per_discourse[tau] = present.sum()
        mean_coverage[tau] = present.sum(axis=1).mean()
    return pd.DataFrame(per_discourse).T[codes], pd.Series(mean_coverage)


def load_coder_recovery(results: Path, judge_models: list[str],
                        codes: list[str]) -> dict[str, pd.Series]:
    """
    Pooled availability per discourse for each LLM coder.

    A coder's verdict table holds one row per (generator, repetition,
    discourse) with `available` already collapsed across the three replicates
    by majority vote. Missing coder runs are skipped with a warning rather than
    failing, so the figures can still be built from a partial pipeline.
    """
    out = {}
    for judge in judge_models:
        path = results / f"judge_verdicts_{model_slug(judge)}.csv"
        if not path.exists():
            print(f"  ! no verdicts for {judge}; skipping in Figure 1(b)")
            continue
        verdicts = pd.read_csv(path)
        counts = (verdicts.groupby("discourse")["available"].sum()
                  .reindex(codes).fillna(0).astype(int))
        out[f"Coder: {short_model_name(judge)}"] = counts
    return out


def load_treatment(results: Path, codes: list[str], form: str) -> pd.DataFrame:
    """
    Per-discourse visibility and rhetorical treatment, from the sentence table.

    Four quantities, all computed over the sentences assigned to each discourse:
      visibility  -- share of all scored sentences that went to this discourse
      attributed  -- put in somebody else's mouth rather than asserted directly
      dismissed   -- raised and then denied
      direct_nd   -- asserted in the essay's own voice and not denied
                     (the strictest reading of representation available here)
    """
    path = results / "attribution.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run src/attribution.py first"
        )

    # src/attribution.py reads sentence_scores_depersonalised.csv unconditionally
    # -- it has no --form flag -- so the dismissal column below always reflects
    # the depersonalised statements. Attribution itself is form-independent (it
    # inspects the generated text, not the benchmark statements), but dismissal
    # is not: rebuttal rates differ substantially between the two forms. Say so
    # rather than let panels 1 and 2 silently disagree with panel 4.
    if form != "depersonalised":
        print(f"  ! Figure 4 uses depersonalised dismissal verdicts even though "
              f"--form {form} was requested: src/attribution.py is hard-wired to "
              f"sentence_scores_depersonalised.csv. Panels 1 and 2 honour --form.")

    sentences = pd.read_csv(path)
    total = len(sentences)

    rows = []
    for code in codes:
        block = sentences[sentences["discourse"] == code]
        dismissed = block["verdict"] == "dismissed"
        rows.append({
            "discourse": code,
            "sentences": len(block),
            "visibility": 100 * len(block) / total,
            "attributed": 100 * block["attributed"].mean(),
            "dismissed": 100 * dismissed.mean(),
            "direct_nd": 100 * (~block["attributed"] & ~dismissed).mean(),
        })
    return pd.DataFrame(rows).set_index("discourse")


# --------------------------------------------------------------------------
# Figure 1 -- recovery by model and by coding procedure
# --------------------------------------------------------------------------

def figure_1(by_model: pd.DataFrame, pooled: dict[str, pd.Series],
             codes: list[str], names: dict[str, str], n_per_model: int,
             n_texts: int) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1, 1.25]}
    )

    # -- (a) heatmap: generator x discourse ---------------------------------
    data = by_model[codes].to_numpy(dtype=float)
    im = ax_a.imshow(data, cmap=RECOVERY_CMAP, vmin=0, vmax=n_per_model,
                     aspect="auto")

    ax_a.set_xticks(range(len(codes)), codes)
    ax_a.set_yticks(range(len(by_model)), by_model.index)
    ax_a.set_xlabel("Discourse")
    ax_a.set_title("(a) Recovery by generator\n(semantic similarity, "
                   r"$\tau$ = 2)", loc="left")

    # Annotate every cell: the grid is small enough that exact counts are more
    # useful than the colour scale, which is left in as a fast visual sort.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = int(data[i, j])
            # White text on the dark end of the colormap, black on the light end.
            colour = "white" if value > 0.6 * n_per_model else EMPHASIS_INK
            ax_a.text(j, i, f"{value}", ha="center", va="center",
                      color=colour, fontsize=9)

    bar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    bar.set_label(f"texts out of {n_per_model}", fontsize=8)
    bar.ax.tick_params(labelsize=8)

    # -- (b) pooled recovery under all three procedures ---------------------
    procedures = {"Semantic similarity": by_model[codes].sum()} | pooled
    width = 0.8 / len(procedures)
    x = np.arange(len(codes))

    for offset, (label, counts) in enumerate(procedures.items()):
        ax_b.bar(x + offset * width - 0.4 + width / 2, counts[codes].to_numpy(),
                 width, label=label,
                 color=PROCEDURE_COLOURS.get(label, NEUTRAL_GREY))

    ax_b.set_xticks(x, codes)
    ax_b.set_xlabel("Discourse")
    ax_b.set_ylabel(f"texts out of {n_texts}")
    ax_b.set_ylim(0, n_texts * 1.08)
    ax_b.axhline(n_texts, color=RULE_GREY, linewidth=1.0, linestyle=":")
    ax_b.set_title(r"(b) Pooled recovery by coding procedure ($\tau$ = 2)",
                   loc="left")
    ax_b.legend(fontsize=8, loc="upper right", ncols=1)

    # Spell out the discourse names once, under both panels.
    fig.text(0.5, -0.11, "   ".join(f"{c} {names.get(c, '')}" for c in codes),
             ha="center", fontsize=7.5, color=LABEL_GREY)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Figure 2 -- between-model spread and threshold sensitivity
# --------------------------------------------------------------------------

def figure_2(by_model: pd.DataFrame, per_discourse: pd.DataFrame,
             mean_coverage: pd.Series, codes: list[str], n_per_model: int,
             n_texts: int) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 3.9))

    # -- (a) spread across generators ---------------------------------------
    # For each discourse: a vertical line spanning the min and max across the
    # three generators, with one marker per generator on top. This shows both
    # the level and how much of it is model-specific, which a mean would hide.
    markers = ["o", "s", "^"]
    # Where two generators land on the same count the markers would sit exactly
    # on top of each other and only the last drawn would be visible -- which
    # reads as missing data rather than as agreement. A small horizontal offset
    # separates them; the x-axis is categorical, so this is presentational only
    # and changes no value.
    offsets = np.linspace(-0.13, 0.13, len(by_model))

    for j, code in enumerate(codes):
        values = by_model[code].to_numpy(dtype=float)
        ax_a.vlines(j, values.min(), values.max(),
                    color=DISCOURSE_COLOURS[code], linewidth=7, alpha=0.25)
        for k, (model, value) in enumerate(zip(by_model.index, values)):
            ax_a.plot(j + offsets[k], value, markers[k % len(markers)],
                      markersize=6, color=DISCOURSE_COLOURS[code],
                      markeredgecolor="white", markeredgewidth=0.7,
                      label=model if j == 0 else None)

    ax_a.set_xticks(range(len(codes)), codes)
    ax_a.set_xlabel("Discourse")
    ax_a.set_ylabel(f"texts out of {n_per_model}")
    ax_a.set_ylim(-0.5, n_per_model + 0.5)
    ax_a.set_title("(a) Between-generator spread\n(semantic similarity, "
                   r"$\tau$ = 2)", loc="left")
    # The markers carry the model identity; the colours carry the discourse.
    legend = ax_a.legend(fontsize=8, loc="lower left", handletextpad=0.2)
    for handle in legend.legend_handles:
        handle.set_color(NEUTRAL_GREY)

    # -- (b) recovery as a function of tau ----------------------------------
    # Both series are put on a shared percentage axis: per-discourse
    # availability as a share of all texts, mean coverage as a share of the six
    # discourses. That keeps one y-axis instead of two.
    taus = per_discourse.index.to_numpy()
    for i, code in enumerate(codes):
        ax_b.plot(taus, 100 * per_discourse[code] / n_texts,
                  marker="o", markersize=4,
                  # Stagger which points carry a marker, so that lines lying on
                  # top of one another still show two distinct sets of markers.
                  markevery=(i % 2, 2),
                  linewidth=1.7, color=DISCOURSE_COLOURS[code],
                  linestyle=DISCOURSE_LINESTYLES.get(code, "solid"),
                  label=code)

    ax_b.plot(taus, 100 * mean_coverage / len(codes), marker="D", markersize=5,
              linewidth=2.2, linestyle="--", color=EMPHASIS_INK,
              label="mean coverage")

    ax_b.set_xticks(taus)
    ax_b.set_xlabel(r"presence threshold $\tau$ (min. sentences)")
    ax_b.set_ylabel("per cent")
    ax_b.set_ylim(0, 105)
    ax_b.set_title("(b) Sensitivity to the presence threshold", loc="left")
    ax_b.legend(fontsize=8, ncols=4, loc="lower left", columnspacing=1.0)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Figure 3 -- descriptive profiles of differential treatment
# --------------------------------------------------------------------------

# The five indicators that make up a discourse's profile, in the order they are
# plotted. Each is a percentage, but of different denominators and with very
# different spreads, which is why the profile is standardised before anything
# is compared or clustered.
PROFILE_INDICATORS = {
    "recovery": "recovery",
    "visibility": "visibility",
    "attributed": "attribution",
    "dismissed": "dismissal",
    "direct_nd": "direct, not\ndismissed",
}


def discourse_profiles(treatment: pd.DataFrame, pooled_recovery: pd.Series,
                       codes: list[str], n_texts: int
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble the five-indicator profile of each discourse, raw and standardised.

    Recovery is expressed as a percentage of texts so that it shares a scale
    type with the other four; standardisation then puts all five on a common
    footing, since otherwise visibility (range 6.8-36.3) would dominate any
    distance calculation over dismissal (range 1.0-35.8) purely by variance.

    Returns (raw percentages, z-scores across the six discourses).
    """
    raw = pd.DataFrame({
        "recovery": 100 * pooled_recovery[codes] / n_texts,
        "visibility": treatment.loc[codes, "visibility"],
        "attributed": treatment.loc[codes, "attributed"],
        "dismissed": treatment.loc[codes, "dismissed"],
        "direct_nd": treatment.loc[codes, "direct_nd"],
    })
    # Population standard deviation (ddof=0): these six discourses are the whole
    # set under study, not a sample drawn from a larger population of discourses.
    spread = raw.std(ddof=0)

    # An indicator on which every discourse scores the same cannot separate them.
    # Dividing by its zero spread would produce NaNs and abort the clustering;
    # substituting 1 leaves the column at a constant zero, which contributes
    # nothing to any pairwise distance. That is the right answer rather than a
    # patch -- but it is a real loss of information, so say so. This bites on the
    # pre-deliberative map, where the distributional procedure recovers all
    # three discourses in all 30 texts.
    flat = spread[spread == 0].index.tolist()
    if flat:
        print(f"  ! profile indicator(s) {flat} are constant across discourses "
              f"and cannot contribute to Figure 3; plotted as zero")
        spread = spread.replace(0.0, 1.0)

    standardised = (raw - raw.mean()) / spread
    return raw, standardised


def figure_3(standardised: pd.DataFrame, codes: list[str],
             names: dict[str, str]) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1, 1.35]}
    )

    # -- (a) dendrogram ------------------------------------------------------
    # Average linkage on Euclidean distances between standardised profiles.
    # With six objects this is a descriptive summary of which discourses look
    # alike across the five indicators, not a test for latent classes -- so the
    # links are drawn in a single neutral grey and no cut height is marked.
    # Colouring branches by cluster would assert exactly the structure the
    # caption disclaims.
    linkage_matrix = linkage(standardised.to_numpy(), method="average",
                             metric="euclidean")
    dendrogram(linkage_matrix, labels=list(codes), ax=ax_a,
               color_threshold=0, above_threshold_color=NEUTRAL_GREY,
               leaf_font_size=10)

    # The identity of each leaf is carried by the shared discourse palette.
    for label in ax_a.get_xmajorticklabels():
        label.set_color(DISCOURSE_COLOURS[label.get_text()])
        label.set_fontweight("bold")

    ax_a.set_ylabel("distance between standardised profiles")
    ax_a.set_title("(a) Similarity of discourse profiles\n"
                   "average linkage, descriptive only", loc="left")
    ax_a.spines["bottom"].set_visible(False)
    ax_a.tick_params(axis="x", length=0)

    # -- (b) profile display -------------------------------------------------
    # Parallel coordinates: one line per discourse across the five indicators.
    # This is what the dendrogram is summarising, shown directly, so a reader
    # can see *why* two discourses were joined rather than taking it on trust.
    x = np.arange(len(PROFILE_INDICATORS))
    for code in codes:
        ax_b.plot(x, standardised.loc[code, list(PROFILE_INDICATORS)],
                  marker="o", markersize=5, linewidth=1.8,
                  color=DISCOURSE_COLOURS[code],
                  linestyle=DISCOURSE_LINESTYLES.get(code, "solid"),
                  label=f"{code} {names.get(code, '')}")

    ax_b.axhline(0, color=RULE_GREY, linewidth=0.9, linestyle=":", zorder=0)
    ax_b.set_xticks(x, list(PROFILE_INDICATORS.values()))
    ax_b.set_xlim(-0.25, len(x) - 0.75)
    ax_b.set_ylabel("standardised score\n(z across the six discourses)")
    ax_b.set_title("(b) Indicator profiles\n"
                   "above 0 = higher than the six-discourse mean", loc="left")
    ax_b.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Figure 4 -- visibility against treatment
# --------------------------------------------------------------------------

def figure_4(treatment: pd.DataFrame, pooled_recovery: pd.Series,
             codes: list[str], n_texts: int) -> plt.Figure:
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1, 1.2]}
    )

    # -- (a) visibility against dismissal -----------------------------------
    # Point area is scaled to pooled recovery so that the third variable reads
    # as "how dependable is this discourse" without a third axis. The scaling
    # is linear in area from a visible floor, not in radius, so that the eye
    # compares areas correctly.
    recovery = pooled_recovery[codes].to_numpy(dtype=float)
    sizes = 40 + 360 * (recovery / n_texts) ** 2

    for code, size in zip(codes, sizes):
        ax_a.scatter(treatment.loc[code, "visibility"],
                     treatment.loc[code, "dismissed"],
                     s=size, color=DISCOURSE_COLOURS[code],
                     alpha=0.75, edgecolor="white", linewidth=1.2, zorder=3)
        ax_a.annotate(code,
                      (treatment.loc[code, "visibility"],
                       treatment.loc[code, "dismissed"]),
                      textcoords="offset points", xytext=(0, -3),
                      ha="center", va="center", fontsize=8,
                      fontweight="bold", color="white", zorder=4)

    # Even-visibility reference. Descriptive only: the paper does not treat an
    # equal split as the normative target.
    even = 100 / len(codes)
    ax_a.axvline(even, color=RULE_GREY, linewidth=0.9, linestyle=":", zorder=1)
    ax_a.text(even + 0.6, ax_a.get_ylim()[1] * 0.96, "even visibility",
              fontsize=7.5, color=LABEL_GREY, va="top")

    ax_a.set_xlabel("visibility (% of scored sentences)")
    ax_a.set_ylabel("dismissed (% of that discourse's sentences)")
    ax_a.set_title("(a) Visibility against rebuttal\n"
                   "point size = pooled recovery", loc="left")

    # Size legend: two reference points are enough to read the encoding.
    for reference in (n_texts, int(round(0.5 * n_texts))):
        ax_a.scatter([], [], s=40 + 360 * (reference / n_texts) ** 2,
                     color=RULE_GREY, edgecolor="white",
                     label=f"{reference}/{n_texts} texts")
    ax_a.legend(fontsize=7.5, loc="upper right", labelspacing=1.1,
                borderpad=0.8)

    # -- (b) the three treatment indicators ---------------------------------
    # Grouped, never stacked: a sentence can be both attributed and dismissed,
    # so the three bars do not partition anything and must not look as if
    # they do.
    indicators = {
        "attributed to third parties": ("attributed", INDICATOR_COLOURS["attributed"]),
        "dismissed / rebutted": ("dismissed", INDICATOR_COLOURS["dismissed"]),
        "direct and not dismissed": ("direct_nd", INDICATOR_COLOURS["direct_nd"]),
    }
    width = 0.8 / len(indicators)
    x = np.arange(len(codes))

    for offset, (label, (column, colour)) in enumerate(indicators.items()):
        ax_b.bar(x + offset * width - 0.4 + width / 2,
                 treatment.loc[codes, column].to_numpy(),
                 width, label=label, color=colour)

    ax_b.set_xticks(x, codes)
    ax_b.set_xlabel("Discourse")
    ax_b.set_ylabel("per cent of the discourse's sentences")
    ax_b.set_ylim(0, 100)
    ax_b.set_title("(b) Rhetorical treatment\n"
                   "categories overlap; bars are not stacked", loc="left")
    ax_b.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------

def save(fig: plt.Figure, outdir: Path, stem: str, formats: list[str]) -> None:
    for suffix in formats:
        path = outdir / f"{stem}.{suffix}"
        fig.savefig(path)
        print(f"  wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=config.PROMPT_NAME,
                        help="prompt condition to plot (default: %(default)s)")
    parser.add_argument("--baseline", choices=list(config.BASELINES),
                        default="post",
                        help="discourse map to plot (default: %(default)s)")
    parser.add_argument("--form", choices=config.STATEMENT_FORMS,
                        default="depersonalised",
                        help="statement form (default: %(default)s)")
    parser.add_argument("--outdir", type=Path,
                        default=config.ROOT / "paper" / "figures",
                        help="where to write (default: %(default)s)")
    parser.add_argument("--format", nargs="+", default=["pdf", "png"],
                        help="output formats (default: pdf png)")
    args = parser.parse_args()

    # Route the --prompt/--baseline flags through config, exactly as the
    # analysis steps do, so the active discourse set comes from one place.
    config.apply_overrides(args)
    codes = config.ACTIVE_DISCOURSES
    names = load_discourse_names(config.baseline()["discourse_file"])

    results = config.RESULTS_DIR / args.prompt / args.baseline
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Baseline: {config.baseline()['label']}")
    print(f"Form:     {args.form}")
    print(f"Reading:  {results}")

    # --- load ---------------------------------------------------------------
    coverage = load_distributional(results, args.form, codes)
    by_model = recovery_by_model(coverage, codes)
    n_texts = len(coverage)
    n_per_model = int(coverage.groupby("model").size().max())

    pooled_coder = load_coder_recovery(results, config.JUDGE_MODELS, codes)
    per_discourse, mean_coverage = tau_sensitivity(
        coverage, codes, taus=list(config.PRESENCE_SENSITIVITY)
    )
    treatment = load_treatment(results, codes, args.form)

    # A cheap guard against plotting a stale or mismatched table: the headline
    # coverage recomputed here must match the value the analysis step reported.
    headline = by_model[codes].to_numpy().sum() / n_texts
    print(f"Texts:    {n_texts} ({n_per_model} per generator)  "
          f"mean coverage {headline:.2f}/{len(codes)}")

    # --- draw ---------------------------------------------------------------
    print("Figures:")
    save(figure_1(by_model, pooled_coder, codes, names, n_per_model, n_texts),
         args.outdir, "figure1_recovery", args.format)
    save(figure_2(by_model, per_discourse, mean_coverage, codes,
                  n_per_model, n_texts),
         args.outdir, "figure2_reliability", args.format)
    _, standardised = discourse_profiles(treatment, by_model[codes].sum(),
                                         codes, n_texts)
    save(figure_3(standardised, codes, names),
         args.outdir, "figure3_profiles", args.format)
    save(figure_4(treatment, by_model[codes].sum(), codes, n_texts),
         args.outdir, "figure4_treatment", args.format)


if __name__ == "__main__":
    plt.rcParams.update(PLOT_STYLE)
    main()
