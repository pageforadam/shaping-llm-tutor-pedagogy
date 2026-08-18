"""Analyse the AI-tutoring experiment results.

Reads the evaluator's scores (data/scores/) joined with each dialogue's metadata
(data/dialogues/), builds the comparisons the study cares about, prints a readable
summary, and writes tables (CSV) and figures (PNG) into the analysis/ folder.

Run with:  PYTHONPATH=src python -m analyse

Each rubric dimension is scored 1-5 by the evaluator; `overall` is the mean of the three.
The headline comparison is mean score by prompt condition (NP -> BP -> EP): if more specific
system prompts improve tutoring, scores should rise from NP to BP to EP.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:   # script use: render headless; notebook use: keep inline plots
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Greyscale-only house style for the whole figure set — no colour anywhere in the dissertation.
matplotlib.rcParams.update({
    "axes.prop_cycle": matplotlib.cycler(color=["0.15", "0.45", "0.70", "0.55", "0.30"]),
    "hatch.linewidth": 0.6,
})
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

import config
import storage

# The three rubric dimensions (each scored 1-5 by the evaluator).
DIMENSIONS = ["accuracy_coherence", "guidance_no_solution", "encouragement"]
# Short, readable labels for tables and charts.
DIMENSION_LABELS = {
    "accuracy_coherence": "Accuracy & coherence",
    "guidance_no_solution": "Guidance without solving",
    "encouragement": "Encouragement",
}
# Show conditions in increasing-specificity order everywhere.
CONDITION_ORDER = ["NP", "BP", "EP"]     # No Prompt -> Basic Prompt -> Explicit Prompt
# Short readable question names for facet titles.
QUESTION_LABELS = {"algebra_inequality": "Algebra (inequality)",
                   "probability_counters": "Probability (counters)",
                   "simultaneous_equations": "Simultaneous equations",
                   "compound_interest": "Compound interest"}
# Readable tutor-model names for legends/tables.
TUTOR_LABELS = {"gpt4o": "GPT-4o", "gemini_flash": "Gemini 3.5 Flash",
                "claude_sonnet": "Claude Sonnet 4.6"}
# Greyscale-only house style: models are distinguished by shade + marker/linestyle (lines) and
# shade + hatch (bars), never colour. The pooled "Average" bar is mid-grey, hatched.
TUTOR_ORDER = ["gpt4o", "gemini_flash", "claude_sonnet"]
TUTOR_COLORS = {"gpt4o": "0.15", "gemini_flash": "0.45", "claude_sonnet": "0.60"}
TUTOR_MARKERS = {"gpt4o": "o", "gemini_flash": "s", "claude_sonnet": "^"}
TUTOR_LINESTYLES = {"gpt4o": "-", "gemini_flash": "--", "claude_sonnet": ":"}
TUTOR_HATCHES = {"gpt4o": "", "gemini_flash": "///", "claude_sonnet": "xxx"}
AVG_COLOR = "0.72"
# Display settings for the student-participation process measures.
PARTICIPATION_MEASURES = {
    "num_turns": {"label": "Mean messages per dialogue", "title": "Dialogue length",
                  "pct": False, "fmt": "%.1f"},
    "tutor_words_per_turn": {"label": "Mean words per tutor turn", "title": "Tutor verbosity",
                             "pct": False, "fmt": "%.0f"},
    "student_word_share": {"label": "Student share of words (%)", "title": "Who does the talking?",
                           "pct": True, "fmt": "%.0f%%", "ymax": 100, "parity": 50},
}

OUT_DIR = config.ROOT / "analysis"


def load_results() -> pd.DataFrame:
    """Load every score record into a tidy DataFrame: one row per scored dialogue.

    Columns: the four experimental factors (tutor, condition, persona, question), the three
    dimension scores plus their mean (`overall`), and dialogue metadata (termination, length).
    """
    # index dialogues by run_id so we can attach metadata to each score
    dialogues = {d["run_id"]: d for d in storage.all_dialogues()}

    rows = []
    for s in storage.all_scores():
        row = {
            "run_id": s["run_id"],
            "tutor": s["tutor_key"],
            "condition": s["prompt_condition"],
            "persona": s["persona"],
            "question": s["question_id"],
        }
        # the 1-5 score for each rubric dimension
        for dim in DIMENSIONS:
            row[dim] = s["scores"][dim]["score"]
        row["overall"] = sum(row[dim] for dim in DIMENSIONS) / len(DIMENSIONS)
        # attach how the dialogue ended and how long it ran
        d = dialogues.get(s["run_id"], {})
        row["termination"] = d.get("termination")
        row["num_turns"] = d.get("num_turns")
        # student-participation process measures: who does the talking, and how much?
        # (crude whitespace word counts, but applied identically across conditions)
        turns = d.get("turns", [])
        student_words = sum(len(t["content"].split()) for t in turns if t["role"] == "student")
        tutor_words = sum(len(t["content"].split()) for t in turns if t["role"] == "tutor")
        tutor_turns = sum(1 for t in turns if t["role"] == "tutor")
        row["student_turns"] = sum(1 for t in turns if t["role"] == "student")
        row["tutor_turns"] = tutor_turns
        row["student_words"] = student_words
        row["tutor_words"] = tutor_words
        row["tutor_words_per_turn"] = tutor_words / tutor_turns if tutor_turns else None
        total = student_words + tutor_words
        row["student_word_share"] = student_words / total if total else None
        rows.append(row)

    df = pd.DataFrame(rows)
    # make `condition` an ordered category so NP/BP/EP always sort left-to-right
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    return df


def summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the aggregate tables the study reports (means rounded to 2 dp)."""
    metrics = DIMENSIONS + ["overall"]
    return {
        # headline: does prompting help? mean scores by condition
        "by_condition": df.groupby("condition", observed=True)[metrics].mean().round(2),
        # mean scores by tutor model
        "by_tutor": df.groupby("tutor")[metrics].mean().round(2),
        # the 3x3 factorial: overall score for each condition x tutor cell
        "condition_x_tutor": df.pivot_table(index="condition", columns="tutor",
                                            values="overall", observed=True).round(2),
        # how the personas differ
        "by_persona": df.groupby("persona")[metrics].mean().round(2),
        # how the questions differ
        "by_question": df.groupby("question")[metrics].mean().round(2),
    }


def matched_condition_tests(df: pd.DataFrame, dimension: str = "guidance_no_solution"):
    """Nonparametric within-subjects tests of the prompt-condition effect on one rubric dimension.

    Uses the 48 matched triples (one per tutor x persona x question, each observed under NP/BP/EP),
    so model/persona/question are held fixed within a triple and only the prompt varies:
      - Friedman test: is there ANY overall difference among NP/BP/EP? (+ Kendall's W effect size)
      - pairwise Wilcoxon signed-rank on NP-BP, BP-EP, NP-EP, Bonferroni-corrected: which pairs differ?
    Scores are ordinal (1-5), so these rank-based tests are the appropriate choice. Prints the
    Friedman result and returns the pairwise table (which the notebook displays).
    """
    # one row per matched triple; columns NP / BP / EP
    wide = (df.pivot_table(index=["tutor", "persona", "question"], columns="condition",
                           values=dimension, observed=True)[CONDITION_ORDER].dropna())
    n, k = len(wide), len(CONDITION_ORDER)

    chi2, p = friedmanchisquare(wide["NP"], wide["BP"], wide["EP"])
    kendall_w = chi2 / (n * (k - 1))                        # Friedman effect size, 0..1
    print(f"[{DIMENSION_LABELS.get(dimension, dimension)}]  n = {n} matched triples "
          f"(tutor x persona x question)\n")
    print(f"Friedman: chi2({k - 1}) = {chi2:.1f}, p = {p:.2e} "
          f"({'significant' if p < 0.05 else 'not significant'}); Kendall's W = {kendall_w:.2f} "
          f"(0=no effect, 1=perfect)\n")

    # pairwise: is the LATER (more specific) condition higher? Bonferroni over the 3 comparisons.
    rows = []
    for a, b in [("NP", "BP"), ("BP", "EP"), ("NP", "EP")]:
        diff = wide[b] - wide[a]
        try:
            _, p_raw = wilcoxon(wide[a], wide[b])           # drops zero-differences by default
        except ValueError:                                 # all differences zero -> no effect
            p_raw = 1.0
        rows.append({"comparison": f"{a} → {b}",
                     "later_higher": int((diff > 0).sum()),
                     "earlier_higher": int((diff < 0).sum()),
                     "tied": int((diff == 0).sum()),
                     "median_diff": diff.median(),
                     "p_raw": p_raw, "p_bonferroni": min(p_raw * 3, 1.0)})
    print("Pairwise Wilcoxon signed-rank ('later_higher' = triples where the more specific prompt "
          "scored higher):")
    return pd.DataFrame(rows).set_index("comparison")


def cross_cell_parity(df: pd.DataFrame, cell_a=("gemini_flash", "BP"),
                      cell_b=("gpt4o", "EP"), dimension: str = "guidance_no_solution"):
    """Between-groups test of one striking cross-cell comparison: does a rubric dimension differ
    between (tutor_a, condition_a) and (tutor_b, condition_b)?

    Unlike `matched_condition_tests` (a within-triple test of the prompt effect), this compares two
    independent groups of dialogues that differ in BOTH model and condition -- by default Gemini under
    the basic prompt vs GPT-4o under the explicit prompt, the "a weak prompt on a pedagogy-trained
    model matches a strong prompt on another" comparison. The scores are ordinal and the two groups
    independent, so a Mann-Whitney U (rank-sum) test is used; the one-sided variant asks whether cell
    a scores higher. Prints a readable summary and returns a one-row DataFrame (which main saves).
    """
    (ta, ca), (tb, cb) = cell_a, cell_b
    a = df[(df.tutor == ta) & (df.condition == ca)][dimension]
    b = df[(df.tutor == tb) & (df.condition == cb)][dimension]
    U, p_two = mannwhitneyu(a, b, alternative="two-sided")
    _, p_gt = mannwhitneyu(a, b, alternative="greater")
    name_a = f"{TUTOR_LABELS.get(ta, ta)} / {ca}"
    name_b = f"{TUTOR_LABELS.get(tb, tb)} / {cb}"
    print(f"[{DIMENSION_LABELS.get(dimension, dimension)}] cross-cell comparison "
          "(independent groups, Mann-Whitney U):")
    print(f"  {name_a}: n = {len(a)}, mean = {a.mean():.2f}, median = {a.median():.1f}")
    print(f"  {name_b}: n = {len(b)}, mean = {b.mean():.2f}, median = {b.median():.1f}")
    print(f"  U = {U:.1f}, p(two-sided) = {p_two:.3f}, p(a > b) = {p_gt:.3f} "
          f"({'significant' if p_two < 0.05 else 'not significant'} at .05)\n")
    return pd.DataFrame([{
        "dimension": dimension,
        "group_a": name_a, "n_a": len(a), "mean_a": round(a.mean(), 2), "median_a": a.median(),
        "group_b": name_b, "n_b": len(b), "mean_b": round(b.mean(), 2), "median_b": b.median(),
        "U": U, "p_two_sided": round(p_two, 3), "p_a_greater": round(p_gt, 3),
    }])


def plot_condition_facets(df: pd.DataFrame, facet_by: str = "persona",
                          dimension: str = "guidance_no_solution"):
    """Robustness small-multiples: the NP -> BP -> EP trajectory of a dimension, one panel per level
    of a secondary factor (persona or question), pooled across models. A faint dashed line in every
    panel is the grand mean across all levels, so a panel that departs from the overall pattern is
    obvious. If the panels look alike, the prompt effect generalises across that factor. Returns Fig.
    """
    label = DIMENSION_LABELS.get(dimension, dimension)
    if facet_by == "persona":
        levels = list(config.PERSONAS)
    elif facet_by == "question":
        levels = [q["id"] for q in config.load_questions()]
    else:
        levels = sorted(df[facet_by].unique())

    grand = df.groupby("condition", observed=True)[dimension].mean().reindex(CONDITION_ORDER)
    per = (df.pivot_table(index="condition", columns=facet_by, values=dimension, observed=True)
             .reindex(CONDITION_ORDER))

    ncol = 2
    nrow = (len(levels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 3.2 * nrow), sharey=True)
    axes = axes.flatten()
    for ax, lvl in zip(axes, levels):
        ax.plot(CONDITION_ORDER, grand, color="0.65", linestyle="--", linewidth=1.5,
                label="all (grand mean)")
        ax.plot(CONDITION_ORDER, per[lvl], marker="o", linewidth=2, color="0.15")
        title = QUESTION_LABELS.get(lvl, str(lvl).title()) if facet_by == "question" else str(lvl).title()
        ax.set_title(title)
        ax.set_ylim(1, 5.2)
        ax.grid(alpha=0.3)
    for ax in axes[len(levels):]:          # hide any unused grid cell
        ax.set_visible(False)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("Prompt condition")
    fig.supylabel(f"Mean {label} (1–5)")
    fig.suptitle(f"{label} by prompt condition — faceted by {facet_by}")
    fig.tight_layout()
    return fig


def plot_dimension_trajectories(df: pd.DataFrame):
    """Figure: mean score for each rubric dimension across NP -> BP -> EP.

    The "where does prompting help?" picture. Accuracy & coherence and encouragement sit near the
    ceiling in every condition (the models are accurate and warm by default); the only dimension
    that moves with prompt specificity is guidance-without-solving. Returns the Figure (the script
    saves it; in a notebook it displays inline).
    """
    means = (df.groupby("condition", observed=True)[DIMENSIONS].mean()
               .reindex(CONDITION_ORDER))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    dim_styles = [("0.15", "o", "-"), ("0.45", "s", "--"), ("0.70", "^", ":")]
    for dim, (c, mk, ls) in zip(DIMENSIONS, dim_styles):
        ax.plot(CONDITION_ORDER, means[dim], color=c, marker=mk, linestyle=ls,
                linewidth=2, markersize=6, label=DIMENSION_LABELS[dim])
    ax.set_xlabel("Prompt condition")
    ax.set_ylabel("Mean score (1–5)")
    ax.set_ylim(1, 5.2)                      # 1 is the scale floor, so the full scale is shown
    ax.set_title("Mean rubric score by prompt condition")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_condition_model_interaction(df: pd.DataFrame, dimension: str = "guidance_no_solution"):
    """Interaction plot: mean score across NP -> BP -> EP, one line per tutor model.

    The "how much prompting does each model need?" figure. Gaps between the lines under NP/BP show
    model differences with weak prompting; convergence at EP shows the explicit prompt levelling
    all models up. Defaults to the guidance dimension (the discriminating one); pass any rubric
    dimension or "overall". Returns the Figure.
    """
    label = DIMENSION_LABELS.get(dimension, "Overall")
    means = (df.pivot_table(index="condition", columns="tutor", values=dimension, observed=True)
               .reindex(CONDITION_ORDER))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tutor in [t for t in TUTOR_ORDER if t in means.columns]:
        ax.plot(CONDITION_ORDER, means[tutor], color=TUTOR_COLORS[tutor],
                marker=TUTOR_MARKERS[tutor], linestyle=TUTOR_LINESTYLES[tutor],
                linewidth=2, markersize=6, label=TUTOR_LABELS.get(tutor, tutor))
    ax.set_xlabel("Prompt condition")
    ax.set_ylabel("Mean score (1–5)")
    ax.set_ylim(1, 5.2)                      # full 1-5 scale shown
    ax.set_title(f"{label} — by prompt condition and tutor model")
    ax.legend(title="Tutor model")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_score_distributions(df: pd.DataFrame, dimension: str = "guidance_no_solution",
                             tutor: str | None = None):
    """Likert-style stacked bars: the share of dialogues at each 1-5 score, per condition.

    Means can hide the shape of ordinal scores; this shows it directly - e.g. whether NP fails
    via a pile of 1s (outright answer-giving) or uniform mediocrity, and how much of EP is
    literally at 5. A ceiling dimension appears as one solid block. Pass `tutor` (e.g. "gpt4o")
    to show a single tutor model's distribution. Returns the Figure.
    """
    label = DIMENSION_LABELS.get(dimension, "Overall")
    data = df if tutor is None else df[df["tutor"] == tutor]
    # counts of each score 1-5 per condition -> percentages (each condition has equal n anyway)
    counts = (data.groupby("condition", observed=True)[dimension]
                .value_counts().unstack(fill_value=0)
                .reindex(CONDITION_ORDER)
                .reindex(columns=[1, 2, 3, 4, 5], fill_value=0))
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    colours = plt.get_cmap("Greys")([0.10, 0.26, 0.42, 0.62, 0.82])   # light -> dark = 1 -> 5
    conditions = list(pct.index.astype(str))
    y = list(range(len(conditions)))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    left = [0.0] * len(conditions)
    for score, colour in zip([1, 2, 3, 4, 5], colours):
        vals = pct[score].tolist()
        ax.barh(y, vals, left=left, color=colour, label=str(score), edgecolor="white")
        for i, v in enumerate(vals):            # annotate segments big enough to hold a label
            if v >= 8:
                ax.text(left[i] + v / 2, y[i], f"{v:.0f}%", ha="center", va="center",
                        fontsize=8, color="white" if score >= 4 else "black")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(y)
    ax.set_yticklabels(conditions)
    ax.invert_yaxis()                            # NP on top, EP at the bottom
    ax.set_xlabel("Share of dialogues (%)")
    ax.set_xlim(0, 100)
    title = f"{label} — score distribution by prompt condition"
    if tutor is not None:
        title += f" — {TUTOR_LABELS.get(tutor, tutor)}"
    ax.set_title(title)
    ax.legend(title="Score", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


def plot_student_participation(df: pd.DataFrame):
    """Process corroboration by condition, with no LLM judge involved: dialogue length, tutor
    verbosity, and the student's share of the words. If explicit prompting makes tutors scaffold
    rather than tell, dialogues should involve more exchanges, tutor turns should get shorter
    (Socratic moves, not lectures), and the student should do more of the talking. Returns the Figure.
    """
    df = df.copy()
    df["tutor_words_per_turn"] = df["tutor_words"] / (df["num_turns"] - df["student_turns"])
    m = (df.groupby("condition", observed=True)[["num_turns", "tutor_words_per_turn",
                                                 "student_word_share"]]
           .mean().reindex(CONDITION_ORDER))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.8))

    b1 = ax1.bar(CONDITION_ORDER, m["num_turns"], color="0.45")
    ax1.bar_label(b1, fmt="%.1f")
    ax1.set_xlabel("Prompt condition")
    ax1.set_ylabel("Mean messages per dialogue")
    ax1.set_title("Dialogue length")
    ax1.grid(axis="y", alpha=0.3)

    b2 = ax2.bar(CONDITION_ORDER, m["tutor_words_per_turn"], color="0.45")
    ax2.bar_label(b2, fmt="%.0f")
    ax2.set_xlabel("Prompt condition")
    ax2.set_ylabel("Mean words per tutor turn")
    ax2.set_title("Tutor verbosity")
    ax2.grid(axis="y", alpha=0.3)

    b3 = ax3.bar(CONDITION_ORDER, m["student_word_share"] * 100, color="0.45")
    ax3.bar_label(b3, fmt="%.0f%%")
    ax3.axhline(50, linestyle="--", color="0.3", alpha=0.6)   # parity: student and tutor talk equally
    ax3.set_xlabel("Prompt condition")
    ax3.set_ylabel("Student share of words (%)")
    ax3.set_ylim(0, 100)                                 # full scale; 50% = parity
    ax3.set_title("Who does the talking?")
    ax3.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_participation_by_model(df: pd.DataFrame, measure: str = "num_turns"):
    """One full-width grouped-bar chart of a participation measure: for each prompt condition, a bar
    per tutor model plus the 3-model average (which equals the overall condition mean, since the
    design is balanced). Call once per measure in PARTICIPATION_MEASURES. Returns the Figure.
    """
    cfg = PARTICIPATION_MEASURES[measure]
    scale = 100 if cfg["pct"] else 1
    per = (df.pivot_table(index="condition", columns="tutor", values=measure, observed=True)
             .reindex(CONDITION_ORDER))
    overall = df.groupby("condition", observed=True)[measure].mean().reindex(CONDITION_ORDER)

    series = [(TUTOR_LABELS[t], per[t] * scale, TUTOR_COLORS[t], TUTOR_HATCHES[t]) for t in TUTOR_ORDER]
    series.append(("Average", overall * scale, AVG_COLOR, "//"))   # aggregate bar, hatched

    x = list(range(len(CONDITION_ORDER)))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for idx, (name, vals, colour, hatch) in enumerate(series):
        offset = [xi - 0.4 + width * (idx + 0.5) for xi in x]
        bars = ax.bar(offset, vals, width=width, label=name, color=colour, hatch=hatch,
                      edgecolor="0.2", linewidth=0.6, zorder=3)
        ax.bar_label(bars, fmt=cfg["fmt"], fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER)
    ax.set_xlabel("Prompt condition")
    ax.set_ylabel(cfg["label"])
    ax.set_title(cfg["title"])
    if "ymax" in cfg:
        ax.set_ylim(0, cfg["ymax"])
        if "parity" in cfg:
            ax.axhline(cfg["parity"], linestyle="--", color="0.3", alpha=0.6)   # 50% = student/tutor parity
    else:
        ax.set_ylim(0, max(v for _, vals, _, _ in series for v in vals) * 1.15)   # headroom for labels
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main(out_dir: Path = OUT_DIR):
    """Regenerate every analysis output as files: the summary tables (CSV), the matched-triples test
    table (CSV), and the full figure set (PNG) for the write-up. One command: `python -m analyse`."""
    df = load_results()
    if df.empty:
        print("No score records in data/scores/ yet - run the experiment + evaluation first.")
        return
    print(f"Loaded {len(df)} scored dialogues.\n")

    # tables
    for name, table in summary_tables(df).items():
        table.to_csv(out_dir / f"table_{name}.csv")
    (df.groupby("condition", observed=True)["termination"].value_counts().unstack(fill_value=0)
       .to_csv(out_dir / "table_termination.csv"))
    matched_condition_tests(df, "guidance_no_solution").to_csv(out_dir / "table_condition_tests.csv")
    cross_cell_parity(df).to_csv(out_dir / "table_parity_test.csv", index=False)

    # figures (each plotting function returns a Figure; save it then close to free memory)
    figures = {
        "fig_dimension_trajectories": plot_dimension_trajectories(df),
        "fig_condition_model_interaction": plot_condition_model_interaction(df),
        "fig_guidance_distribution": plot_score_distributions(df),
        "fig_participation": plot_student_participation(df),
        "fig_facet_persona": plot_condition_facets(df, "persona"),
        "fig_facet_question": plot_condition_facets(df, "question"),
    }
    for tutor in TUTOR_ORDER:                     # per-model guidance distributions
        figures[f"fig_guidance_distribution_{tutor}"] = plot_score_distributions(df, tutor=tutor)
    for measure in PARTICIPATION_MEASURES:        # per-model participation charts
        figures[f"fig_participation_{measure}_by_model"] = plot_participation_by_model(df, measure)

    for name, fig in figures.items():
        fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"\nWrote {len(figures)} figures + tables to {out_dir}")


if __name__ == "__main__":
    main()
