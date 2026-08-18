"""Evaluator test-retest (determinism) spot check.

temperature=0 reduces but does NOT guarantee determinism (float non-associativity across GPU
batching, MoE expert routing, OpenRouter backend variation between calls). This script does not try
to prove determinism. It bounds the MEASUREMENT noise: it re-scores a stratified sample of
already-completed dialogues K times with the evaluator (the reasoning model, our riskiest component)
and reports test-retest reliability per rubric dimension.

The dialogues themselves are held fixed - only the evaluator is re-run - so this isolates scoring
noise from generation noise. Selection is a fixed-seed stratified random sample across NP/BP/EP so
the check spans both the ceiling'd dimensions and the discriminating guidance range.

Usage:
    PYTHONPATH=src python -m determinism_check                      # 4 per condition, 3 repeats
    PYTHONPATH=src python -m determinism_check --per-condition 5 --repeats 3 --seed 0
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

import pandas as pd

import config
import evaluate
import storage

DIMENSIONS = ["accuracy_coherence", "guidance_no_solution", "encouragement"]
CONDITION_ORDER = ["NP", "BP", "EP"]
OUT_DIR = config.ROOT / "analysis"


def select_sample(dialogues: list[dict], per_condition: int, seed: int) -> list[dict]:
    """Fixed-seed stratified random sample: `per_condition` dialogues from each of NP/BP/EP."""
    rng = random.Random(seed)
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for record in dialogues:
        by_condition[record["prompt_condition"]].append(record)

    chosen: list[dict] = []
    for condition in CONDITION_ORDER:
        pool = sorted(by_condition[condition], key=lambda r: r["run_id"])  # deterministic order
        chosen += rng.sample(pool, min(per_condition, len(pool)))
    return chosen


def score_once(record, models, rubric, *, attempts: int = 3) -> dict[str, int]:
    """Re-score one dialogue, returning {dimension: score}. Retries transient parse/API failures."""
    last_error = None
    for _ in range(attempts):
        try:
            result = evaluate.evaluate_record(record, models, rubric)
            return {d: int(result["scores"][d]["score"]) for d in DIMENSIONS}
        except Exception as e:  # noqa: BLE001 - JSON parse / transient API errors: retry
            last_error = e
    raise RuntimeError(f"scoring failed after {attempts} attempts: {last_error}")


def rescore_sample(sample: list[dict], repeats: int) -> pd.DataFrame:
    """Score each sampled dialogue `repeats` times. Returns one row per (dialogue, repeat)."""
    models = config.load_models()
    rubric = config.load_evaluator_prompt()

    rows = []
    for i, record in enumerate(sample, 1):
        print(f"[{i}/{len(sample)}] {record['run_id']}", flush=True)
        for k in range(repeats):
            scores = score_once(record, models, rubric)
            rows.append({
                "run_id": record["run_id"],
                "condition": record["prompt_condition"],
                "tutor_key": record["tutor_key"],
                "question_id": record["question_id"],
                "repeat": k,
                **scores,
            })
    return pd.DataFrame(rows)


def summarise(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Per-dimension test-retest summary + per-dialogue spread + overall all-dimensions match rate."""
    long = wide.melt(
        id_vars=["run_id", "condition", "tutor_key", "question_id", "repeat"],
        value_vars=DIMENSIONS, var_name="dimension", value_name="score",
    )

    # per dialogue x dimension, across the repeats: the standard deviation (run-to-run variability,
    # the headline measure of how much the evaluator moves) and the spread (max - min). 0 == stable.
    per = (long.groupby(["run_id", "dimension"], observed=True)["score"]
               .agg(sd=lambda s: s.std(ddof=1),
                    spread=lambda s: s.max() - s.min()).reset_index())

    summary = (per.groupby("dimension", observed=True)
                  .agg(n_dialogues=("spread", "count"),
                       mean_sd=("sd", lambda s: round(s.mean(), 3)),
                       max_sd=("sd", lambda s: round(s.max(), 3)),
                       pct_identical=("spread", lambda s: round((s == 0).mean() * 100, 1)),
                       mean_spread=("spread", lambda s: round(s.mean(), 2)),
                       max_spread=("spread", "max"))
                  .reindex(DIMENSIONS))

    # a dialogue is "fully stable" only if every dimension reproduced exactly across all repeats
    per_dialogue_max = per.groupby("run_id", observed=True)["spread"].max()
    pct_all_identical = round((per_dialogue_max == 0).mean() * 100, 1)
    return summary, per, pct_all_identical


def main(*, per_condition: int = 4, repeats: int = 3, seed: int = 0, out_dir=OUT_DIR):
    dialogues = storage.all_dialogues()
    if not dialogues:
        print("No dialogues in data/dialogues/ - run the experiment first.")
        return

    sample = select_sample(dialogues, per_condition, seed)
    n_calls = len(sample) * repeats
    print(f"Re-scoring {len(sample)} dialogues x {repeats} repeats = {n_calls} evaluator calls "
          f"(seed={seed}).\n")

    wide = rescore_sample(sample, repeats)
    summary, spread, pct_all_identical = summarise(wide)

    wide.to_csv(out_dir / "table_determinism.csv", index=False)          # raw repeats (auditable)
    summary.to_csv(out_dir / "table_determinism_summary.csv")            # per-dimension reliability

    print("\n=== Evaluator test-retest reliability (per dimension) ===")
    print(summary.to_string())
    print(f"\nDialogues reproduced EXACTLY on all three dimensions: {pct_all_identical}% "
          f"of {len(sample)}")
    worst = spread.loc[spread["spread"] >= 2]
    if len(worst):
        print(f"\n{len(worst)} dialogue x dimension case(s) moved by >=2 points:")
        print(worst.to_string(index=False))
    else:
        print("No dialogue x dimension case moved by more than 1 point.")
    print(f"\nWrote table_determinism.csv + table_determinism_summary.csv to {out_dir}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluator test-retest (determinism) spot check.")
    p.add_argument("--per-condition", type=int, default=4, help="dialogues sampled per NP/BP/EP")
    p.add_argument("--repeats", type=int, default=3, help="times to re-score each dialogue")
    p.add_argument("--seed", type=int, default=0, help="stratified-sample seed (reproducible)")
    args = p.parse_args()
    main(per_condition=args.per_condition, repeats=args.repeats, seed=args.seed)
