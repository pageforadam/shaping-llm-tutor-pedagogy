"""Run the full factorial experiment: every (tutor, condition, persona, question) cell.

Re-runnable: cells whose dialogue already exists are skipped, so a crash or a single
failed API call never loses completed work. A failure in one cell is logged and the run
continues -- nothing is saved for that cell, so it simply retries on the next run.

Usage:
    PYTHONPATH=src python -m run_experiment            # full 144-cell run
    PYTHONPATH=src python -m run_experiment --limit 1  # smoke test: one new dialogue
"""
from __future__ import annotations

import argparse

import config
import storage
from orchestrator import run_dialogue


def iter_cells(models, questions):
    """Yield (tutor_key, condition, persona, question) for the full matrix (deterministic order)."""
    for tutor_key in models["tutors"]:
        for condition in config.CONDITIONS:
            for persona in config.PERSONAS:
                for question in questions:
                    yield tutor_key, condition, persona, question


def main(*, max_rounds=None, limit=None, max_consecutive_failures=5, chat_fn=None):
    models = config.load_models()
    questions = config.load_questions()
    cells = list(iter_cells(models, questions))
    total = len(cells)
    new = skipped = failed = 0
    consecutive_failures = 0

    for i, (tutor_key, condition, persona, question) in enumerate(cells, 1):
        if limit is not None and new >= limit:
            break
        run_id = storage.make_run_id(tutor_key, condition, persona, question["id"])
        if storage.dialogue_exists(run_id):
            skipped += 1
            continue
        try:
            record = run_dialogue(tutor_key, condition, persona, question, models,
                                  chat_fn=chat_fn, max_rounds=max_rounds)
            storage.save_dialogue(record)
            new += 1
            consecutive_failures = 0
            print(f"[{i}/{total}] OK   {run_id} ({record['termination']}, {record['num_turns']} turns)")
        except Exception as e:  # noqa: BLE001 - keep the run alive; cell retries next run
            failed += 1
            consecutive_failures += 1
            print(f"[{i}/{total}] FAIL {run_id}: {type(e).__name__}: {e}")
            if consecutive_failures >= max_consecutive_failures:
                print(f"\nAborting after {consecutive_failures} consecutive failures "
                      "(check OPENROUTER_API_KEY / connectivity).")
                break

    summary = {"new": new, "skipped": skipped, "failed": failed, "total": total}
    print(f"\nDone. {summary}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the AI tutoring experiment matrix.")
    p.add_argument("--max-rounds", type=int, default=None,
                   help="Rounds before forced termination (default: from config).")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N new dialogues (for smoke tests).")
    p.add_argument("--max-consecutive-failures", type=int, default=5,
                   help="Abort if this many cells fail in a row (bad key / outage).")
    args = p.parse_args()
    main(max_rounds=args.max_rounds, limit=args.limit,
         max_consecutive_failures=args.max_consecutive_failures)
