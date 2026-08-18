"""Evaluation pipeline: score stored dialogues against the rubric.

Loads each dialogue, sends the transcript to the evaluator model with the rubric
system prompt, parses the JSON scores robustly, and saves one score record per run.
Re-runnable: dialogues whose score already exists are skipped.

Usage:
    PYTHONPATH=src python -m evaluate            # score all dialogues
    PYTHONPATH=src python -m evaluate --limit 1  # smoke test: one score
"""
from __future__ import annotations

import argparse
import json
import re

import api
import config
import storage


def format_dialogue(record) -> str:
    label = {"student": "Student", "tutor": "Tutor"}
    return "\n\n".join(f"{label[t['role']]}: {t['content']}" for t in record["turns"])


def parse_scores(text: str) -> dict:
    """Extract the JSON object from the evaluator's reply (tolerates fences / surrounding prose)."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"No JSON object in evaluator output: {text[:200]!r}")
        return json.loads(m.group(0))


def evaluate_record(record, models, rubric, *, chat_fn=None) -> dict:
    chat_fn = chat_fn or api.chat
    d = models["defaults"]
    extra = {"provider": {"require_parameters": d["provider"]["require_parameters"]}}
    messages = [
        {"role": "system", "content": rubric},
        {"role": "user", "content": format_dialogue(record)},
    ]
    raw = chat_fn(messages, models["evaluator"]["slug"], temperature=d["temperature"],
                  max_tokens=d["max_tokens"], extra_body=extra) or ""
    return {
        "run_id": record["run_id"],
        "tutor_key": record["tutor_key"],
        "prompt_condition": record["prompt_condition"],
        "persona": record["persona"],
        "question_id": record["question_id"],
        "evaluator_model": models["evaluator"]["slug"],
        "scores": parse_scores(raw),
    }


def main(*, limit=None, chat_fn=None):
    models = config.load_models()
    rubric = config.load_evaluator_prompt()
    dialogues = storage.all_dialogues()
    new = skipped = failed = 0

    for i, record in enumerate(dialogues, 1):
        if limit is not None and new >= limit:
            break
        run_id = record["run_id"]
        if storage.score_exists(run_id):
            skipped += 1
            continue
        try:
            storage.save_score(evaluate_record(record, models, rubric, chat_fn=chat_fn))
            new += 1
            print(f"[{i}] OK   {run_id}")
        except Exception as e:  # noqa: BLE001 - keep going; cell retries next run
            failed += 1
            print(f"[{i}] FAIL {run_id}: {type(e).__name__}: {e}")

    summary = {"new": new, "skipped": skipped, "failed": failed, "total": len(dialogues)}
    print(f"\nDone. {summary}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate stored dialogues against the rubric.")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(limit=args.limit)
