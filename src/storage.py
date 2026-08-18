"""Persist dialogues and evaluation scores as human-readable JSON.

One file per run, named by run_id, so dialogues are easy to browse, filter by
metadata, and pull verbatim excerpts from later (supports the qualitative analysis).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIALOGUES_DIR = ROOT / "data" / "dialogues"
SCORES_DIR = ROOT / "data" / "scores"


def make_run_id(tutor_key: str, condition: str, persona: str, question_id: str) -> str:
    """Stable identifier for one experimental cell (double underscores separate fields)."""
    return f"{tutor_key}__{condition}__{persona}__{question_id}"


def _dialogue_path(run_id: str) -> Path:
    return DIALOGUES_DIR / f"{run_id}.json"


def _score_path(run_id: str) -> Path:
    return SCORES_DIR / f"{run_id}.json"


def _write_json(path: Path, record: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_dialogue(record: dict) -> Path:
    return _write_json(_dialogue_path(record["run_id"]), record)


def load_dialogue(run_id: str) -> dict:
    return json.loads(_dialogue_path(run_id).read_text(encoding="utf-8"))


def dialogue_exists(run_id: str) -> bool:
    """Used by the experiment loop to skip already-completed cells on re-run."""
    return _dialogue_path(run_id).exists()


def save_score(record: dict) -> Path:
    return _write_json(_score_path(record["run_id"]), record)


def score_exists(run_id: str) -> bool:
    return _score_path(run_id).exists()


def load_score(run_id: str) -> dict:
    return json.loads(_score_path(run_id).read_text(encoding="utf-8"))


def all_dialogues() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(DIALOGUES_DIR.glob("*.json"))]


def all_scores() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(SCORES_DIR.glob("*.json"))]
