"""Load experiment configuration: models, prompts, and questions.

Everything the experiment needs that lives outside code -- model slugs/params, the
tutor/persona/evaluator prompt text, and the question set -- is loaded here, so the
rest of the code stays declarative. Prompts are rendered with str.replace (not
str.format) to avoid clashing with the literal braces in the evaluator's JSON spec.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PROMPTS_DIR = ROOT / "prompts"
QUESTIONS_DIR = ROOT / "questions"

# Prompt condition -> filename
TUTOR_PROMPT_FILES = {
    "NP": "tutor_np.txt",
    "BP": "tutor_bp.txt",
    "EP": "tutor_ep.txt",
}
CONDITIONS = list(TUTOR_PROMPT_FILES)
PERSONAS = ["struggling", "impulsive", "passive", "curious"]


def load_models() -> dict:
    """Return the parsed models.yaml (defaults, tutors, student, evaluator)."""
    return yaml.safe_load((CONFIG_DIR / "models.yaml").read_text(encoding="utf-8"))


def load_questions() -> list[dict]:
    """Return the list of question dicts (id, topic, question, model_answer, final_answer)."""
    return json.loads((QUESTIONS_DIR / "questions.json").read_text(encoding="utf-8"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_tutor_prompt(condition: str, question: dict) -> str:
    """Tutor system prompt for a condition, with the problem and model answer filled in."""
    template = _read(PROMPTS_DIR / TUTOR_PROMPT_FILES[condition])
    return (template
            .replace("{question}", question["question"])
            .replace("{model_answer}", question["model_answer"]))


def render_persona_prompt(persona: str, question: dict) -> str:
    """Student system prompt for a persona, with the problem filled in.

    The model answer is deliberately NOT provided to the student.
    """
    template = _read(PROMPTS_DIR / "personas" / f"{persona}.txt")
    return template.replace("{question}", question["question"])


def load_evaluator_prompt() -> str:
    """Return the evaluator system prompt (used as-is; no placeholder substitution)."""
    return _read(PROMPTS_DIR / "evaluator_rubric.txt")
