"""Run a single tutoring dialogue between the student and tutor models.

The student opens (seeded by an unrecorded neutral greeting), then the two models
alternate. After each round an independent referee model judges -- from the question's
model answer -- whether the student has reached the correct final answer; if so the
dialogue ends, else it runs to a max-round cap. Termination is orchestration only, not
an outcome measure, and no termination logic is placed in the tutor/student prompts, so
the NP/BP/EP manipulation stays clean.
"""
from __future__ import annotations

from datetime import datetime, timezone

import api
import config
import storage

# Unrecorded seed to get the student talking first; never shown to the tutor or stored.
STUDENT_KICKOFF = "Hi, I'm your maths tutor. What are you working on, and where are you stuck?"

REFEREE_SYSTEM = (
    "You decide whether a student has reached the correct final answer in a maths tutoring "
    "dialogue. You are given the correct answer and the dialogue so far. Answer strictly YES or "
    "NO. Answer YES only if the student themselves has stated or clearly arrived at the correct "
    "final answer (not merely the tutor stating it). Otherwise answer NO."
)


def _messages(transcript, system_prompt, self_role, lead_user=None):
    """Build a model's message list: its own turns as 'assistant', the other's as 'user'."""
    msgs = [{"role": "system", "content": system_prompt}]
    if lead_user is not None:
        msgs.append({"role": "user", "content": lead_user})
    for turn in transcript:
        role = "assistant" if turn["role"] == self_role else "user"
        msgs.append({"role": role, "content": turn["content"]})
    return msgs


def _params(models):
    d = models["defaults"]
    extra = {"provider": {"require_parameters": d["provider"]["require_parameters"]}}
    return d["temperature"], d["max_tokens"], extra


def _transcript_text(transcript):
    label = {"student": "Student", "tutor": "Tutor"}
    return "\n".join(f"{label[t['role']]}: {t['content']}" for t in transcript)


def referee_reached_answer(transcript, question, models, chat_fn, temperature, extra_body):
    """Independent yes/no check: has the student reached the correct final answer?"""
    user = (
        f"Correct final answer: {question['final_answer']}\n"
        f"Full worked answer: {question['model_answer']}\n\n"
        f"Dialogue so far:\n{_transcript_text(transcript)}\n\n"
        "Has the student reached the correct final answer? Answer YES or NO."
    )
    msgs = [{"role": "system", "content": REFEREE_SYSTEM}, {"role": "user", "content": user}]
    # Referee cap comes from config. DeepSeek is a reasoning model and spends hidden tokens
    # before answering, so the cap must leave room for reasoning + the YES/NO (a tiny cap like
    # 5 truncates the reasoning and returns empty content).
    out = chat_fn(msgs, models["evaluator"]["slug"], temperature=temperature,
                  max_tokens=models["defaults"]["referee_max_tokens"], extra_body=extra_body) or ""
    return out.strip().upper().startswith("Y")


def run_dialogue(tutor_key, condition, persona, question, models, *,
                 chat_fn=None, max_rounds=None):
    """Run one dialogue for a single experimental cell and return the record.

    A round is one tutor turn + one student turn; the student's opening message is not
    counted. max_rounds defaults to the value in config (models.yaml).
    """
    chat_fn = chat_fn or api.chat
    if max_rounds is None:
        max_rounds = models["defaults"]["max_rounds"]
    temperature, max_tokens, extra = _params(models)

    tutor_slug = models["tutors"][tutor_key]["slug"]
    student_slug = models["student"]["slug"]
    tutor_system = config.render_tutor_prompt(condition, question)
    student_system = config.render_persona_prompt(persona, question)

    def call(system, self_role, slug, lead_user=None):
        return chat_fn(_messages(transcript, system, self_role, lead_user), slug,
                       temperature=temperature, max_tokens=max_tokens, extra_body=extra) or ""

    transcript = []
    # Student opens (kickoff seeds generation only).
    transcript.append({"role": "student",
                       "content": call(student_system, "student", student_slug, STUDENT_KICKOFF)})

    termination = "max_rounds"
    for _ in range(max_rounds):
        transcript.append({"role": "tutor", "content": call(tutor_system, "tutor", tutor_slug)})
        if referee_reached_answer(transcript, question, models, chat_fn, temperature, extra):
            termination = "solved"
            break
        transcript.append({"role": "student",
                           "content": call(student_system, "student", student_slug, STUDENT_KICKOFF)})

    return {
        "run_id": storage.make_run_id(tutor_key, condition, persona, question["id"]),
        "tutor_key": tutor_key,
        "tutor_model": tutor_slug,
        "student_model": student_slug,
        "prompt_condition": condition,
        "persona": persona,
        "question_id": question["id"],
        "params": {"temperature": temperature, "max_tokens": max_tokens},
        "tutor_system_prompt": tutor_system,
        "student_system_prompt": student_system,
        "turns": transcript,
        "termination": termination,
        "num_turns": len(transcript),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
