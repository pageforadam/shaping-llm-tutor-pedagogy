# Shaping the Pedagogical Behaviour of LLM Tutors: Prompt Specificity and Model Choice

Code, data, and analysis accompanying the master's dissertation of Adam Cranfield.

## Abstract

System prompts are the primary lever by which developers, and the major AI providers, shape how large
language models (LLMs) behave as tutors — yet how reliably the specificity of a prompt governs a
model's pedagogical behaviour, and how far that depends on the model, remains poorly established. This
dissertation reports a fully automated, simulation-based experiment in which LLMs perform all three
roles of a tutoring interaction: tutor, student and evaluator. Three leading tutor models — GPT-4o,
Gemini 3.5 Flash and Claude Sonnet 4.6 — are each run under three levels of system-prompt specificity
(no prompt, a basic role prompt, and an explicit pedagogical prompt) against four simulated learner
personas and four GCSE mathematics problems, yielding 144 dialogues scored on a three-dimension
pedagogical rubric.

Prompt specificity changed a single aspect of tutoring behaviour — the tutor's restraint in guiding the
learner rather than giving the answer — while accuracy and encouragement stayed high throughout. The
effect was strongly model-dependent: a lightly prompted model with base-model pedagogical training
(Gemini 3.5 Flash) scaffolded as well as a fully prompted general-purpose one (GPT-4o), and it was the
content of the instruction,
not a mere tutoring label, that produced the change. Under explicit prompting the effect held across
every persona and problem, though models withheld answers more reliably than they elicited reasoning.
The study contributes a reusable, reproducible pipeline for generating and evaluating tutoring
dialogues at scale, and argues that prompt and model must be chosen together — a prompt that produces
excellent tutoring on one model may fall flat on another.

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and add your `OPENROUTER_API_KEY`

## Layout

- `config/models.yaml` — the model roster (slugs, providers, parameters); single source of truth
- `prompts/` — the three tutor conditions (`tutor_np/bp/ep.txt`), the four student personas
  (`personas/`), and the evaluator rubric (`evaluator_rubric.txt`)
- `questions/questions.json` — the four GCSE questions with model answers
- `src/` — the pipeline: conversation orchestration, experiment runner, evaluation, and analysis
- `data/` — the generated dialogues and their scores (`scores/` current; `scores_v1/` pre-revision),
  plus the human validation scores (`human scores/`)
- `analysis/` — the analysis notebook, figures, summary tables, and human-evaluation materials

## Reproducing the study

The pipeline runs in stages — generation (`run_experiment.py`), evaluation (`evaluate.py`), analysis
(`analyse.py`), and the evaluator validity checks (`human_agreement.py` for human–LLM agreement,
`determinism_check.py` for test–retest). Run each with, e.g.:

```
python src/analyse.py
```

All model calls use temperature 0. Because providers update their models over time, exact outputs may
not reproduce identically, so the generated dialogues and scores are included in `data/` and the
analysis can be re-run against them as-is.

## Source materials

The four problems are genuine Edexcel GCSE (9-1) Mathematics Sample Assessment Materials, reproduced
under fair dealing for non-commercial research. The source document is not redistributed here.
