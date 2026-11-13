"""Evaluator Agent: judges a lesson with strict PASS/FAIL checkpoints and returns validated JSON.

Safety rules built into this file:
* The evaluator's JSON is VALIDATED before the workflow uses it.
* The overall status is RECOMPUTED from the individual checks. We never trust the model's own
  "overall_status" - so a model cannot approve a lesson that has a failed checkpoint.
* If the output stays invalid (or the API fails) we return status "ERROR", which is never "PASS".
"""
import json
import re
from pathlib import Path
from typing import Dict, List

from agents.checkpoints import CHECKPOINT_BY_ID, CHECKPOINTS, LESSON_END, LESSON_START, rubric_text
from agents.prompt_utils import render_template
from config import EVALUATOR_FORMAT_RETRIES, PROMPTS_DIR
from llm.client import BaseLLM, LLMError

EVALUATOR_SYSTEM = (
    "You are a strict, skeptical reviewer of educational content about AI. You check facts carefully, "
    "never give partial marks, and answer only with the requested JSON."
)

MIN_FAIL_REASON_LENGTH = 15  # a FAIL must come with a real explanation, not just "bad"


class EvaluationFormatError(ValueError):
    """The evaluator output is not valid JSON, or does not follow the required structure."""


def extract_json(raw: str) -> dict:
    """Pull the JSON object out of the model's text (it may add ```json fences or small extra text)."""
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise EvaluationFormatError("No JSON object found in the evaluator output.")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise EvaluationFormatError(f"Evaluator output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EvaluationFormatError("Evaluator JSON must be an object.")
    return data


def _match_checkpoint_id(item: dict) -> str:
    """Find which rubric checkpoint an item belongs to, by id or by criterion name."""
    raw_id = str(item.get("id", "")).strip().lower()
    if raw_id in CHECKPOINT_BY_ID:
        return raw_id
    criterion = str(item.get("criterion", "")).strip().lower()
    for checkpoint in CHECKPOINTS:
        if checkpoint.criterion.lower() == criterion:
            return checkpoint.id
    raise EvaluationFormatError(f"Unknown checkpoint in evaluator output: id={item.get('id')!r}, criterion={item.get('criterion')!r}.")


def validate_evaluation(data: dict) -> dict:
    """Check the structure and return a clean, normalised evaluation dictionary.

    Raises EvaluationFormatError when anything is wrong, so a broken answer is never used.
    """
    if not isinstance(data.get("checks"), list):
        raise EvaluationFormatError('"checks" must be a list.')

    by_id: Dict[str, dict] = {}
    for item in data["checks"]:
        if not isinstance(item, dict):
            raise EvaluationFormatError("Every item in checks must be an object.")
        checkpoint_id = _match_checkpoint_id(item)
        if checkpoint_id in by_id:
            raise EvaluationFormatError(f"Checkpoint '{checkpoint_id}' appears more than once.")
        status = str(item.get("status", "")).strip().upper()
        if status not in ("PASS", "FAIL"):
            raise EvaluationFormatError(f"Checkpoint '{checkpoint_id}' has status {item.get('status')!r}; only PASS or FAIL is allowed (no partial marks).")
        reason = str(item.get("reason", "")).strip()
        if status == "FAIL" and len(reason) < MIN_FAIL_REASON_LENGTH:
            raise EvaluationFormatError(f"Checkpoint '{checkpoint_id}' is FAIL but has no specific reason.")
        by_id[checkpoint_id] = {"id": checkpoint_id, "criterion": CHECKPOINT_BY_ID[checkpoint_id].criterion,
                                "status": status, "reason": reason}

    missing = [c.id for c in CHECKPOINTS if c.id not in by_id]
    if missing:
        raise EvaluationFormatError("Evaluator did not judge these checkpoints: " + ", ".join(missing))

    checks: List[dict] = [by_id[c.id] for c in CHECKPOINTS]  # always in rubric order
    computed_overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"

    notes: List[str] = []
    claimed = str(data.get("overall_status", "")).strip().upper()
    if claimed and claimed != computed_overall:
        notes.append(f"Model claimed overall_status={claimed}, but the checkpoints give {computed_overall}. The checkpoints win.")

    suggestions = data.get("improvement_suggestions", [])
    if not isinstance(suggestions, list):
        raise EvaluationFormatError('"improvement_suggestions" must be a list.')
    suggestions = [str(s).strip() for s in suggestions if str(s).strip()]
    if computed_overall == "FAIL" and not suggestions:
        suggestions = [f"Fix '{c['criterion']}': {c['reason']}" for c in checks if c["status"] == "FAIL"]
        notes.append("No suggestions were returned, so they were built from the failure reasons.")

    return {"overall_status": computed_overall, "checks": checks,
            "improvement_suggestions": suggestions, "validation_notes": notes}


def make_error_evaluation(reason: str) -> dict:
    """Used when evaluation itself broke. Status ERROR can never be treated as approval."""
    return {"overall_status": "ERROR", "checks": [], "improvement_suggestions": [],
            "validation_notes": [], "error": reason}


def failed_checks(evaluation: dict) -> List[dict]:
    return [c for c in evaluation.get("checks", []) if c.get("status") == "FAIL"]


class EvaluatorAgent:
    def __init__(self, llm: BaseLLM, prompt_path: Path = PROMPTS_DIR / "evaluation_prompt.txt"):
        self.llm = llm
        self.prompt_path = prompt_path

    def evaluate(self, topic: str, lesson: str) -> dict:
        """Return a validated evaluation. Never raises: failures become an ERROR evaluation."""
        lesson_block = f"{LESSON_START}\n{lesson}\n{LESSON_END}"
        base_prompt = render_template(self.prompt_path, topic=topic, checkpoints=rubric_text(), lesson_block=lesson_block)

        last_error = ""
        for _ in range(1 + EVALUATOR_FORMAT_RETRIES):
            prompt = base_prompt
            if last_error:
                prompt += ("\n\nYOUR PREVIOUS ANSWER WAS REJECTED BECAUSE: " + last_error +
                           "\nReturn ONLY one valid JSON object that follows the format exactly.")
            try:
                raw = self.llm.complete("evaluate", EVALUATOR_SYSTEM, prompt)
            except LLMError as exc:
                return make_error_evaluation(f"Evaluator API call failed: {exc}")
            try:
                return validate_evaluation(extract_json(raw))
            except EvaluationFormatError as exc:
                last_error = str(exc)
        return make_error_evaluation(f"Evaluator output stayed invalid after {1 + EVALUATOR_FORMAT_RETRIES} tries: {last_error}")
