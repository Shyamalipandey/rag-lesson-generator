"""Regeneration (Improvement) Agent: rewrites a REJECTED lesson using the evaluator's feedback."""
from pathlib import Path

from agents.checkpoints import LESSON_END, LESSON_START
from agents.evaluator import failed_checks
from agents.generator import GENERATOR_SYSTEM, clean_lesson
from agents.prompt_utils import render_template
from config import PROMPTS_DIR
from llm.client import BaseLLM


class RegenerationAgent:
    def __init__(self, llm: BaseLLM, prompt_path: Path = PROMPTS_DIR / "regeneration_prompt.txt"):
        self.llm = llm
        self.prompt_path = prompt_path

    def regenerate(self, topic: str, rejected_lesson: str, evaluation: dict, prevention_instructions: str = "") -> str:
        """Create an improved lesson. Allowed ONLY after a FAILED evaluation."""
        if evaluation.get("overall_status") != "FAIL":
            raise ValueError("Regeneration is only allowed after a FAILED evaluation.")
        failures = failed_checks(evaluation)
        if not failures:
            raise ValueError("The evaluation has no failed checkpoints to fix.")

        failed_text = "\n".join(f"- [{c['id']}] {c['criterion']}: {c['reason']}" for c in failures)
        suggestions = evaluation.get("improvement_suggestions") or []
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) or "- Fix the failed checkpoints above."

        prompt = render_template(
            self.prompt_path,
            topic=topic,
            failed_checks=failed_text,
            suggestions=suggestions_text,
            prevention_instructions=prevention_instructions.strip() or "No repeated problems have been recorded yet.",
            rejected_lesson=f"{LESSON_START}\n{rejected_lesson}\n{LESSON_END}",
        )
        return clean_lesson(self.llm.complete("regenerate", GENERATOR_SYSTEM, prompt))
