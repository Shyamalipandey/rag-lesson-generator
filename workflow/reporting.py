"""Builds the files that leave the workflow: final lesson, evaluation report, and the demo transcript."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from agents.evaluator import failed_checks
from agents.generator import DEMO_ERROR_SENTENCE
from config import MAX_ATTEMPTS


def _error_score(attempt: dict) -> int:
    """Lower is better. PASS = 0, FAIL = number of failed checks, ERROR (evaluation broke) = worst."""
    evaluation = attempt["evaluation"]
    if evaluation["overall_status"] == "PASS":
        return 0
    if evaluation["overall_status"] == "FAIL":
        return len(failed_checks(evaluation))
    return 999


def pick_best_attempt(attempts: List[dict]) -> Optional[dict]:
    """Best = fewest failed checks; on a tie the later (more improved) attempt wins."""
    if not attempts:
        return None
    return min(attempts, key=lambda a: (_error_score(a), -a["attempt"]))


def attempt_summary(attempt: dict) -> dict:
    evaluation = attempt["evaluation"]
    failed = failed_checks(evaluation)
    return {
        "attempt": attempt["attempt"],
        "status": evaluation["overall_status"],
        "passed_checks": sum(1 for c in evaluation.get("checks", []) if c["status"] == "PASS"),
        "failed_checks": [{"id": c["id"], "criterion": c["criterion"], "reason": c["reason"]} for c in failed],
        "error": evaluation.get("error", ""),
    }


def build_final_markdown(state: dict) -> str:
    """The final lesson file. A failed lesson gets a loud banner so it can never be mistaken for approved."""
    lesson = state.get("final_lesson") or ""
    attempts_used = len(state.get("attempts", []))
    if state["final_status"] == "APPROVED":
        footer = (f"\n\n---\n*Evaluation status: **APPROVED** - all checkpoints passed "
                  f"(attempt {state['best_attempt']} of {MAX_ATTEMPTS} allowed).*\n")
        return lesson.rstrip() + footer
    failed = failed_checks(state.get("final_evaluation") or {})
    reasons = "\n".join(f"> - {c['criterion']}: {c['reason']}" for c in failed) or f"> - {state.get('error') or 'Evaluation could not be completed.'}"
    banner = (f"> **STATUS: FAILED - THIS LESSON IS NOT APPROVED.**\n"
              f"> It did not pass every checkpoint after {attempts_used} attempt(s). A human must review it before use.\n"
              f"> Problems still open:\n{reasons}\n\n")
    return banner + (lesson or "_No lesson could be generated. See the evaluation report for details._") + "\n"


def build_report(state: dict) -> dict:
    attempts = state.get("attempts", [])
    provider = state.get("provider", "unknown")
    return {
        "run_id": state["run_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topic": state["topic"],
        "final_status": state["final_status"],
        "approved": state["final_status"] == "APPROVED",
        "attempts_used": len(attempts),
        "max_attempts": MAX_ATTEMPTS,
        "regenerations_used": max(0, len(attempts) - 1),
        "best_attempt": state.get("best_attempt"),
        "llm_provider": provider,
        "evaluation_method": ("rule-based checker (offline mode - simpler than an LLM judge)"
                              if provider == "offline" else "LLM-based strict rubric"),
        "demo_mode": state.get("demo_mode", False),
        "error": state.get("error", ""),
        "prevention_instructions_used": bool(state.get("prevention_instructions")),
        "final_evaluation": state.get("final_evaluation"),
        "attempt_history": [attempt_summary(a) for a in attempts],
    }


def write_outputs(state: dict, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lesson_path = output_dir / "final_lesson.md"
    report_path = output_dir / "evaluation_report.json"
    lesson_path.write_text(build_final_markdown(state), encoding="utf-8")
    report_path.write_text(json.dumps(build_report(state), indent=2, ensure_ascii=False), encoding="utf-8")
    paths = {"lesson": str(lesson_path), "report": str(report_path)}
    if state.get("demo_mode"):
        transcript = output_dir / "demo_transcript.md"
        transcript.write_text(build_demo_transcript(state), encoding="utf-8")
        paths["demo_transcript"] = str(transcript)
    return paths


def _excerpt_around(lesson: str, needle: str, radius: int = 220) -> str:
    index = lesson.find(needle)
    if index == -1:
        return "(the wrong sentence is not in this version)"
    start = lesson.rfind("\n", 0, max(0, index - radius)) + 1       # snap to the start of a line
    end = lesson.find("\n", index + len(needle) + radius)           # snap to the end of a line
    return lesson[start: end if end != -1 else len(lesson)].strip()


def build_demo_transcript(state: dict) -> str:
    """Readable record of the deliberate-error demonstration, built from what REALLY happened in the run."""
    attempts = state.get("attempts", [])
    lines = ["# Deliberate Error Demonstration", "",
             f"- Run ID: `{state['run_id']}`", f"- LLM provider: `{state.get('provider')}`",
             f"- Final status: **{state['final_status']}** after {len(attempts)} attempt(s)", ""]
    if not attempts:
        return "\n".join(lines + ["No attempt was completed. See the evaluation report."])

    first = attempts[0]
    lines += ["## 1. Initial (incorrect) lesson",
              "Demonstration mode inserted this wrong sentence on purpose into attempt 1:", "",
              f"> {DEMO_ERROR_SENTENCE}", "", "Context inside the lesson:", "", "```",
              _excerpt_around(first["lesson"], DEMO_ERROR_SENTENCE), "```", ""]

    lines += ["## 2. Evaluator result for attempt 1", f"Overall status: **{first['evaluation']['overall_status']}**", ""]
    for check in first["evaluation"].get("checks", []):
        lines.append(f"- {'PASS' if check['status'] == 'PASS' else 'FAIL'} - {check['criterion']}")
    lines.append("")

    failures = failed_checks(first["evaluation"])
    lines.append("## 3. Failed checkpoint(s) and rejection reason")
    lines += [f"- **{c['criterion']}**: {c['reason']}" for c in failures] or ["- (none - the evaluator did NOT catch the error)"]
    lines.append("")

    if len(attempts) > 1:
        second = attempts[1]
        still_wrong = DEMO_ERROR_SENTENCE in second["lesson"]
        lines += ["## 4. Regeneration",
                  f"The regeneration agent received the rejected lesson, the failed checkpoints and the suggestions, and wrote attempt {second['attempt']}.",
                  f"The wrong sentence is {'STILL PRESENT' if still_wrong else 'gone'} in the new lesson.", "",
                  "## 5. Corrected lesson (the part about weights)", "```"]
        lines.append(_excerpt_around(second["lesson"], "does not change", 300) if "does not change" in second["lesson"]
                     else second["lesson"][:600])
        lines += ["```", ""]
        lines += [f"## 6. Evaluation of attempt {second['attempt']}", f"Overall status: **{second['evaluation']['overall_status']}**", ""]
        for check in second["evaluation"].get("checks", []):
            lines.append(f"- {'PASS' if check['status'] == 'PASS' else 'FAIL'} - {check['criterion']}")
    else:
        lines += ["## 4. Regeneration", "No regeneration happened in this run."]
    lines += ["", "## Final result", f"**{state['final_status']}**"]
    return "\n".join(lines) + "\n"
