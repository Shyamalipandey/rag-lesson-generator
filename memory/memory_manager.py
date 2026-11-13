"""Persistent memory stored in two JSON files.

memory.json          - long-term summary: failure counts, common mistakes, successful improvements,
                       recurrence tracking, user feedback and a history of runs.
rejection_logs.json  - one entry for every rejected attempt (the raw evidence).

IMPORTANT: this is prompt and workflow improvement using stored feedback. It is NOT model training or
fine-tuning. The LLM's weights never change. We only add better instructions to the next prompt.

Demonstration runs are logged but do NOT count towards "repeated problems", because their mistake is
injected on purpose and would otherwise teach the system a false lesson.
"""
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from agents.checkpoints import CHECKPOINT_BY_ID
from config import MEMORY_PATH, REJECTION_LOG_PATH, REPEAT_THRESHOLD

MAX_RUNS_KEPT = 100
MAX_REJECTIONS_KEPT = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_memory() -> dict:
    return {
        "version": 1,
        "total_runs": 0,
        "total_regenerations": 0,
        "failure_counts": {},          # checkpoint id -> number of real runs in which it failed
        "common_mistakes": {},         # checkpoint id -> criterion, count, last_reason, last_seen
        "successful_improvements": [],  # problems that regeneration fixed
        "recurrence": {},              # checkpoint id -> did it come back after a prevention rule was added?
        "user_feedback": [],
        "runs": [],                    # history, including regeneration attempts
    }


class MemoryManager:
    def __init__(self, memory_path=MEMORY_PATH, rejection_path=REJECTION_LOG_PATH):
        self.memory_path = memory_path
        self.rejection_path = rejection_path

    # ------------------------------------------------------------------ file helpers
    def _read(self, path, default):
        """Read JSON. Missing file -> default. Broken file -> keep a .corrupt copy and start fresh."""
        if not path.exists():
            return default
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            shutil.copy(path, str(path) + ".corrupt")
            return default
        if isinstance(default, dict):
            if not isinstance(data, dict):
                return default
            merged = default_memory() if default.get("version") == 1 else dict(default)
            merged.update(data)  # new keys added in later versions are filled in automatically
            return merged
        return data if isinstance(data, list) else default

    def _write(self, path, data) -> None:
        """Write atomically so a crash never leaves a half-written JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)

    def load_memory(self) -> dict:
        return self._read(self.memory_path, default_memory())

    def load_rejections(self) -> List[dict]:
        return self._read(self.rejection_path, [])

    # ------------------------------------------------------------------ rejection log
    def log_rejection(self, run_id: str, topic: str, attempt: int, evaluation: dict,
                      demo_mode: bool, provider: str) -> None:
        """Save WHY an attempt was rejected. Called right before every regeneration."""
        entries = self.load_rejections()
        entries.append({
            "run_id": run_id,
            "timestamp": _now(),
            "topic": topic,
            "attempt": attempt,
            "demo_mode": demo_mode,
            "provider": provider,
            "failed_checks": [{"id": c["id"], "criterion": c["criterion"], "reason": c["reason"]}
                              for c in evaluation.get("checks", []) if c["status"] == "FAIL"],
            "improvement_suggestions": evaluation.get("improvement_suggestions", []),
        })
        self._write(self.rejection_path, entries[-MAX_REJECTIONS_KEPT:])

    # ------------------------------------------------------------------ learning from the past
    def repeated_problems(self) -> Dict[str, int]:
        """Checkpoint id -> number of DIFFERENT real runs in which it was rejected (demo runs ignored)."""
        runs_per_check: Dict[str, set] = {}
        for entry in self.load_rejections():
            if entry.get("demo_mode"):
                continue
            for check in entry.get("failed_checks", []):
                runs_per_check.setdefault(check["id"], set()).add(entry["run_id"])
        return {cid: len(runs) for cid, runs in runs_per_check.items()}

    def build_prevention_instructions(self) -> Tuple[str, List[str]]:
        """Turn repeated problems into extra instructions for the next generation prompt.

        Returns (instruction_text, list_of_checkpoint_ids_that_got_a_rule).
        """
        counts = self.repeated_problems()
        active = [cid for cid, n in counts.items() if n >= REPEAT_THRESHOLD and cid in CHECKPOINT_BY_ID]
        if not active:
            return "", []
        last_reasons = {cid: m.get("last_reason", "") for cid, m in self.load_memory()["common_mistakes"].items()}
        lines = ["Earlier lessons were rejected more than once for the reasons below. Make sure this lesson avoids them:"]
        for cid in sorted(active, key=lambda c: -counts[c]):
            checkpoint = CHECKPOINT_BY_ID[cid]
            line = f"- {checkpoint.criterion} (failed in {counts[cid]} earlier runs): {checkpoint.prevention}"
            if last_reasons.get(cid):
                line += f" Last time the evaluator said: \"{last_reasons[cid][:200]}\""
            lines.append(line)
        return "\n".join(lines), active

    # ------------------------------------------------------------------ updating memory after a run
    def record_run(self, summary: dict) -> dict:
        """Update memory.json after a completed run. `summary` is built by the workflow's memory node.

        Expected keys: run_id, topic, provider, demo_mode, final_status, attempts_used,
        attempts (list of {attempt, status, failed_ids}), failed_reasons (id -> first reason),
        active_prevention_ids.
        """
        memory = self.load_memory()
        attempts = summary["attempts"]
        demo = summary["demo_mode"]
        failed_any = {cid for a in attempts for cid in a["failed_ids"]}
        first_attempt_failed = set(attempts[0]["failed_ids"]) if attempts else set()
        regenerations = max(0, len(attempts) - 1)

        memory["total_runs"] += 1
        memory["total_regenerations"] += regenerations

        if not demo:
            for cid in failed_any:
                memory["failure_counts"][cid] = memory["failure_counts"].get(cid, 0) + 1
                mistake = memory["common_mistakes"].setdefault(
                    cid, {"criterion": CHECKPOINT_BY_ID[cid].criterion, "count": 0, "last_reason": "", "last_seen": ""})
                mistake["count"] += 1
                mistake["last_reason"] = summary["failed_reasons"].get(cid, mistake["last_reason"])
                mistake["last_seen"] = _now()

            # Did a problem come back although we had already added a prevention rule for it?
            for cid in summary["active_prevention_ids"]:
                record = memory["recurrence"].setdefault(
                    cid, {"runs_with_prevention_rule": 0, "problem_came_back": 0, "problem_avoided": 0})
                record["runs_with_prevention_rule"] += 1
                if cid in first_attempt_failed:
                    record["problem_came_back"] += 1
                else:
                    record["problem_avoided"] += 1

        # A fix is "successful" if the problem failed earlier in this run and the final lesson passed it.
        improvements = []
        if summary["final_status"] == "APPROVED" and len(attempts) > 1:
            for cid in sorted(failed_any):
                improvements.append({
                    "run_id": summary["run_id"], "demo_mode": demo, "checkpoint": cid,
                    "criterion": CHECKPOINT_BY_ID[cid].criterion,
                    "problem": summary["failed_reasons"].get(cid, ""),
                    "fixed_in_attempt": len(attempts),
                })
            memory["successful_improvements"].extend(improvements)

        memory["runs"].append({
            "run_id": summary["run_id"], "timestamp": _now(), "topic": summary["topic"],
            "provider": summary["provider"], "demo_mode": demo,
            "final_status": summary["final_status"], "attempts_used": summary["attempts_used"],
            "regenerations": regenerations, "attempt_results": attempts,
            "prevention_rules_active": summary["active_prevention_ids"],
        })
        memory["runs"] = memory["runs"][-MAX_RUNS_KEPT:]
        self._write(self.memory_path, memory)
        return memory

    def add_user_feedback(self, rating: int, comment: str, run_id: str = "") -> None:
        """Store feedback from a person (for example from the Streamlit page)."""
        if not 1 <= int(rating) <= 5:
            raise ValueError("Rating must be between 1 and 5.")
        memory = self.load_memory()
        memory["user_feedback"].append(
            {"timestamp": _now(), "run_id": run_id, "rating": int(rating), "comment": comment.strip()})
        self._write(self.memory_path, memory)
