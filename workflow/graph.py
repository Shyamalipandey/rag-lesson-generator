"""The LangGraph workflow.

    START -> input -> generate -> evaluate -> decision --approved--------> final_output -> memory_update -> END
                          |                       |   \\--exhausted------> final_output
                          |                       \\--retry--> save_rejection -> regenerate -> evaluate (loop)
                          \\--(API error)---------------------------------> final_output

Termination is guaranteed because:
1. the decision node refuses to retry when attempt >= MAX_ATTEMPTS (3),
2. the regenerate node also raises an error if it is ever called past the limit (safety net),
3. every API failure routes to final_output instead of looping,
4. LangGraph's own recursion_limit is set as a last guard.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.evaluator import EvaluatorAgent, failed_checks
from agents.generator import GeneratorAgent
from agents.improvement import RegenerationAgent
from config import MAX_ATTEMPTS, MAX_TOPIC_LENGTH, OUTPUTS_DIR
from llm.client import BaseLLM, LLMError, get_llm
from memory.memory_manager import MemoryManager
from workflow.reporting import attempt_summary, pick_best_attempt, write_outputs


class LessonState(TypedDict, total=False):
    run_id: str
    topic: str
    demo_mode: bool
    provider: str
    prevention_instructions: str
    active_prevention_ids: List[str]
    attempt: int            # number of the most recent generation (1..3)
    lesson: str             # most recent lesson text
    evaluation: dict        # most recent evaluation result
    attempts: List[dict]    # [{attempt, lesson, evaluation}, ...]
    decision: str           # approved | retry | exhausted
    error: str
    final_status: str       # APPROVED | FAILED
    final_lesson: str
    final_evaluation: dict
    best_attempt: int
    output_paths: dict
    events: List[str]       # human-readable log shown in the CLI and the Streamlit page


def _events(state: dict, message: str) -> List[str]:
    return list(state.get("events", [])) + [message]


def build_graph(llm: BaseLLM, memory: MemoryManager, output_dir: Path, max_attempts: int = MAX_ATTEMPTS):
    max_attempts = min(max_attempts, MAX_ATTEMPTS)  # the assessment forbids more than 3 attempts
    generator, evaluator, regenerator = GeneratorAgent(llm), EvaluatorAgent(llm), RegenerationAgent(llm)

    # ------------------------------------------------------------------ nodes
    def input_node(state: LessonState) -> dict:
        topic = (state.get("topic") or "").strip()
        if not topic:
            raise ValueError("Please enter a topic.")
        if len(topic) > MAX_TOPIC_LENGTH:
            raise ValueError(f"Topic is too long (maximum {MAX_TOPIC_LENGTH} characters).")
        instructions, active_ids = memory.build_prevention_instructions()
        run_id = state.get("run_id") or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        note = (f"Memory: {len(active_ids)} prevention rule(s) added to the prompt." if active_ids
                else "Memory: no repeated problems yet, so no extra rules.")
        return {"topic": topic, "run_id": run_id, "provider": llm.name, "prevention_instructions": instructions,
                "active_prevention_ids": active_ids, "attempt": 0, "attempts": [], "error": "",
                "events": _events(state, f"INPUT: topic accepted. {note}")}

    def generate_node(state: LessonState) -> dict:
        try:
            lesson = generator.generate(state["topic"], state["prevention_instructions"], state.get("demo_mode", False))
        except LLMError as exc:
            return {"error": f"Lesson generation failed: {exc}",
                    "events": _events(state, f"GENERATE: failed - {exc}")}
        message = "GENERATE: attempt 1 created."
        if state.get("demo_mode"):
            message += " DEMO MODE: a wrong sentence was inserted on purpose."
        return {"attempt": 1, "lesson": lesson, "events": _events(state, message)}

    def evaluate_node(state: LessonState) -> dict:
        evaluation = evaluator.evaluate(state["topic"], state["lesson"])
        attempts = state["attempts"] + [{"attempt": state["attempt"], "lesson": state["lesson"], "evaluation": evaluation}]
        if evaluation["overall_status"] == "ERROR":
            message = f"EVALUATE: attempt {state['attempt']} could not be evaluated - {evaluation['error']}"
        else:
            failed = len(failed_checks(evaluation))
            message = f"EVALUATE: attempt {state['attempt']} -> {evaluation['overall_status']} ({failed} failed checkpoint(s))."
        return {"evaluation": evaluation, "attempts": attempts, "events": _events(state, message)}

    def decision_node(state: LessonState) -> dict:
        status = state["evaluation"]["overall_status"]
        if status == "PASS":
            decision = "approved"
        elif status == "FAIL" and state["attempt"] < max_attempts:
            decision = "retry"
        else:
            decision = "exhausted"  # either out of attempts, or evaluation itself is broken (ERROR)
        return {"decision": decision, "events": _events(state, f"DECISION: {decision}.")}

    def save_rejection_node(state: LessonState) -> dict:
        memory.log_rejection(state["run_id"], state["topic"], state["attempt"], state["evaluation"],
                             state.get("demo_mode", False), llm.name)
        return {"events": _events(state, f"REJECTION SAVED: attempt {state['attempt']} written to the rejection log.")}

    def regenerate_node(state: LessonState) -> dict:
        if state["attempt"] >= max_attempts:  # safety net: the decision node should make this impossible
            raise RuntimeError("Retry limit exceeded - regeneration must never run a 4th time.")
        try:
            lesson = regenerator.regenerate(state["topic"], state["lesson"], state["evaluation"],
                                            state["prevention_instructions"])
        except LLMError as exc:
            return {"error": f"Regeneration failed: {exc}", "events": _events(state, f"REGENERATE: failed - {exc}")}
        number = state["attempt"] + 1
        return {"attempt": number, "lesson": lesson,
                "events": _events(state, f"REGENERATE: attempt {number} created from the rejection feedback.")}

    def final_output_node(state: LessonState) -> dict:
        attempts = state.get("attempts", [])
        best = pick_best_attempt(attempts)
        if best is not None and best["evaluation"]["overall_status"] == "PASS":
            status = "APPROVED"
        else:
            status = "FAILED"  # never label an unapproved lesson as approved
        update = {
            "final_status": status,
            "final_lesson": best["lesson"] if best else "",
            "final_evaluation": best["evaluation"] if best else None,
            "best_attempt": best["attempt"] if best else 0,
        }
        merged = {**state, **update}
        update["output_paths"] = write_outputs(merged, output_dir)
        update["events"] = _events(state, f"FINAL OUTPUT: status {status}; files saved.")
        return update

    def memory_update_node(state: LessonState) -> dict:
        attempts = [{"attempt": a["attempt"], "status": a["evaluation"]["overall_status"],
                     "failed_ids": [c["id"] for c in failed_checks(a["evaluation"])]} for a in state.get("attempts", [])]
        failed_reasons = {}
        for a in state.get("attempts", []):
            for c in failed_checks(a["evaluation"]):
                failed_reasons.setdefault(c["id"], c["reason"])
        memory.record_run({
            "run_id": state["run_id"], "topic": state["topic"], "provider": llm.name,
            "demo_mode": state.get("demo_mode", False), "final_status": state["final_status"],
            "attempts_used": len(attempts), "attempts": attempts, "failed_reasons": failed_reasons,
            "active_prevention_ids": state.get("active_prevention_ids", []),
        })
        return {"events": _events(state, "MEMORY UPDATED.")}

    # ------------------------------------------------------------------ routing (conditional edges)
    def route_after_generate(state: LessonState) -> str:
        return "final_output" if state.get("error") else "evaluate"

    def route_after_decision(state: LessonState) -> str:
        return "save_rejection" if state["decision"] == "retry" else "final_output"

    def route_after_regenerate(state: LessonState) -> str:
        return "final_output" if state.get("error") else "evaluate"

    graph = StateGraph(LessonState)
    graph.add_node("input", input_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("decision", decision_node)
    graph.add_node("save_rejection", save_rejection_node)
    graph.add_node("regenerate", regenerate_node)
    graph.add_node("final_output", final_output_node)
    graph.add_node("memory_update", memory_update_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "generate")
    graph.add_conditional_edges("generate", route_after_generate, {"evaluate": "evaluate", "final_output": "final_output"})
    graph.add_edge("evaluate", "decision")
    graph.add_conditional_edges("decision", route_after_decision, {"save_rejection": "save_rejection", "final_output": "final_output"})
    graph.add_edge("save_rejection", "regenerate")
    graph.add_conditional_edges("regenerate", route_after_regenerate, {"evaluate": "evaluate", "final_output": "final_output"})
    graph.add_edge("final_output", "memory_update")
    graph.add_edge("memory_update", END)
    return graph.compile()


def run_workflow(topic: str, demo_mode: bool = False, llm: BaseLLM = None, memory: MemoryManager = None,
                 output_dir: Path = None, max_attempts: int = MAX_ATTEMPTS) -> dict:
    """Run the complete cycle once and return the final state."""
    llm = llm or get_llm()
    memory = memory or MemoryManager()
    output_dir = Path(output_dir) if output_dir else OUTPUTS_DIR
    app = build_graph(llm, memory, output_dir, max_attempts)
    return app.invoke({"topic": topic, "demo_mode": demo_mode}, config={"recursion_limit": 40})


__all__ = ["build_graph", "run_workflow", "LessonState", "attempt_summary"]
