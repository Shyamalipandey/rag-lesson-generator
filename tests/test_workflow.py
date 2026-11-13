"""Tests for the whole project. They need NO API key and cost nothing.

Two kinds of fake LLM are used:
* ScriptedLLM - returns pre-written answers (lets us force bad JSON, API errors, endless failures ...).
* OfflineLLM  - the project's offline provider, whose evaluator genuinely reads the lesson text.
"""
import json

import pytest

from agents.checkpoints import CHECKPOINT_IDS
from agents.evaluator import (EvaluationFormatError, EvaluatorAgent, extract_json, failed_checks,
                              validate_evaluation)
from agents.generator import DEMO_ERROR_SENTENCE, GeneratorAgent, inject_demo_error
from agents.improvement import RegenerationAgent
from llm.client import BaseLLM, LLMError, get_llm
from llm.offline import OfflineLLM, load_reference_lesson, rule_based_evaluation
from memory.memory_manager import MemoryManager
from workflow.graph import run_workflow


# ----------------------------------------------------------------------------- helpers
class ScriptedLLM(BaseLLM):
    """Answers from a script: {"generate": [...], "evaluate": [...], "regenerate": [...]}.
    The last answer of each list repeats forever. An Exception in the list is raised."""

    name = "scripted"

    def __init__(self, script):
        self.script = {task: list(items) for task, items in script.items()}
        self.calls = []

    def complete(self, task, system, prompt):
        self.calls.append((task, prompt))
        queue = self.script.get(task)
        if not queue:
            raise LLMError(f"No scripted answer for task '{task}'.")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    def count(self, task):
        return sum(1 for t, _ in self.calls if t == task)


def make_evaluation_json(failing=None):
    """Build evaluator JSON by hand. `failing` = {checkpoint_id: reason}."""
    failing = failing or {}
    checks = [{"id": cid, "criterion": cid, "status": "FAIL" if cid in failing else "PASS",
               "reason": failing.get(cid, "Looks correct.")} for cid in CHECKPOINT_IDS]
    return json.dumps({"overall_status": "FAIL" if failing else "PASS", "checks": checks,
                       "improvement_suggestions": ["Fix it."] if failing else []})


@pytest.fixture
def good_lesson():
    return load_reference_lesson()


@pytest.fixture
def bad_lesson(good_lesson):
    return inject_demo_error(good_lesson)


@pytest.fixture
def memory(tmp_path):
    return MemoryManager(tmp_path / "memory.json", tmp_path / "rejections.json")


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


WEIGHTS_FAIL = {"technical_accuracy": "The lesson says RAG changes the model weights, which is wrong; RAG only supplies context."}


# ----------------------------------------------------------------------------- 1. successful lesson generation
def test_generator_returns_a_lesson_and_uses_memory_instructions(good_lesson):
    llm = ScriptedLLM({"generate": [good_lesson]})
    lesson = GeneratorAgent(llm).generate("Intro to RAG", prevention_instructions="- Always explain tokens.")
    assert lesson.startswith("# Introduction to RAG")
    prompt = llm.calls[0][1]
    assert "Intro to RAG" in prompt and "Always explain tokens." in prompt


def test_generator_rejects_tiny_answers():
    with pytest.raises(LLMError):
        GeneratorAgent(ScriptedLLM({"generate": ["ok"]})).generate("Intro to RAG")


def test_demo_mode_injects_the_wrong_sentence(good_lesson):
    lesson = GeneratorAgent(ScriptedLLM({"generate": [good_lesson]})).generate("x", demo_mode=True)
    assert DEMO_ERROR_SENTENCE in lesson
    assert DEMO_ERROR_SENTENCE not in good_lesson


# ----------------------------------------------------------------------------- 2. evaluator detects incorrect information
def test_rule_based_evaluator_accepts_the_correct_lesson(good_lesson):
    result = rule_based_evaluation(good_lesson)
    assert result["overall_status"] == "PASS"
    assert len(result["checks"]) >= 12


def test_evaluator_detects_the_weights_mistake(bad_lesson):
    """This reads the real text - it is not mocked. A rubber-stamp evaluator would fail this test."""
    result = EvaluatorAgent(OfflineLLM()).evaluate("Intro to RAG", bad_lesson)
    assert result["overall_status"] == "FAIL"
    failed = {c["id"]: c for c in failed_checks(result)}
    assert list(failed) == ["technical_accuracy"]
    assert "weights" in failed["technical_accuracy"]["reason"].lower()


def test_evaluator_fails_a_lesson_with_missing_sections(good_lesson):
    truncated = good_lesson.split("## 15.")[0]
    result = EvaluatorAgent(OfflineLLM()).evaluate("Intro to RAG", truncated)
    assert result["overall_status"] == "FAIL"
    assert "assessment_questions" in [c["id"] for c in failed_checks(result)]


def test_question_about_weights_is_not_treated_as_a_wrong_claim(good_lesson):
    # The lesson contains the quiz question "Does RAG change the weights of the LLM?" - must not be flagged.
    assert "Does RAG change the weights" in good_lesson
    result = rule_based_evaluation(good_lesson)
    assert next(c for c in result["checks"] if c["id"] == "technical_accuracy")["status"] == "PASS"


def test_evaluator_never_trusts_the_models_overall_status():
    data = json.loads(make_evaluation_json(WEIGHTS_FAIL))
    data["overall_status"] = "PASS"  # the model lies
    validated = validate_evaluation(data)
    assert validated["overall_status"] == "FAIL"
    assert validated["validation_notes"]


# ----------------------------------------------------------------------------- 3. regeneration corrects a failed checkpoint
def test_regeneration_prompt_contains_failures_and_suggestions(good_lesson, bad_lesson):
    evaluation = validate_evaluation(json.loads(make_evaluation_json(WEIGHTS_FAIL)))
    llm = ScriptedLLM({"regenerate": [good_lesson]})
    RegenerationAgent(llm).regenerate("Intro to RAG", bad_lesson, evaluation)
    prompt = llm.calls[0][1]
    assert "technical_accuracy" in prompt
    assert WEIGHTS_FAIL["technical_accuracy"] in prompt
    assert DEMO_ERROR_SENTENCE in prompt  # the rejected lesson is included


def test_regeneration_refuses_to_run_without_a_failed_evaluation(good_lesson):
    evaluation = validate_evaluation(json.loads(make_evaluation_json()))
    with pytest.raises(ValueError):
        RegenerationAgent(ScriptedLLM({"regenerate": [good_lesson]})).regenerate("x", good_lesson, evaluation)


def test_full_cycle_regeneration_fixes_the_deliberate_error(memory, out):
    state = run_workflow("Intro to RAG", demo_mode=True, llm=OfflineLLM(), memory=memory, output_dir=out)
    first, second = state["attempts"]
    assert first["evaluation"]["overall_status"] == "FAIL"
    assert DEMO_ERROR_SENTENCE in first["lesson"]
    assert DEMO_ERROR_SENTENCE not in second["lesson"]
    assert second["evaluation"]["overall_status"] == "PASS"
    assert state["final_status"] == "APPROVED"
    assert (out / "demo_transcript.md").exists()


# ----------------------------------------------------------------------------- 4. successful final approval
def test_approved_on_first_attempt_writes_both_output_files(memory, out):
    llm = ScriptedLLM({"generate": [load_reference_lesson()], "evaluate": [make_evaluation_json()]})
    state = run_workflow("Intro to RAG", llm=llm, memory=memory, output_dir=out)
    assert state["final_status"] == "APPROVED"
    assert len(state["attempts"]) == 1
    assert llm.count("regenerate") == 0  # stops immediately when everything passes
    report = json.loads((out / "evaluation_report.json").read_text())
    assert report["approved"] is True and report["final_status"] == "APPROVED"
    assert "APPROVED" in (out / "final_lesson.md").read_text()


# ----------------------------------------------------------------------------- 5. maximum retry limit
def test_retry_limit_is_three_attempts_and_result_is_failed(memory, out, bad_lesson):
    llm = ScriptedLLM({"generate": [bad_lesson], "regenerate": [bad_lesson],
                       "evaluate": [make_evaluation_json(WEIGHTS_FAIL)]})
    state = run_workflow("Intro to RAG", llm=llm, memory=memory, output_dir=out)
    assert llm.count("generate") == 1
    assert llm.count("regenerate") == 2          # never a third regeneration
    assert llm.count("evaluate") == 3            # every attempt is evaluated
    assert len(state["attempts"]) == 3
    assert state["final_status"] == "FAILED"     # never falsely approved
    report = json.loads((out / "evaluation_report.json").read_text())
    assert report["approved"] is False
    assert (out / "final_lesson.md").read_text().startswith("> **STATUS: FAILED")


def test_best_attempt_is_returned_when_all_attempts_fail(memory, out, bad_lesson):
    two_fails = {**WEIGHTS_FAIL, "beginner_language": "Sentences are far too long for a beginner learner."}
    llm = ScriptedLLM({"generate": [bad_lesson], "regenerate": [bad_lesson],
                       "evaluate": [make_evaluation_json(two_fails), make_evaluation_json(two_fails),
                                    make_evaluation_json(WEIGHTS_FAIL)]})
    state = run_workflow("Intro to RAG", llm=llm, memory=memory, output_dir=out)
    assert state["final_status"] == "FAILED"
    assert state["best_attempt"] == 3  # only 1 failed check, the fewest


def test_max_attempts_can_never_be_raised_above_three(memory, out, bad_lesson):
    llm = ScriptedLLM({"generate": [bad_lesson], "regenerate": [bad_lesson],
                       "evaluate": [make_evaluation_json(WEIGHTS_FAIL)]})
    state = run_workflow("Intro to RAG", llm=llm, memory=memory, output_dir=out, max_attempts=10)
    assert len(state["attempts"]) == 3


# ----------------------------------------------------------------------------- 6. persistent memory
def test_memory_survives_a_restart_and_logs_rejections(tmp_path, bad_lesson):
    paths = (tmp_path / "m.json", tmp_path / "r.json")
    llm = ScriptedLLM({"generate": [bad_lesson], "regenerate": [load_reference_lesson()],
                       "evaluate": [make_evaluation_json(WEIGHTS_FAIL), make_evaluation_json()]})
    run_workflow("Intro to RAG", llm=llm, memory=MemoryManager(*paths), output_dir=tmp_path / "o")

    fresh = MemoryManager(*paths)  # a brand-new object reading the same files = "next day"
    memory = fresh.load_memory()
    assert memory["total_runs"] == 1 and memory["total_regenerations"] == 1
    assert memory["failure_counts"] == {"technical_accuracy": 1}
    assert memory["successful_improvements"][0]["checkpoint"] == "technical_accuracy"
    rejections = fresh.load_rejections()
    assert len(rejections) == 1 and rejections[0]["failed_checks"][0]["id"] == "technical_accuracy"


def test_repeated_problem_becomes_a_prevention_rule_in_the_next_prompt(tmp_path, bad_lesson):
    paths = (tmp_path / "m.json", tmp_path / "r.json")
    always_bad = {"generate": [bad_lesson], "regenerate": [bad_lesson], "evaluate": [make_evaluation_json(WEIGHTS_FAIL)]}
    run_workflow("Intro to RAG", llm=ScriptedLLM(always_bad), memory=MemoryManager(*paths), output_dir=tmp_path / "o1")
    assert MemoryManager(*paths).build_prevention_instructions() == ("", [])  # one run is not "repeated"
    run_workflow("Intro to RAG", llm=ScriptedLLM(always_bad), memory=MemoryManager(*paths), output_dir=tmp_path / "o2")

    text, active = MemoryManager(*paths).build_prevention_instructions()
    assert active == ["technical_accuracy"] and "does NOT change the model's weights" in text

    third = ScriptedLLM({"generate": [load_reference_lesson()], "evaluate": [make_evaluation_json()]})
    state = run_workflow("Intro to RAG", llm=third, memory=MemoryManager(*paths), output_dir=tmp_path / "o3")
    assert "does NOT change the model's weights" in third.calls[0][1]  # the rule reached the generation prompt
    assert state["active_prevention_ids"] == ["technical_accuracy"]
    recurrence = MemoryManager(*paths).load_memory()["recurrence"]["technical_accuracy"]
    assert recurrence == {"runs_with_prevention_rule": 1, "problem_came_back": 0, "problem_avoided": 1}


def test_recurrence_is_tracked_when_the_problem_comes_back(tmp_path, bad_lesson):
    paths = (tmp_path / "m.json", tmp_path / "r.json")
    always_bad = {"generate": [bad_lesson], "regenerate": [bad_lesson], "evaluate": [make_evaluation_json(WEIGHTS_FAIL)]}
    for index in range(3):  # runs 1-2 create the rule, run 3 runs WITH the rule and fails again
        run_workflow("x", llm=ScriptedLLM(always_bad), memory=MemoryManager(*paths), output_dir=tmp_path / str(index))
    recurrence = MemoryManager(*paths).load_memory()["recurrence"]["technical_accuracy"]
    assert recurrence["problem_came_back"] == 1


def test_demo_runs_do_not_teach_false_lessons(tmp_path):
    paths = (tmp_path / "m.json", tmp_path / "r.json")
    for index in range(3):
        run_workflow("x", demo_mode=True, llm=OfflineLLM(), memory=MemoryManager(*paths), output_dir=tmp_path / str(index))
    memory = MemoryManager(*paths)
    assert len(memory.load_rejections()) == 3          # logged ...
    assert memory.build_prevention_instructions() == ("", [])  # ... but not counted as repeated problems
    assert memory.load_memory()["failure_counts"] == {}


def test_user_feedback_is_stored_and_validated(memory):
    memory.add_user_feedback(5, "Very clear", run_id="r1")
    assert memory.load_memory()["user_feedback"][0]["rating"] == 5
    with pytest.raises(ValueError):
        memory.add_user_feedback(9, "invalid rating")


def test_corrupt_memory_file_is_recovered(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{ this is not json")
    manager = MemoryManager(path, tmp_path / "r.json")
    assert manager.load_memory()["total_runs"] == 0
    assert (tmp_path / "m.json.corrupt").exists()


# ----------------------------------------------------------------------------- 7. invalid evaluator JSON handling
@pytest.mark.parametrize("raw", [
    "not json at all",
    '{"checks": "should be a list"}',
    '{"checks": []}',  # no checkpoints judged
])
def test_invalid_evaluator_output_is_rejected(raw):
    with pytest.raises(EvaluationFormatError):
        validate_evaluation(extract_json(raw))


def test_partial_marks_are_not_allowed():
    data = json.loads(make_evaluation_json())
    data["checks"][0]["status"] = "PARTIAL"
    with pytest.raises(EvaluationFormatError):
        validate_evaluation(data)


def test_fail_without_a_reason_is_rejected():
    data = json.loads(make_evaluation_json({"technical_accuracy": "bad"}))
    with pytest.raises(EvaluationFormatError):
        validate_evaluation(data)


def test_json_inside_markdown_fence_is_accepted():
    raw = "Here is my verdict:\n```json\n" + make_evaluation_json() + "\n```"
    assert validate_evaluation(extract_json(raw))["overall_status"] == "PASS"


def test_evaluator_retries_once_after_bad_json_then_succeeds(good_lesson):
    llm = ScriptedLLM({"evaluate": ["garbage", make_evaluation_json()]})
    result = EvaluatorAgent(llm).evaluate("x", good_lesson)
    assert result["overall_status"] == "PASS" and llm.count("evaluate") == 2
    assert "REJECTED BECAUSE" in llm.calls[1][1]  # the second request explains what was wrong


def test_evaluator_that_keeps_returning_garbage_never_approves(memory, out, good_lesson):
    llm = ScriptedLLM({"generate": [good_lesson], "evaluate": ["garbage"]})
    state = run_workflow("x", llm=llm, memory=memory, output_dir=out)
    assert state["attempts"][0]["evaluation"]["overall_status"] == "ERROR"
    assert state["final_status"] == "FAILED"  # an unverifiable lesson is never "approved"
    assert llm.count("regenerate") == 0


# ----------------------------------------------------------------------------- 8. workflow termination (including API failures)
def test_generation_api_failure_terminates_safely(memory, out):
    llm = ScriptedLLM({"generate": [LLMError("API is down")]})
    state = run_workflow("x", llm=llm, memory=memory, output_dir=out)
    assert state["final_status"] == "FAILED" and "API is down" in state["error"]
    assert llm.count("evaluate") == 0
    assert "NOT APPROVED" in (out / "final_lesson.md").read_text()
    assert json.loads((out / "evaluation_report.json").read_text())["approved"] is False


def test_regeneration_api_failure_returns_best_available_lesson(memory, out, bad_lesson):
    llm = ScriptedLLM({"generate": [bad_lesson], "evaluate": [make_evaluation_json(WEIGHTS_FAIL)],
                       "regenerate": [LLMError("rate limit")]})
    state = run_workflow("x", llm=llm, memory=memory, output_dir=out)
    assert state["final_status"] == "FAILED"
    assert len(state["attempts"]) == 1 and state["final_lesson"] == bad_lesson.strip()
    assert "rate limit" in state["error"]


def test_evaluator_api_failure_terminates_without_retry_loop(memory, out, good_lesson):
    llm = ScriptedLLM({"generate": [good_lesson], "evaluate": [LLMError("timeout")]})
    state = run_workflow("x", llm=llm, memory=memory, output_dir=out)
    assert state["final_status"] == "FAILED" and llm.count("evaluate") == 1


def test_memory_is_updated_even_when_the_run_fails(tmp_path):
    memory = MemoryManager(tmp_path / "m.json", tmp_path / "r.json")
    run_workflow("x", llm=ScriptedLLM({"generate": [LLMError("down")]}), memory=memory, output_dir=tmp_path / "o")
    assert memory.load_memory()["total_runs"] == 1


@pytest.mark.parametrize("topic", ["", "   ", "x" * 500])
def test_invalid_topic_is_rejected_before_any_llm_call(memory, out, topic):
    llm = ScriptedLLM({"generate": ["never used"]})
    with pytest.raises(ValueError):
        run_workflow(topic, llm=llm, memory=memory, output_dir=out)
    assert llm.calls == []


# ----------------------------------------------------------------------------- providers / configuration
def test_get_llm_falls_back_to_offline_without_keys(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    assert get_llm().name == "offline"


def test_missing_api_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_llm("anthropic")


def test_unknown_provider_is_rejected():
    with pytest.raises(LLMError):
        get_llm("banana")
