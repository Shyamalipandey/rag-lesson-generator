# Project Design

This document explains *why* the system is built the way it is.

## Architecture diagram

```
 +-----------+      +-----------+      +------------+
 | Streamlit |      |  main.py  |      |   Tests    |
 |  app.py   |      |   (CLI)   |      | (mock LLM) |
 +-----+-----+      +-----+-----+      +-----+------+
       |                  |                  |
       +----------------+------------------+
                        |  run_workflow(topic, demo_mode, llm, memory)
                        v
 +---------------------------------------------------------------------------+
 |                       LangGraph workflow  (workflow/graph.py)             |
 |                                                                           |
 |  [input] -> [generate] -> [evaluate] -> [decision] -approved-> [final_output] -> [memory_update] -> END
 |                 ^   |          ^           |  |                    ^
 |                 |   |          |           |  +-exhausted----------+
 |   API error ----|---+----------|-----------|--------------------->--+
 |                 |              |           | retry (attempt < 3)
 |                 |         [regenerate] <- [save_rejection]
 +-----------------+--------------+-----------+------------------------------+
                   |              |           |
        +----------v--+  +--------v-----+  +--v-----------+
        | Generator   |  | Evaluator    |  | Regeneration |      Agents (agents/)
        | Agent       |  | Agent        |  | Agent        |
        +------+------+  +------+-------+  +------+-------+
               \                |                 /
                +---------------v----------------+
                |   LLM provider (llm/client.py) |   anthropic | openai-compatible | offline
                +--------------------------------+

 +--------------------------------------+      +-------------------------------+
 | Memory manager (memory/)             | <--> | memory.json, rejection_logs.json |
 | reads rules BEFORE generation        |      +-------------------------------+
 | logs rejections, updates AFTER run   |
 +--------------------------------------+
```

## Why an agentic architecture?

A single prompt ("write a lesson") gives one answer with no quality control. The assessment needs a *cycle*: produce, inspect, repair. That cycle is the core idea of an agentic system: separate roles, each with one job, connected by explicit control flow, with the ability to loop based on results. It also makes behaviour **explainable**: at any time we can say which step ran and why.

We keep it modest. There are three agents and eight workflow nodes. We did not add tools, planners or autonomous behaviour that the task does not need.

## Why are the generator and evaluator separate?

* **Different goals.** The generator tries to be helpful and fluent. The evaluator must be sceptical and look for faults. Mixing both goals in one prompt tends to make a model approve its own work.
* **Independent prompts.** The evaluator never sees the generator's reasoning, only the finished lesson.
* **Replaceable.** We can change the evaluator (for example a stronger model or a rule-based pre-check) without touching generation.
* **Testable.** We can test the evaluator alone with known-good and known-bad lessons.

Important honesty point: if both agents use the *same* model, they can share the same blind spots. Separation reduces self-approval, but does not remove it. See "Limitations of LLM-based evaluation".

## Why strict PASS/FAIL evaluation?

* A beginner lesson with one false statement is not "90% good"; it teaches something wrong. A single serious failure must block approval, so there are **no partial marks**.
* Binary decisions make routing simple and unambiguous (approved, retry, stop).
* Every FAIL must contain a *specific* reason, so the regeneration agent knows exactly what to fix and a human can check the reasoning. The validator rejects a FAIL without a real reason.
* The code recomputes the overall status from the individual checks. A model that writes `"overall_status": "PASS"` next to a failed check is overruled (there is a test for this).

## Why LangGraph?

The workflow has a **loop** and several **conditional exits** (approve, retry, exhausted, API error). LangGraph models this directly:

* nodes are small functions that read and write one shared state,
* routing is a visible function attached to a conditional edge,
* the graph is easy to draw and explain, which matches the workflow diagram in the assessment,
* it has a built-in recursion limit that we use as an extra safety guard.

Plain `while` loops would work for three attempts, but a graph makes the decision points explicit and extensible (for example adding a human-approval node later).

## Why retry limits are necessary

An evaluator can keep failing a lesson forever (a strict rubric, a hard topic, a bug, or an evaluator that is simply wrong). Without a limit the system would burn money and time and never return. Three attempts (1 generation + 2 regenerations) is a practical balance: the first rewrite usually fixes concrete, named problems; further rewrites rarely help and risk "fixing" one issue by breaking another.

## How the system prevents endless regeneration

Four independent protections:

1. **Decision node**: retries only if `attempt < 3`.
2. **Regenerate node**: raises an error if it is ever entered with `attempt >= 3` (a safety net that should be unreachable).
3. **Config cap**: `build_graph` uses `min(max_attempts, 3)`, so no caller can raise the limit (tested).
4. **LangGraph `recursion_limit`**: a last guard at the framework level.

Also, API errors route straight to the final-output node instead of looping.

## How persistent memory improves future generations

Memory is *workflow-level learning*:

1. Every rejection is written to `rejection_logs.json` (what failed, why, what was suggested).
2. Before a new run, the memory manager counts how many different real runs each checkpoint failed in.
3. A checkpoint that failed in 2+ runs becomes a **prevention instruction**, for example: "RAG does NOT change the model's weights." It is added to the generation and regeneration prompts together with the last failure reason.
4. After the run, the system records whether the problem **came back** despite the rule, or was **avoided**. That shows whether a rule actually helps.
5. Successful fixes (failed earlier, passed in the final lesson) are stored as `successful_improvements`.

**This is not training.** The model is not fine-tuned and its weights never change. We improve the *prompt* using stored feedback. If memory is deleted, the improvement is gone. Demonstration runs are excluded from the repeated-problem count, because their error is deliberate.

## What happens when the LLM API fails?

Each provider retries transient errors (3 tries in total with growing waits). If it still fails, an `LLMError` is raised and handled by the node that made the call:

| Where it fails | What the system does |
|---|---|
| Generator (attempt 1) | No lesson exists. Go to final output with status `FAILED`, an explanatory banner, and the error message in the report. |
| Evaluator | The evaluation becomes status `ERROR` (never PASS). No retry loop. The lesson is returned as `FAILED` ("could not be verified"). |
| Regeneration | Stop. Return the best lesson so far as `FAILED`, with the error recorded. |

In every case the run ends, files are written, memory is updated, and nothing is labelled approved.

## What happens when all retries fail?

After attempt 3 fails, the final-output node picks the **best available** attempt (fewest failed checkpoints; a later attempt wins a tie). The saved `final_lesson.md` starts with a banner: *"STATUS: FAILED - THIS LESSON IS NOT APPROVED"*, followed by the open problems. `evaluation_report.json` has `"approved": false`. The CLI exits with a non-zero code and the Streamlit page shows a red error. The rejections are in the log, so a human can see exactly what remained wrong.

## Limitations of LLM-based evaluation

* **It can be wrong.** An LLM judge may accept a false statement it does not recognise, or reject a correct one.
* **Shared blind spots.** If the same model writes and judges, it may repeat the same misconception on both sides.
* **Inconsistency.** The same lesson may be judged slightly differently on different runs.
* **Leniency or strictness drift.** Prompts and model versions change behaviour.
* **No ground truth.** The judge relies on its own knowledge; it is not checking against sources.
* **Cost and latency.** Every attempt needs an extra call.

Mitigations in this project: strict prompt that says "do not approve because it is well written", explicit list of facts to verify, validated JSON, code-side overall status, the rule-based offline checker as a second style of evidence, and a deliberate-error demonstration. Mitigations *not* built yet: a different judge model, retrieval-based fact checking, and mandatory human sign-off.

## How a human can review the final output

* Open `outputs/final_lesson.md`. It ends with the evaluation status; failed lessons start with a banner.
* Open `outputs/evaluation_report.json` for every checkpoint, its reason, the attempt history, the provider used and whether the evaluation was rule-based or LLM-based.
* Open `memory/rejection_logs.json` to see what was rejected and why.
* In Streamlit, use the *Evaluation results* and *Rejection log* tabs and the feedback form.
* Compare the lesson to a trusted source. **The evaluation result is evidence, not proof.** A human reviewer, ideally a teacher or AI practitioner, should approve the lesson before learners use it.
