# Self-Evaluating Lesson Content Generator

A small agentic AI system that writes a beginner lesson, **checks it with a strict PASS/FAIL rubric, and rewrites it if it fails**. The submission topic is *Introduction to RAG (Retrieval-Augmented Generation)*, written for a 12th-grade learner from India.

```
INPUT  ->  GENERATE  ->  EVALUATE  ->  REGENERATE (max 2 times)  ->  FINAL OUTPUT
```

> **Honest status note.** The code, tests and workflow in this repository were built and run in a sandbox that had **no LLM API key**. All runs recorded in `outputs/` and `memory/` therefore used the built-in **offline provider** (a hand-written reference lesson plus a rule-based checker), and each report says so (`"llm_provider": "offline"`). To get a lesson written by a real LLM, add an API key (see [API configuration](#api-configuration)) and run `python main.py`. The code paths for the Anthropic and OpenAI providers are written but were **not run against a live API** in the sandbox.

---

## 1. Problem statement

Language models can write fluent lessons, but fluent is not the same as correct. A beginner cannot tell when a lesson contains a wrong claim (for example, "RAG changes the model's weights"), and one-shot generation has no quality control. We need a system that does not simply trust its first answer.

## 2. Project objectives

1. Generate a complete, beginner-friendly RAG lesson from a topic.
2. Evaluate it with at least 12 strict PASS/FAIL checkpoints (no partial marks, specific reasons for every FAIL).
3. Regenerate the lesson from the evaluator's feedback, with a hard limit of 3 attempts in total.
4. Remember past failures in JSON files and use them to improve future prompts (not model training).
5. Prove that the evaluator checks content, using a deliberate-error demonstration.
6. Stay simple, reproducible and testable without paying for API calls.

## 3. Architecture in one picture

```
                    +-------------------------- persistent memory (JSON) --------------------------+
                    | memory.json (summary)                    rejection_logs.json (evidence)      |
                    +-------------^------------------------------------------^---------------------+
                                  | prevention rules (read)                  | log rejection / update
                                  |                                          |
 START -> [input] -> [generate] -> [evaluate] -> [decision] --approved--> [final_output] -> [memory_update] -> END
              topic     Generator      Evaluator       |   \--exhausted-->      ^
                        Agent          Agent           |                        |
                           ^                           |retry (attempt < 3)     |
                           |                           v                        |
                           +-------- [regenerate] <- [save_rejection]           |
                                     Regeneration                               |
                                     Agent -> back to [evaluate]                |
                                                                                |
                   any LLM/API failure --------------------------------------->-+
```

Detailed reasoning is in **[PROJECT_DESIGN.md](PROJECT_DESIGN.md)**.

## 4. Agent responsibilities

| Agent / node | File | Responsibility |
|---|---|---|
| Generator Agent | `agents/generator.py` | Writes attempt 1 from the generation prompt + memory rules. In demo mode, inserts the deliberate mistake. |
| Evaluator Agent | `agents/evaluator.py` | Judges 13 checkpoints (PASS/FAIL only), returns JSON, **validates** it, recomputes the overall status itself, never returns PASS on broken output. |
| Regeneration Agent | `agents/improvement.py` | Receives the rejected lesson, failed checkpoints and suggestions. Refuses to run unless the evaluation is FAIL. |
| Rubric | `agents/checkpoints.py` | The 13 checkpoints (A-L from the assessment + one extra) and their prevention rules. |
| Memory manager | `memory/memory_manager.py` | Reads/writes the two JSON files, finds repeated problems, builds prevention instructions, tracks recurrence. |
| Workflow | `workflow/graph.py` | LangGraph state machine with 8 nodes and conditional routing. |
| Reports | `workflow/reporting.py` | Final lesson file (with a loud banner if FAILED), evaluation report, demo transcript. |
| LLM providers | `llm/client.py`, `llm/offline.py` | Anthropic, OpenAI-compatible, and offline providers behind one small interface. |

### The 13 checkpoints

All are mandatory. One FAIL rejects the lesson.

A Technical accuracy - B Beginner-friendly language - C Clear explanation of RAG - D Why RAG is needed - E Complete RAG workflow - F Embeddings - G Vector databases - H Practical real-world example - I No unexplained jargon - J Logical teaching sequence - K Advantages and limitations - L Assessment questions - M (extra) All 15 sections, RAG vs normal LLM, recap.

## 5. Technology stack

| Purpose | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | Beginner-friendly, standard for AI work |
| Orchestration | LangGraph | Explicit nodes, conditional edges and loops with shared state |
| LLM | Anthropic API (default), OpenAI-compatible API (alternative), offline mode | Real generation, a practical alternative, and a zero-cost test mode |
| UI | Streamlit | A usable page in about 150 lines |
| Storage | JSON files | Human-readable, no database to install |
| Tests | pytest | Simple, with mock LLMs |
| Version control | Git / GitHub | See [GITHUB_UPLOAD_INSTRUCTIONS.md](GITHUB_UPLOAD_INSTRUCTIONS.md) |

## 6. Folder structure

```
rag-lesson-generator/
├── app.py                      Streamlit interface
├── main.py                     Command-line entry point
├── config.py                   Paths and limits (MAX_ATTEMPTS = 3)
├── requirements.txt
├── .env.example                Template for API keys (copy to .env)
├── .gitignore
├── pytest.ini
├── agents/
│   ├── checkpoints.py          The 13-checkpoint rubric
│   ├── generator.py            Generator Agent (+ deliberate-error injection)
│   ├── evaluator.py            Evaluator Agent + JSON validation
│   ├── improvement.py          Regeneration Agent
│   └── prompt_utils.py         {{placeholder}} filling
├── llm/
│   ├── client.py               Anthropic / OpenAI / provider selection, retries
│   ├── offline.py              Offline provider with rule-based evaluator
│   └── reference_lesson.md     Reference lesson used by the offline provider
├── workflow/
│   ├── graph.py                LangGraph workflow
│   └── reporting.py            Output files and demo transcript
├── memory/
│   ├── memory_manager.py       Persistent memory logic
│   ├── memory.json             Long-term memory
│   └── rejection_logs.json     Every rejected attempt
├── prompts/
│   ├── generation_prompt.txt
│   ├── evaluation_prompt.txt
│   └── regeneration_prompt.txt
├── outputs/
│   ├── final_lesson.md         Final lesson (normal run)
│   ├── evaluation_report.json  Final evaluation report
│   └── demo/                   Output of the deliberate-error demonstration
├── tests/test_workflow.py      39 tests, no API key needed
├── README.md
├── PROJECT_DESIGN.md
├── VIDEO_DEMO_SCRIPT.md
└── GITHUB_UPLOAD_INSTRUCTIONS.md
```

## 7. Installation

```bash
git clone <your-repository-url>      # or just download the folder
cd rag-lesson-generator
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 8. API configuration

1. Copy the template: `cp .env.example .env` (Windows: `copy .env.example .env`).
2. Open `.env` and fill in **one** provider:

| Provider | Set these values |
|---|---|
| Anthropic (default) | `ANTHROPIC_API_KEY=...`  and optionally `ANTHROPIC_MODEL=...` |
| OpenAI | `OPENAI_API_KEY=...`  and optionally `OPENAI_MODEL=...` |
| Any OpenAI-compatible server (for example a local Ollama server) | `OPENAI_API_KEY=any-text`, `OPENAI_BASE_URL=http://localhost:11434/v1`, `OPENAI_MODEL=<your model>` |
| No API at all | `LLM_PROVIDER=offline` |

3. With `LLM_PROVIDER` empty, the system picks: Anthropic key, then OpenAI key, then offline. When it falls back to offline, it prints a clear notice, the UI shows a warning, and the report records `"llm_provider": "offline"`.

API keys are read only from environment variables. `.env` is listed in `.gitignore`. Model names in `.env.example` are defaults; use any model your account can access.

## 9. How to run

```bash
# Command line
python main.py                          # default topic, auto provider
python main.py --provider offline       # free, no key
python main.py --demo                   # deliberate-error demonstration
python main.py --topic "Introduction to RAG (Retrieval-Augmented Generation)"
python main.py --feedback 4 "Add one more example"   # store feedback in memory

# Web interface
streamlit run app.py
```

Output files: `outputs/final_lesson.md` and `outputs/evaluation_report.json` (demo mode writes to `outputs/demo/`, including `demo_transcript.md`). The command exits with code 0 only when the lesson is APPROVED.

The Streamlit page has: title and description, topic input, **Generate Lesson** button, PASS/FAIL results, attempt counter, rejection log, final lesson, two download buttons, a demonstration-mode toggle, and a feedback form.

## 10. How to run the tests

```bash
python -m pytest -v
```

The tests use scripted mock LLMs (and the offline provider), so they need no API key and cost nothing. They run in about half a second and use temporary folders, so they never touch your real memory. Results recorded when this repository was built are listed in the final summary of the build (39 passed).

What is covered: lesson generation, the evaluator catching incorrect information (on real text, not mocked), regeneration fixing a failed checkpoint, final approval, the 3-attempt limit, persistent memory (including recurrence tracking), invalid evaluator JSON, API failures at each stage, and workflow termination.

## 11. How regeneration works

1. The evaluator returns FAIL.
2. The decision node checks the attempt number. If `attempt < 3`, it routes to **save_rejection**; otherwise it stops.
3. The rejection (failed checkpoints, reasons, suggestions) is written to `memory/rejection_logs.json`.
4. The Regeneration Agent receives the rejected lesson, every failed checkpoint with its reason, and the suggestions. It writes a new lesson.
5. The new lesson goes back through the evaluator. Every attempt is evaluated.
6. The loop ends at the first PASS, or after attempt 3. If attempt 3 still fails, the system returns the version with the **fewest failed checkpoints** and marks it `FAILED` with a banner at the top of the file.

Counting: the initial generation is attempt 1. At most 2 regenerations follow, so the total can never exceed 3. This limit is enforced in three places (see PROJECT_DESIGN.md).

## 12. How memory works

Two JSON files in `memory/`:

* `rejection_logs.json` - one entry per rejected attempt: run ID, attempt, failed checkpoints, reasons, suggestions.
* `memory.json` - totals, `failure_counts`, `common_mistakes`, `successful_improvements`, `recurrence`, `user_feedback`, and a `runs` history that includes every regeneration attempt.

**Learning loop (before each new run):**
1. Read the rejection log and count, for each checkpoint, in how many *different real runs* it failed.
2. If a checkpoint failed in 2 or more runs (`REPEAT_THRESHOLD` in `config.py`), add a prevention instruction for it, plus the evaluator's last reason, to the generation and regeneration prompts.
3. After the run, record whether that problem came back (`problem_came_back`) or was avoided (`problem_avoided`).
4. Update memory after every completed run, including failed runs.

> **This is not model training.** No weights are changed and nothing is fine-tuned. The system only improves the *instructions* it sends to the model, using stored feedback. If the stored rules are deleted, behaviour returns to the start.

Demonstration runs are logged but **not** counted as repeated problems, because their mistake is planted deliberately and would teach a false lesson.

## 13. Design decisions and trade-offs

| Decision | Reason | Trade-off |
|---|---|---|
| Separate generator and evaluator | A writer is a poor judge of its own work | Extra API calls and cost |
| Strict PASS/FAIL, no partial marks | Clear decisions; one failure must block approval | Harsh: a nearly perfect lesson can still be rejected |
| Overall status recomputed in code | A model must not be able to approve a lesson with a failed check | None worth noting |
| Evaluator error = `ERROR`, never PASS | Unverified content must not look approved | A flaky evaluator can cause a FAILED result |
| Maximum 3 attempts | Bounds cost and time; avoids endless loops | Some lessons might pass on attempt 4 |
| JSON files for memory | Simple, readable, no setup | Not for many users at once; no locking |
| Prompt/workflow learning, not fine-tuning | Cheap, transparent, reversible | Weaker than real training; only as good as the stored rules |
| Offline provider | Tests and demos without cost | Its evaluator is simple (see limitations) |

## 14. Known limitations

* **No live-API run was performed during the build.** Outputs in `outputs/` come from the offline provider. The Anthropic and OpenAI provider code paths are untested against real services.
* The offline evaluator uses patterns and counting. It reliably catches the planted mistake and structural gaps, but it can miss subtle technical errors and can be fooled by rewording.
* An LLM evaluator is better at meaning but can still be wrong, lenient or inconsistent, and it can share blind spots with the generator if both use the same model. See PROJECT_DESIGN.md.
* The lesson uses an imaginary college and an imaginary fee date; it says so inside the lesson.
* Memory is a single local JSON file: no concurrency protection, no per-user separation.
* The rubric is fixed to the RAG topic. Another topic needs a different rubric.

## 15. Future improvements

* Run both a rule-based pre-check and an LLM judge, and require both to agree.
* Use a different model for evaluation than for generation.
* Retrieve real reference text (for example official documentation) so the evaluator can fact-check against sources.
* Add a human-approval step in the Streamlit app before a lesson is marked final.
* Store memory in SQLite, with per-user history.
* Generate lessons in Hindi and other Indian languages, with matching rubrics.
* Track cost and latency per run.

---
Built as a beginner-level GenAI engineering portfolio project.
