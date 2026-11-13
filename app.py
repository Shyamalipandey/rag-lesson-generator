"""Streamlit interface. Run with:  streamlit run app.py"""
import json

import streamlit as st

from config import DEFAULT_TOPIC, MAX_ATTEMPTS, OUTPUTS_DIR
from llm.client import LLMError, get_llm
from memory.memory_manager import MemoryManager
from workflow.graph import run_workflow
from workflow.reporting import build_report

st.set_page_config(page_title="Self-Evaluating Lesson Generator", page_icon="📘", layout="wide")

memory = MemoryManager()

# ----------------------------------------------------------------------------- header
st.title("📘 Self-Evaluating Lesson Content Generator")
st.write(
    "Type a topic. An AI agent writes a beginner-friendly lesson. A **separate** agent checks it with strict "
    "PASS/FAIL checkpoints. If any checkpoint fails, the lesson is rewritten (at most **2 times**, so "
    f"**{MAX_ATTEMPTS} attempts in total**). Nothing is called *approved* unless every checkpoint passes."
)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    provider_choice = st.selectbox("LLM provider", ["auto", "anthropic", "openai", "offline"],
                                   help="'auto' uses the API key found in your .env file. 'offline' needs no key.")
    demo_mode = st.toggle("Demonstration mode", value=False,
                          help="Inserts a deliberate technical mistake into attempt 1 to prove that the evaluator really checks the content.")
    st.caption(f"Maximum attempts: {MAX_ATTEMPTS} (1 generation + 2 regenerations)")
    st.divider()
    st.subheader("What memory has learned")
    counts = memory.repeated_problems()
    if counts:
        for check_id, number in sorted(counts.items(), key=lambda item: -item[1]):
            st.write(f"- `{check_id}` was rejected in {number} run(s)")
    else:
        st.write("No repeated problems recorded yet.")
    st.caption("Memory improves the next prompt. It does not train or change the LLM.")

# ----------------------------------------------------------------------------- input
topic = st.text_input("Lesson topic", value=DEFAULT_TOPIC)
generate_clicked = st.button("Generate Lesson", type="primary")

if generate_clicked:
    try:
        llm = get_llm(None if provider_choice == "auto" else provider_choice)
    except LLMError as exc:
        st.error(f"Configuration problem: {exc}")
        st.stop()
    output_dir = OUTPUTS_DIR / "demo" if demo_mode else OUTPUTS_DIR
    with st.spinner("Generating, evaluating and (if needed) regenerating..."):
        try:
            st.session_state["result"] = run_workflow(topic, demo_mode=demo_mode, llm=llm, memory=memory, output_dir=output_dir)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

result = st.session_state.get("result")
if not result:
    st.info("Enter a topic and click **Generate Lesson**.")
    st.stop()

# ----------------------------------------------------------------------------- results
approved = result["final_status"] == "APPROVED"
attempts = result["attempts"]

if result["provider"] == "offline":
    st.warning("Offline provider: the lesson is a hand-written reference lesson and the evaluator is a simple "
               "rule-based checker. Set an API key in `.env` to use a real LLM.")
if result.get("demo_mode"):
    st.info("Demonstration mode was ON: a wrong sentence was inserted into attempt 1 on purpose.")

if approved:
    st.success("✅ APPROVED - every checkpoint passed.")
else:
    st.error("❌ FAILED - the lesson did NOT pass all checkpoints. The best available version is shown below. "
             "A human must review it.")
    if result.get("error"):
        st.error(result["error"])

col1, col2, col3 = st.columns(3)
col1.metric("Attempts used", f"{len(attempts)} / {MAX_ATTEMPTS}")
col2.metric("Regenerations", max(0, len(attempts) - 1))
col3.metric("Final status", result["final_status"])

tab_lesson, tab_eval, tab_log, tab_events = st.tabs(
    ["📖 Final lesson", "🧪 Evaluation results", "📝 Rejection log", "🔁 Workflow steps"])

with tab_lesson:
    if not approved:
        st.warning("This is the best available version, but it is NOT approved.")
    st.markdown(result["final_lesson"] or "_No lesson was generated._")
    lesson_text = open(result["output_paths"]["lesson"], encoding="utf-8").read()
    report_text = json.dumps(build_report(result), indent=2, ensure_ascii=False)
    download1, download2 = st.columns(2)
    download1.download_button("⬇️ Download lesson (.md)", lesson_text, file_name="final_lesson.md", mime="text/markdown")
    download2.download_button("⬇️ Download evaluation report (.json)", report_text,
                              file_name="evaluation_report.json", mime="application/json")

with tab_eval:
    for attempt in attempts:
        evaluation = attempt["evaluation"]
        label = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "ERROR": "⚠️ ERROR"}[evaluation["overall_status"]]
        with st.expander(f"Attempt {attempt['attempt']} - {label}", expanded=(attempt["attempt"] == len(attempts))):
            if evaluation["overall_status"] == "ERROR":
                st.error(evaluation["error"])
                continue
            for check in evaluation["checks"]:
                icon = "✅" if check["status"] == "PASS" else "❌"
                st.markdown(f"{icon} **{check['criterion']}** - {check['status']}")
                if check["status"] == "FAIL":
                    st.caption(f"Reason: {check['reason']}")
            if evaluation["improvement_suggestions"]:
                st.markdown("**Improvement suggestions**")
                for suggestion in evaluation["improvement_suggestions"]:
                    st.write(f"- {suggestion}")

with tab_log:
    rejected = [a for a in attempts if a["evaluation"]["overall_status"] == "FAIL"]
    if not rejected:
        st.write("No attempt was rejected in this run.")
    for attempt in rejected:
        st.markdown(f"**Attempt {attempt['attempt']} was rejected**")
        for check in attempt["evaluation"]["checks"]:
            if check["status"] == "FAIL":
                st.write(f"- ❌ {check['criterion']}: {check['reason']}")
    with st.expander("All saved rejections (memory/rejection_logs.json)"):
        st.json(memory.load_rejections()[-20:])

with tab_events:
    for line in result["events"]:
        st.write(f"- {line}")

# ----------------------------------------------------------------------------- feedback
st.divider()
st.subheader("Your feedback (saved to memory)")
with st.form("feedback_form", clear_on_submit=True):
    rating = st.slider("How good was this lesson?", 1, 5, 4)
    comment = st.text_input("Comment (optional)")
    if st.form_submit_button("Save feedback"):
        memory.add_user_feedback(rating, comment, run_id=result["run_id"])
        st.success("Thank you. Feedback saved.")
