"""Command-line entry point.

Examples
    python main.py                                  # normal run with the default topic
    python main.py --demo                           # deliberate-error demonstration
    python main.py --provider offline               # no API key needed (reference lesson + rule-based checker)
    python main.py --feedback 4 "Good, but add one more example"   # store user feedback in memory
"""
import argparse
import sys

from config import DEFAULT_TOPIC, OUTPUTS_DIR
from llm.client import LLMError, get_llm
from memory.memory_manager import MemoryManager
from workflow.graph import run_workflow


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Self-Evaluating Lesson Content Generator")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Lesson topic")
    parser.add_argument("--demo", action="store_true", help="Insert a deliberate technical error into attempt 1")
    parser.add_argument("--provider", default=None, help="anthropic | openai | offline (default: auto)")
    parser.add_argument("--output-dir", default=None, help="Where to write the lesson and report")
    parser.add_argument("--feedback", nargs=2, metavar=("RATING_1_TO_5", "COMMENT"), help="Save user feedback and exit")
    args = parser.parse_args(argv)

    memory = MemoryManager()
    if args.feedback:
        memory.add_user_feedback(int(args.feedback[0]), args.feedback[1])
        print("Feedback saved to memory.")
        return 0

    try:
        llm = get_llm(args.provider)
    except LLMError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2
    if llm.name == "offline":
        print("NOTE: offline provider in use (reference lesson + rule-based checker). "
              "Set an API key in .env for real LLM generation.\n")

    output_dir = args.output_dir or (OUTPUTS_DIR / "demo" if args.demo else OUTPUTS_DIR)
    try:
        state = run_workflow(args.topic, demo_mode=args.demo, llm=llm, memory=memory, output_dir=output_dir)
    except ValueError as exc:  # for example an empty topic
        print(f"Input problem: {exc}", file=sys.stderr)
        return 2

    for line in state["events"]:
        print(line)
    print(f"\nFINAL STATUS: {state['final_status']}  |  attempts used: {len(state['attempts'])}/3")
    for name, path in state["output_paths"].items():
        print(f"  {name}: {path}")
    return 0 if state["final_status"] == "APPROVED" else 1


if __name__ == "__main__":
    sys.exit(main())
