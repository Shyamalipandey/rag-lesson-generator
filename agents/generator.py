"""Generator Agent: writes the first version of the lesson (attempt 1)."""
import re
from pathlib import Path

from agents.prompt_utils import render_template
from config import PROMPTS_DIR
from llm.client import BaseLLM, LLMError

GENERATOR_SYSTEM = (
    "You are an expert teacher of Artificial Intelligence for beginners in India. You write simple, "
    "friendly and technically correct lessons in simple Indian English."
)

# The deliberate mistake used by demonstration mode (taken from the assessment).
DEMO_ERROR_SENTENCE = "RAG means that an LLM permanently learns new information by changing its model weights."

MIN_LESSON_LENGTH = 200


def clean_lesson(text: str) -> str:
    """Remove an outer ```markdown fence if the model added one, and check the answer is not tiny."""
    text = (text or "").strip()
    fenced = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    if len(text) < MIN_LESSON_LENGTH:
        raise LLMError("The model returned a lesson that is far too short.")
    return text


def inject_demo_error(lesson: str) -> str:
    """Insert the wrong claim right under the 'What is RAG?' heading (used only in demonstration mode)."""
    heading = re.search(r"^##\s*(?:\d+\.\s*)?what is rag.*$", lesson, re.I | re.M)
    insertion = f"\n\n{DEMO_ERROR_SENTENCE}"
    if heading:
        return lesson[:heading.end()] + insertion + lesson[heading.end():]
    first_line_end = lesson.find("\n")
    if first_line_end == -1:
        return lesson + insertion
    return lesson[:first_line_end] + insertion + lesson[first_line_end:]


class GeneratorAgent:
    def __init__(self, llm: BaseLLM, prompt_path: Path = PROMPTS_DIR / "generation_prompt.txt"):
        self.llm = llm
        self.prompt_path = prompt_path

    def generate(self, topic: str, prevention_instructions: str = "", demo_mode: bool = False) -> str:
        """Generate the lesson. `prevention_instructions` come from persistent memory."""
        prevention = prevention_instructions.strip() or "No repeated problems have been recorded yet."
        prompt = render_template(self.prompt_path, topic=topic, prevention_instructions=prevention)
        lesson = clean_lesson(self.llm.complete("generate", GENERATOR_SYSTEM, prompt))
        if demo_mode:
            lesson = inject_demo_error(lesson)
        return lesson
