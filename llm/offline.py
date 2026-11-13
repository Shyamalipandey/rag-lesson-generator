"""Offline provider: no API key, no cost, fully deterministic.

HONEST DESCRIPTION - please read:
* The "evaluator" here is a RULE-BASED checker (regular expressions and counting). It really does
  read the lesson text, so it genuinely FAILS a lesson that contains the wrong claim "RAG changes the
  model weights". But it is much simpler than an LLM judge and can miss subtle mistakes.
* The "generator" and "regenerator" simply return the hand-written reference lesson stored in
  reference_lesson.md. They show how the workflow behaves; they are NOT real AI generation.

Use a real provider (anthropic / openai) for real lessons. Every report records which provider was used.
"""
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from agents.checkpoints import CHECKPOINTS, CHECKPOINT_BY_ID, LESSON_END, LESSON_START
from llm.client import BaseLLM, LLMError

REFERENCE_LESSON_PATH = Path(__file__).with_name("reference_lesson.md")

NEGATION = re.compile(r"\b(not|no|never|without|cannot|doesn|don|isn|aren)\b|n't", re.I)
CODE_BLOCK = re.compile(r"```.*?```", re.S)

# (key, regex that must match a "## " heading). Order = the required teaching order.
REQUIRED_HEADINGS: List[Tuple[str, str]] = [
    ("ai", r"artificial intelligence"),
    ("genai", r"generative ai"),
    ("llm", r"what is an llm|large language"),
    ("rag", r"what is rag"),
    ("why", r"why do we need"),
    ("how", r"how does rag work"),
    ("embeddings", r"embedding"),
    ("vector_db", r"vector database"),
    ("retrieval", r"what is retrieval"),
    ("answer", r"generate an answer"),
    ("example", r"practical|real-world"),
    ("pros_cons", r"advantages"),
    ("compare", r"\bvs\b|versus"),
    ("recap", r"recap"),
    ("quiz", r"check your|assessment"),
]

# Technical terms that must be explained where they first appear in the text.
JARGON_TERMS = ["model", "token", "weights", "prompt", "hallucination", "embedding", "vector",
                "database", "context", "chunk", "llm"]
DEFINITION_CUE = re.compile(
    r"\b(means|mean|is a|is an|is the|are the|called|stands for|refers to|known as|such as)\b|\(", re.I)
HARD_WORDS = ["furthermore", "moreover", "consequently", "utilize", "utilise", "leverage", "paradigm",
              "methodology", "aforementioned", "heterogeneous", "notwithstanding", "henceforth"]


def load_reference_lesson() -> str:
    return REFERENCE_LESSON_PATH.read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- text helpers
def _sections(lesson: str) -> List[Tuple[str, str]]:
    """Split the lesson into (heading, body) pairs at every '## ' heading."""
    parts = re.split(r"^##\s+(.*)$", lesson, flags=re.M)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def _find_section(sections: List[Tuple[str, str]], pattern: str) -> str:
    for heading, body in sections:
        if re.search(pattern, heading, re.I):
            return body
    return ""


def _sentences(lesson: str) -> List[str]:
    """Prose sentences only: code blocks, headings and table rows are skipped."""
    sentences: List[str] = []
    for line in CODE_BLOCK.sub(" ", lesson).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("|"):
            continue
        line = re.sub(r"^(?:[-*>]+\s+|\d+\.\s+)", "", line)
        line = re.sub(r"[*_`]", "", line)
        sentences += [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
    return sentences


def _bullets(text: str) -> int:
    return len(re.findall(r"^\s*[-*]\s+\S", text, re.M))


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


# ----------------------------------------------------------------------------- the 13 checks
# Each check returns (passed, reason). On FAIL the reason is specific.
def _technical_accuracy(lesson, sections, sentences) -> Tuple[bool, str]:
    problems = []
    for s in sentences:
        if s.endswith("?"):
            continue  # a question is not a claim
        low = s.lower()
        talks_about_rag = re.search(r"\brag\b|retriev", low)
        if (talks_about_rag and re.search(r"\b(weights?|retrain\w*|fine-?tun\w*)\b", low)
                and re.search(r"(chang|updat|modif|permanent|adjust|retrain)", low) and not NEGATION.search(low)):
            problems.append(
                f'Incorrect claim: "{s[:160]}". RAG retrieves relevant external information and supplies it '
                "as context to the LLM. It does not normally update the model's weights.")
        if (re.search(r"\brag\b", low) and re.search(r"\b(guarantee\w*|always|never)\b", low)
                and re.search(r"(correct|accurate|hallucinat|mistake|wrong)", low) and not NEGATION.search(low)):
            problems.append(f'Overclaim: "{s[:160]}". RAG reduces wrong answers but cannot guarantee correctness.')
        if (re.search(r"\b(removes?|eliminates?|stops?|ends?)\b\s+(all\s+|every\s+)?(hallucination|mistake)", low)
                and not NEGATION.search(low)):
            problems.append(f'Overclaim: "{s[:160]}". RAG reduces hallucination; it does not remove it completely.')
    if problems:
        return False, " ".join(problems[:2])
    lower = lesson.lower()
    if not re.search(r"(not|never|without)[^.\n]*weights|weights[^.\n]*(not change|stay the same|same)", lower):
        return False, ("The lesson never states that RAG does not change the LLM's weights. "
                       "This common misconception must be corrected explicitly.")
    return True, "No incorrect technical claim found, and the lesson states that RAG does not change the model weights."


def _beginner_language(lesson, sections, sentences) -> Tuple[bool, str]:
    if not sentences:
        return False, "The lesson has no readable sentences."
    lengths = [_word_count(s) for s in sentences]
    average, longest = sum(lengths) / len(lengths), max(lengths)
    hard = [w for w in HARD_WORDS if w in lesson.lower()]
    if hard:
        return False, f"Difficult words are used: {', '.join(hard)}. Use everyday words."
    if average > 18 or longest > 40:
        return False, (f"Sentences are too long for a beginner (average {average:.1f} words, longest {longest}). "
                       "Use short sentences.")
    return True, f"Sentences are short (average {average:.1f} words) and everyday words are used."


def _clear_rag(lesson, sections, sentences) -> Tuple[bool, str]:
    body = _find_section(sections, r"what is rag").lower()
    missing = [name for name, pat in [("the full form (Retrieval-Augmented Generation)", r"retrieval-augmented generation"),
                                       ("retrieval", r"retriev"), ("generation", r"generat"),
                                       ("an outside/external source", r"outside|external")] if not re.search(pat, body)]
    if not body:
        return False, 'There is no section called "What is RAG?".'
    if missing:
        return False, "The RAG definition is incomplete. Missing: " + "; ".join(missing) + "."
    return True, "RAG is defined with its full form, retrieval, generation and an outside source."


def _why_needed(lesson, sections, sentences) -> Tuple[bool, str]:
    body = _find_section(sections, r"why do we need").lower()
    groups = {"old knowledge / knowledge cutoff": r"cutoff|outdated|latest|old knowledge",
              "private information": r"private|your own|college|company",
              "hallucination / wrong answers": r"hallucinat|wrong"}
    missing = [name for name, pat in groups.items() if not re.search(pat, body)]
    if not body:
        return False, 'There is no section explaining "Why do we need RAG?".'
    if missing:
        return False, "The 'why RAG' section does not explain: " + "; ".join(missing) + "."
    return True, "Old knowledge, missing private information and wrong answers are all explained."


def _workflow(lesson, sections, sentences) -> Tuple[bool, str]:
    body = _find_section(sections, r"how does rag work").lower()
    needed = {"a user question": r"question", "embedding/numbers": r"embedding|numbers", "retrieval/search": r"retriev|search",
              "building a prompt with the chunks": r"prompt", "answer generation": r"generat", "chunking": r"chunk",
              "storing in a vector database": r"vector database"}
    missing = [name for name, pat in needed.items() if not re.search(pat, body)]
    steps = len(re.findall(r"step\s+[a-z0-9]", body))
    if missing or steps < 8 or "```" not in body:
        problems = []
        if missing:
            problems.append("missing " + ", ".join(missing))
        if steps < 8:
            problems.append(f"only {steps} labelled steps found (need the preparation steps and the question-time steps)")
        if "```" not in body:
            problems.append("no simple text diagram")
        return False, "The RAG workflow is incomplete: " + "; ".join(problems) + "."
    return True, "The full workflow (prepare, chunk, embed, store, ask, retrieve, prompt, generate) is explained with a diagram."


def _embeddings(lesson, sections, sentences) -> Tuple[bool, str]:
    body = _find_section(sections, r"embedding")
    low = body.lower()
    missing = [name for name, ok in [("that embeddings are numbers/vectors", "numbers" in low),
                                      ("that they represent meaning", "meaning" in low),
                                      ("that similar texts get similar numbers", "similar" in low),
                                      ("a small numeric example", bool(re.search(r"\[\s*\d", body)))] if not ok]
    if not body:
        return False, 'There is no section "What are embeddings?".'
    if missing:
        return False, "The explanation of embeddings does not show: " + "; ".join(missing) + "."
    return True, "Embeddings are explained as numbers that capture meaning, with a numeric example."


def _vector_db(lesson, sections, sentences) -> Tuple[bool, str]:
    low = _find_section(sections, r"vector database").lower()
    missing = [name for name, pat in [("that it stores embeddings", r"stor\w+.*embedding|embedding.*stor"),
                                       ("similarity search", r"similar"), ("finding the closest/most similar items", r"closest|nearest|most similar")]
               if not re.search(pat, low, re.S)]
    if not low:
        return False, 'There is no section "What is a vector database?".'
    if missing:
        return False, "The explanation of vector databases does not show: " + "; ".join(missing) + "."
    return True, "A vector database is explained as storing embeddings and finding the most similar ones."


def _real_world(lesson, sections, sentences) -> Tuple[bool, str]:
    body = _find_section(sections, r"practical|real-world")
    low = body.lower()
    if _word_count(body) < 120 or not re.search(r"college|library|university|india|exam", low):
        return False, "There is no practical, relatable example (for example a college or library helpdesk) of enough length."
    if not re.search(r"1\.|step", low):
        return False, "The practical example is not explained step by step."
    return True, "A step-by-step college/library example is given."


def _jargon(lesson, sections, sentences) -> Tuple[bool, str]:
    unexplained = []
    for term in JARGON_TERMS:
        first = next((s for s in sentences if re.search(rf"\b{term}s?\b", s, re.I)), None)
        if first and not DEFINITION_CUE.search(first):
            unexplained.append(f'"{term}" (first used in: "{first[:90]}")')
    if unexplained:
        return False, "Technical terms used without an explanation at first use: " + "; ".join(unexplained[:3]) + "."
    return True, "All checked technical terms are explained where they first appear."


def _sequence(lesson, sections, sentences) -> Tuple[bool, str]:
    positions = {}
    for key, pattern in REQUIRED_HEADINGS:
        for index, (heading, _) in enumerate(sections):
            if re.search(pattern, heading, re.I):
                positions[key] = index
                break
    order = [positions[k] for k, _ in REQUIRED_HEADINGS if k in positions]
    if order != sorted(order):
        return False, "The sections are not in a simple-to-advanced order (AI, GenAI, LLM, RAG, why, how, embeddings, vector database, ...)."
    if len(order) < 8:
        return False, "Too many required sections are missing to judge the teaching sequence."
    return True, "Sections follow a logical simple-to-advanced order."


def _pros_cons(lesson, sections, sentences) -> Tuple[bool, str]:
    body = _find_section(sections, r"advantages")
    split = re.split(r"^###\s+limitations.*$", body, flags=re.M | re.I)
    if len(split) < 2:
        return False, "There is no 'Limitations' subsection next to the advantages."
    advantages, limitations = _bullets(split[0]), _bullets(split[1])
    if advantages < 3 or limitations < 3:
        return False, f"Need at least 3 advantages and 3 limitations (found {advantages} and {limitations})."
    if not re.search(r"wrong|poor|bad", split[1], re.I):
        return False, "The limitations do not mention that poor retrieval leads to wrong answers."
    return True, f"{advantages} advantages and {limitations} honest limitations are listed."


def _questions(lesson, sections, sentences) -> Tuple[bool, str]:
    body = _find_section(sections, r"check your|assessment")
    questions = re.findall(r"^\*{0,2}Q[1-9]\d*[.)]", body, re.M)
    answers = len(re.findall(r"^\*{0,2}Answer", body, re.M | re.I))
    if len(questions) != 5 or answers != 5:
        return False, f"Need exactly five questions, each with an answer (found {len(questions)} questions and {answers} answers)."
    return True, "Five questions, each with an answer."


def _required_sections(lesson, sections, sentences) -> Tuple[bool, str]:
    missing = [key for key, pattern in REQUIRED_HEADINGS
               if not any(re.search(pattern, heading, re.I) for heading, _ in sections)]
    if missing:
        return False, "Required sections are missing: " + ", ".join(missing) + "."
    compare = _find_section(sections, r"\bvs\b|versus").lower()
    if "normal" not in compare or "rag" not in compare:
        return False, "The comparison between RAG and a normal LLM is not clear."
    return True, "All 15 required sections are present, including the comparison and the recap."


RULES: Dict[str, Callable] = {
    "technical_accuracy": _technical_accuracy,
    "beginner_language": _beginner_language,
    "clear_rag_explanation": _clear_rag,
    "why_rag_needed": _why_needed,
    "complete_rag_workflow": _workflow,
    "embeddings_correct": _embeddings,
    "vector_database_correct": _vector_db,
    "real_world_example": _real_world,
    "no_unexplained_jargon": _jargon,
    "logical_sequence": _sequence,
    "advantages_limitations": _pros_cons,
    "assessment_questions": _questions,
    "required_sections_and_comparison": _required_sections,
}


def rule_based_evaluation(lesson: str) -> dict:
    """Run all 13 rules and return evaluation JSON in the same shape the LLM evaluator must use."""
    sections = _sections(lesson)
    sentences = _sentences(lesson)
    checks, suggestions = [], []
    for checkpoint in CHECKPOINTS:
        passed, reason = RULES[checkpoint.id](lesson, sections, sentences)
        checks.append({"id": checkpoint.id, "criterion": checkpoint.criterion,
                       "status": "PASS" if passed else "FAIL", "reason": reason})
        if not passed:
            suggestions.append(f"{checkpoint.criterion}: {CHECKPOINT_BY_ID[checkpoint.id].prevention}")
    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    return {"overall_status": overall, "checks": checks, "improvement_suggestions": suggestions}


class OfflineLLM(BaseLLM):
    name = "offline"

    def complete(self, task: str, system: str, prompt: str) -> str:
        if task in ("generate", "regenerate"):
            return load_reference_lesson()
        if task == "evaluate":
            found = re.findall(re.escape(LESSON_START) + r"\n(.*?)\n" + re.escape(LESSON_END), prompt, re.S)
            if not found:
                raise LLMError("Offline evaluator could not find the lesson in the prompt.")
            return json.dumps(rule_based_evaluation(found[-1]), indent=2)
        raise LLMError(f"Offline provider does not know the task '{task}'.")
