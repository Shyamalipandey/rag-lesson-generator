"""The evaluation rubric.

13 strict PASS/FAIL checkpoints: the 12 areas (A-L) required by the assessment,
plus one extra checkpoint (M) that makes sure every required lesson section exists.
Every checkpoint is MANDATORY. There are no partial marks.
"""
from dataclasses import dataclass

# Markers that wrap the lesson inside prompts, so the model can clearly see where it starts and ends.
LESSON_START = "<<<LESSON_START>>>"
LESSON_END = "<<<LESSON_END>>>"


@dataclass(frozen=True)
class Checkpoint:
    id: str
    group: str  # letter from the assessment (A-L, M = extra)
    criterion: str  # human-readable name
    description: str  # what the evaluator must verify
    prevention: str  # instruction added to future generation prompts if this check keeps failing


CHECKPOINTS = [
    Checkpoint(
        "technical_accuracy", "A", "Technical accuracy",
        "Every technical claim is correct. In particular RAG must NOT be described as changing, "
        "updating or retraining the LLM's weights, and RAG must NOT be said to guarantee correct answers.",
        "Double-check every technical claim. RAG retrieves outside information and gives it to the LLM as "
        "context; it does NOT change the model's weights and it does NOT guarantee correct answers.",
    ),
    Checkpoint(
        "beginner_language", "B", "Beginner-friendly language",
        "Simple Indian English, short sentences, everyday words, suitable for a Class 12 learner from a "
        "non-English-medium background.",
        "Use very short sentences (about 15 words), everyday words, and simple Indian English.",
    ),
    Checkpoint(
        "clear_rag_explanation", "C", "Clear explanation of RAG",
        "The lesson clearly defines Retrieval-Augmented Generation: retrieve information from an external "
        "source, then generate the answer using it.",
        "Define RAG clearly: expand the full form, explain Retrieval, Augmented and Generation separately.",
    ),
    Checkpoint(
        "why_rag_needed", "D", "Explanation of why RAG is needed",
        "The lesson explains the limits of a normal LLM (old knowledge, no private data, wrong answers) "
        "and why RAG helps.",
        "Explain why RAG is needed: knowledge cutoff, no access to private documents, and hallucination.",
    ),
    Checkpoint(
        "complete_rag_workflow", "E", "Explanation of the complete RAG workflow",
        "The full pipeline is explained step by step: prepare documents, chunk, embed, store, then "
        "question, embed question, retrieve, build prompt, generate answer.",
        "Explain the COMPLETE RAG workflow in numbered steps, including a simple text diagram.",
    ),
    Checkpoint(
        "embeddings_correct", "F", "Correct explanation of embeddings",
        "Embeddings are explained as numbers (vectors) that represent the meaning of text, with similar "
        "meanings giving nearby numbers, plus a simple example.",
        "Explain embeddings as lists of numbers that capture meaning, with a tiny numeric example.",
    ),
    Checkpoint(
        "vector_database_correct", "G", "Correct explanation of vector databases",
        "A vector database stores embeddings (with the original text) and finds the most similar ones "
        "quickly using similarity search.",
        "Explain that a vector database stores embeddings and finds the most similar ones (similarity search).",
    ),
    Checkpoint(
        "real_world_example", "H", "Practical real-world examples",
        "At least one practical, relatable example (for example a college helpdesk or library) is shown step by step.",
        "Include a step-by-step practical example from an Indian college or library.",
    ),
    Checkpoint(
        "no_unexplained_jargon", "I", "No unexplained technical jargon",
        "Every technical term (LLM, token, prompt, embedding, vector, chunk, hallucination, ...) is explained "
        "at its first use.",
        "Explain every technical term in simple words the first time it appears.",
    ),
    Checkpoint(
        "logical_sequence", "J", "Logical teaching sequence",
        "Topics go from simple to advanced: AI, Generative AI, LLM, RAG, why, how, embeddings, vector "
        "database, retrieval, answer generation, example, pros/cons, comparison, recap, questions.",
        "Keep the teaching order simple-to-advanced and never use a concept before explaining it.",
    ),
    Checkpoint(
        "advantages_limitations", "K", "Advantages and limitations",
        "Both advantages AND honest limitations of RAG are listed (at least three each).",
        "List at least three advantages AND three honest limitations of RAG.",
    ),
    Checkpoint(
        "assessment_questions", "L", "Knowledge assessment questions",
        "Exactly five beginner-friendly questions, each with a correct answer.",
        "End with exactly five beginner-friendly questions, each followed by its answer.",
    ),
    Checkpoint(
        "required_sections_and_comparison", "M", "Required sections, RAG vs normal LLM, and recap",
        "All 15 required sections exist, including a RAG versus normal LLM comparison and a simple recap.",
        "Include all 15 required sections, especially the RAG vs normal LLM comparison and the recap.",
    ),
]

CHECKPOINT_BY_ID = {c.id: c for c in CHECKPOINTS}
CHECKPOINT_IDS = [c.id for c in CHECKPOINTS]


def rubric_text() -> str:
    """The rubric as numbered text, used inside the evaluation prompt."""
    lines = []
    for number, c in enumerate(CHECKPOINTS, start=1):
        lines.append(f'{number}. id="{c.id}" | criterion="{c.criterion}"\n   Verify: {c.description}')
    return "\n".join(lines)
