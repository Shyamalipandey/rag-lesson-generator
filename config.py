"""Central settings for the project. Paths and limits live here so every module agrees."""
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Read API keys from a local .env file (if present). Keys are NEVER written in code.
load_dotenv(BASE_DIR / ".env")

PROMPTS_DIR = BASE_DIR / "prompts"
MEMORY_DIR = BASE_DIR / "memory"
OUTPUTS_DIR = BASE_DIR / "outputs"

MEMORY_PATH = MEMORY_DIR / "memory.json"
REJECTION_LOG_PATH = MEMORY_DIR / "rejection_logs.json"

DEFAULT_TOPIC = "Introduction to RAG (Retrieval-Augmented Generation)"
MAX_TOPIC_LENGTH = 200

# Initial generation = attempt 1, plus at most 2 regenerations = 3 attempts in total.
MAX_ATTEMPTS = 3

# A problem becomes a "repeated problem" when it appears in this many different (non-demo) runs.
REPEAT_THRESHOLD = 2

# How many times the evaluator is asked again when its JSON is invalid.
EVALUATOR_FORMAT_RETRIES = 1
