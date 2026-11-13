"""Tiny helper to fill {{placeholders}} in the prompt text files.

We use {{double braces}} (and plain str.replace) so the JSON examples inside the prompts,
which contain single braces, are never confused with placeholders.
"""
from pathlib import Path


def render_template(path: Path, **values: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text
