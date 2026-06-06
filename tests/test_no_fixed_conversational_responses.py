from __future__ import annotations

from pathlib import Path

FORBIDDEN = (
    "Yes sir. I am listening.",
    "Yes sir. I can hear you clearly.",
    "I had trouble thinking that through, sir.",
    "I understand, sir.",
    "Certainly, sir.",
    "I found the error.",
    "The tests are running.",
    "One issue remains.",
    "I need your approval before continuing.",
    "Done. The action is complete.",
    "I'm verifying the result now.",
    "I'm working on it.",
    "I noticed something. Would you like help?",
    "I found a build error. Want me to help inspect it?",
    "This task is taking a while. Want me to check its status?",
    "The app may have crashed. Want me to prepare recovery info?",
    "Welcome back. Want me to summarize where you left off?",
    "PID has three terms",
)


ALLOWED_PATH_PARTS = (
    "\\tests\\",
    "/tests/",
    "completion_gate.py",
    "validation.py",
    "smoke",
)


def test_no_fixed_conversational_responses_in_production() -> None:
    violations: list[str] = []

    for path in Path("jarvis").rglob("*.py"):
        normalized = str(path).casefold()
        if any(part.casefold() in normalized for part in ALLOWED_PATH_PARTS):
            continue

        text = path.read_text(encoding="utf-8")
        for phrase in FORBIDDEN:
            if phrase.casefold() in text.casefold():
                violations.append(f"{path}: {phrase!r}")

    assert not violations, "\n".join(violations)