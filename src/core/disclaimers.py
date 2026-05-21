"""Financial safety disclaimer helpers."""

from __future__ import annotations

EDUCATIONAL_DISCLAIMER = (
    "This is for financial education only and is not financial advice. "
    "Consider consulting a qualified professional for personalized guidance."
)


def append_disclaimer(content: str, disclaimer: str = EDUCATIONAL_DISCLAIMER) -> str:
    """Append the standard educational disclaimer when it is not already present."""

    cleaned = content.strip()
    if not cleaned:
        raise ValueError("content cannot be empty")

    if "not financial advice" in cleaned.lower():
        return cleaned
    return f"{cleaned}\n\n{disclaimer}"


def disclaimer_text() -> str:
    """Return the standard financial safety disclaimer."""

    return EDUCATIONAL_DISCLAIMER


def contains_disclaimer(content: str) -> bool:
    """Check whether text contains the expected safety phrase."""

    return "not financial advice" in content.lower()
