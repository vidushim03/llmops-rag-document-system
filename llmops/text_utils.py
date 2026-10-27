"""Shared heuristics for identifying turns that make no factual claim."""

REFUSAL_MARKERS = (
    "sorry",
    "i couldn't find",
    "i could not",
    "i don't know",
    "i do not know",
)

_FILLER_MARKERS = (
    "let me know",
    "happy to help",
    "how can i help",
    "you're welcome",
    "is there anything",
    "feel free",
    "i'm here to help",
    "would you like me to",
    "what would you like",
    "anything else",
)


def is_conversational_filler(text: str) -> bool:
    """True for short social/follow-up replies that make no factual claim.

    Without this, a turn like "Sure, let me know what you'd like to explore
    next." is scored as an ungrounded claim and counted as a hallucination.
    """
    t = (text or "").strip().lower()
    if not t or len(t.split()) > 30:
        return False
    return any(marker in t for marker in _FILLER_MARKERS)


def is_non_answer(text: str) -> bool:
    """True for refusals, empty responses, and conversational filler."""
    t = (text or "").strip().lower()
    if not t:
        return True
    if t.startswith(REFUSAL_MARKERS):
        return True
    return is_conversational_filler(text)
