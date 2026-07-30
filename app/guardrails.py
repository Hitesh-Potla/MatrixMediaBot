import re

from pydantic import BaseModel, Field, field_validator

INJECTION_PATTERNS = (
    r"ignore (all |any |the )?(previous|above|prior) instructions",
    r"reveal (the )?(system|developer) prompt",
    r"you are now",
    r"act as (an? )?(unrestricted|jailbroken)",
)
PII_PATTERNS = (
    r"\b\d{3}-\d{2}-\d{4}\b",  # US SSN
    r"\b(?:\d[ -]*?){13,16}\b",  # likely payment card
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    conversation_id: str | None = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        value = " ".join(value.split())
        if any(re.search(pattern, value, re.IGNORECASE) for pattern in INJECTION_PATTERNS):
            raise ValueError("This request cannot be processed.")
        if any(re.search(pattern, value) for pattern in PII_PATTERNS):
            raise ValueError("Do not send sensitive personal or payment information.")
        return value


def safe_output(answer: str) -> str:
    """Keep output short, remove accidental secrets, and never return blank text."""
    answer = re.sub(r"(?i)(api[_ -]?key|authorization)\s*[:=]\s*\S+", "[redacted]", answer)
    answer = " ".join(answer.split())[:1800]
    return answer or "I could not find a supported answer in the company information."

