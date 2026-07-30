"""Consent-based local follow-up lead storage.

This deliberately uses the Python standard library so lead capture remains
inexpensive and is available in the Docker image without another service.
"""
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
WEBSITE_RE = re.compile(
    r"\b(?:https?://|www\.)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9][a-z0-9-]{0,61})+"
    r"(?::\d{2,5})?(?:/[^\s<>()]*)?", re.IGNORECASE
)
FOLLOW_UP_RE = re.compile(
    r"\b(?:follow[ -]?up|contact me|call me|email me|reach me|connect with me|"
    r"get in touch|please contact|please call|please email)\b", re.IGNORECASE
)
SALES_INTENT_RE = re.compile(
    r"\b(?:interested in (?:your|matrix media(?:'s)?)?\s*(?:services|solutions)|"
    r"(?:interested|keen) (?:to|in) (?:discuss(?:ing)?|explor(?:e|ing)) (?:your )?(?:services|solutions|a project)|"
    r"need (?:a |an )?(?:website|web(?:site)?|mobile app|application|project|quote|proposal)|"
    r"looking for (?:a |an )?(?:website|web(?:site)?|mobile app|application|development|agency|team)|"
    r"want (?:a |an )?(?:website|web(?:site)?|mobile app|application|quote|proposal)|"
    r"start (?:a |an )?(?:project|website|web(?:site)?|mobile app|application)|"
    r"our (?:project|website|web(?:site)?|mobile app|application))\b", re.IGNORECASE
)
CAREER_RE = re.compile(
    r"\b(?:job|jobs|career|careers|hiring|hire|internship|intern|resume|résumé|cv|employment|vacanc\w*|"
    r"open position|work (?:at|for)|developer role)\b", re.IGNORECASE
)
# Deliberately recognise only the unambiguous "my name is ..." form.  Phrases
# such as "I am interested" must never be mistaken for a visitor name.
NAME_RE = re.compile(
    r"\bmy name is\s+([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,4}?)"
    r"(?=\s*(?:,|\.|;|and\s+(?:my\s+)?(?:phone|email)|(?:phone|email)\s|$))", re.IGNORECASE
)


def redact_contact_details(message: str) -> str:
    """Keep personal contact details out of Groq classifier and answer prompts."""
    message = EMAIL_RE.sub("[email redacted]", message)
    message = PHONE_RE.sub("[phone redacted]", message)
    message = WEBSITE_RE.sub("[website redacted]", message)
    return NAME_RE.sub("[name redacted]", message)


def extract_contact_details(message: str) -> tuple[str | None, str | None, str | None]:
    email = next(iter(EMAIL_RE.findall(message)), None)
    phone_match = PHONE_RE.search(message)
    phone = phone_match.group(0).strip() if phone_match else None
    website_match = WEBSITE_RE.search(message)
    website = website_match.group(0).rstrip(".,;:!?") if website_match else None
    return email, phone, website


def extract_name(message: str) -> str | None:
    match = NAME_RE.search(message)
    return match.group(1).strip(" .") if match else None


def is_follow_up_request(message: str) -> bool:
    """Require contact details plus an explicit follow-up or client-sales signal."""
    email, phone, website = extract_contact_details(message)
    has_contact = bool(email or phone or website)
    explicit_follow_up = bool(FOLLOW_UP_RE.search(message))
    sales_interest = bool(SALES_INTENT_RE.search(message)) and not CAREER_RE.search(message)
    return has_contact and (explicit_follow_up or sales_interest)


class FollowUpStore:
    def __init__(self, database_path: str, retention_days: int = 90):
        self.database_path = Path(database_path)
        self.retention_days = retention_days
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS follow_ups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'new',
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    website TEXT,
                    message TEXT NOT NULL
                )
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(follow_ups)")}
            if "name" not in columns:
                connection.execute("ALTER TABLE follow_ups ADD COLUMN name TEXT")
            if "website" not in columns:
                connection.execute("ALTER TABLE follow_ups ADD COLUMN website TEXT")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_follow_ups_created_at ON follow_ups(created_at)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def save_if_requested(self, message: str) -> bool:
        if not is_follow_up_request(message):
            return False
        email, phone, website = extract_contact_details(message)
        name = extract_name(message)
        now = datetime.now(UTC)
        expires_at = now + timedelta(days=self.retention_days)
        with self._connect() as connection:
            connection.execute("DELETE FROM follow_ups WHERE expires_at < ?", (now.isoformat(),))
            cursor = connection.execute(
                "INSERT INTO follow_ups(created_at, expires_at, name, email, phone, website, message) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now.isoformat(), expires_at.isoformat(), name, email, phone, website, message[:2000]),
            )
        # Do not log raw PII or the visitor message. Full records are available
        # only from the protected internal admin endpoint.
        # Uvicorn's default logging configuration does not always emit INFO logs
        # from application modules, so use a deliberately redacted stdout line
        # for the operator-visible confirmation requested for this workflow.
        print(
            "Follow-up saved: "
            f"id={cursor.lastrowid} name={bool(name)} email={bool(email)} "
            f"phone={bool(phone)} website={bool(website)}",
            flush=True,
        )
        return True

    def list_recent(self, limit: int = 100) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, created_at, expires_at, status, name, email, phone, website, message "
                "FROM follow_ups ORDER BY id DESC LIMIT ?", (max(1, min(limit, 500)),)
            ).fetchall()
        return [dict(row) for row in rows]
