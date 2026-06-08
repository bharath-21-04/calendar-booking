import logging
import re
from datetime import datetime, timedelta

from langchain_core.tools import tool

from integrations.calendar import CalendarClient
from integrations.sheets import SheetsClient

log = logging.getLogger(__name__)

_calendar = CalendarClient()
_sheets = SheetsClient()

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

_SPOKEN_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "oh": "0",
}

_PLACEHOLDER_NAMES = frozenset({
    "unknown", "unknown caller", "unknown user", "n/a", "na", "none",
    "caller", "user", "customer", "client", "person", "someone",
    "name", "full name", "your name",
})


def _normalize_phone(phone: str) -> str:
    if not phone:
        return phone
    tokens = re.split(r"[\s,]+", phone.lower())
    parts = []
    any_word_matched = False
    for tok in tokens:
        tok_clean = re.sub(r"[^\w]", "", tok)
        if tok_clean in _SPOKEN_DIGIT:
            parts.append(_SPOKEN_DIGIT[tok_clean])
            any_word_matched = True
        else:
            parts.append(tok_clean)
    if any_word_matched:
        phone = "".join(parts)
    return re.sub(r"(?!^\+)[^\d]", "", phone)


def _validate_phone(phone: str) -> str | None:
    if len(re.sub(r"\D", "", phone)) < 10:
        return "ERROR: the phone number looks incomplete. Ask the caller for their full phone number, then try again."
    return None


def _validate_name(name: str) -> str | None:
    if not name or not name.strip():
        return "ERROR: caller_name is empty. Ask the caller for their full name before calling this tool."
    if name.strip().lower() in _PLACEHOLDER_NAMES:
        return f"ERROR: '{name}' is not a real name. Ask the caller for their full name, then try again."
    return None


def _normalize_date(date: str) -> str:
    original = date
    date = date.strip().lower()
    now = datetime.now()
    current_year = now.year

    if date == "today":
        return now.strftime("%Y-%m-%d")
    if date in ("tomorrow", "tmrw"):
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")

    for prefix in ("next ", "this ", ""):
        if date.startswith(prefix):
            day_word = date[len(prefix):].strip()
            if day_word in _WEEKDAYS:
                target_dow = _WEEKDAYS[day_word]
                days_ahead = (target_dow - now.weekday()) % 7
                if prefix == "next " and days_ahead == 0:
                    days_ahead = 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    date = original.strip()

    if re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        parsed_year = int(date[:4])
        if parsed_year < current_year:
            date = f"{current_year}{date[4:]}"
        return date

    m = re.match(r"^(\d{1,2})[-/](\d{1,2})$", date)
    if m:
        return f"{current_year}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"

    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%B %d", "%b %d"):
        for candidate in (original.strip(), date.title()):
            try:
                dt = datetime.strptime(candidate, fmt)
                year = dt.year if dt.year != 1900 else current_year
                return f"{year}-{dt.month:02d}-{dt.day:02d}"
            except ValueError:
                continue

    log.warning("_normalize_date: could not parse %r, using as-is", original)
    return original


@tool
def list_upcoming_classes(date: str) -> str:
    """List all classes and their availability for a given date.

    Args:
        date: When to look. Accepts ISO dates (2026-06-01), month+day (June 1),
              weekday names (Monday, this Friday, next Saturday), or relative words
              (today, tomorrow). Current year assumed when no year is given.

    Returns:
        Human-readable summary of classes and open spots.
    """
    log.info("TOOL list_upcoming_classes | raw_date=%r", date)
    try:
        date = _normalize_date(date)
        slots = _calendar.list_classes(date)
        if not slots:
            return "There are no classes scheduled for that day."
        parts = []
        for s in slots:
            if s.spots_left == 0:
                parts.append(f"{s.title} at {s.time} (id:{s.class_id}) is fully booked")
            else:
                label = "spot" if s.spots_left == 1 else "spots"
                parts.append(f"{s.title} at {s.time} (id:{s.class_id}) has {s.spots_left} {label} left")
        return "Here's what we have: " + ", ".join(parts) + "."
    except Exception as exc:
        log.error("list_upcoming_classes failed: %s", exc, exc_info=True)
        return f"ERROR: could not fetch schedule — {exc}"


@tool
def check_class_availability(class_id: str) -> str:
    """Check how many spots are left for a specific class.

    Args:
        class_id: Google Calendar event ID.

    Returns:
        Human-readable availability string.
    """
    try:
        spots = _calendar.check_availability(class_id)
        if spots == 0:
            return "That class is fully booked."
        label = "spot" if spots == 1 else "spots"
        return f"That class has {spots} {label} left."
    except Exception as exc:
        log.error("check_class_availability failed for %s: %s", class_id, exc, exc_info=True)
        return f"ERROR: could not check availability — {exc}"


@tool
def book_class(class_id: str, caller_name: str, caller_phone: str) -> str:
    """Book a class for the caller.

    Args:
        class_id:     Google Calendar event ID.
        caller_name:  Full name of the caller.
        caller_phone: E.164 phone number, e.g. "+15551234567".

    Returns:
        Confirmation string on success, or an error string if fully booked.
    """
    if not caller_name or not caller_phone:
        return "ERROR: caller name and phone number are required before booking. Ask the caller for both, then try again."
    caller_phone = _normalize_phone(caller_phone)
    if err := _validate_name(caller_name):
        return err
    if err := _validate_phone(caller_phone):
        return err
    try:
        return _calendar.book_class(class_id, caller_name, caller_phone)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        log.error("book_class failed for event %s: %s", class_id, exc, exc_info=True)
        return f"ERROR: booking failed — {exc}. Do not retry; tell the caller there was a technical problem."


@tool
def reschedule_booking(old_class_id: str, new_class_id: str, caller_name: str, caller_phone: str) -> str:
    """Move a caller's existing booking from one class to another.

    Args:
        old_class_id: Event ID of the class to cancel.
        new_class_id: Event ID of the class to book.
        caller_name:  Full name of the caller.
        caller_phone: E.164 phone number.

    Returns:
        Confirmation string on success, or an error string on failure.
    """
    if not caller_name or not caller_phone:
        return "ERROR: caller name and phone number are required before rescheduling. Ask the caller for both, then try again."
    caller_phone = _normalize_phone(caller_phone)
    if err := _validate_name(caller_name):
        return err
    if err := _validate_phone(caller_phone):
        return err
    try:
        return _calendar.reschedule_booking(old_class_id, new_class_id, caller_name, caller_phone)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        log.error("reschedule_booking failed (%s->%s): %s", old_class_id, new_class_id, exc, exc_info=True)
        return f"ERROR: reschedule failed — {exc}. Do not retry; tell the caller there was a technical problem."


@tool
def cancel_booking(class_id: str, caller_name: str, caller_phone: str) -> str:
    """Cancel a caller's existing booking for a class.

    Args:
        class_id:     Google Calendar event ID.
        caller_name:  Full name of the caller.
        caller_phone: E.164 phone number, e.g. "+15551234567".

    Returns:
        Confirmation string on success, or an error string if booking not found.
    """
    if not caller_name or not caller_phone:
        return "ERROR: caller name and phone number are required before cancelling. Ask the caller for both, then try again."
    caller_phone = _normalize_phone(caller_phone)
    if err := _validate_name(caller_name):
        return err
    if err := _validate_phone(caller_phone):
        return err
    try:
        return _calendar.cancel_booking(class_id, caller_name, caller_phone)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        log.error("cancel_booking failed for event %s: %s", class_id, exc, exc_info=True)
        return f"ERROR: cancellation failed — {exc}. Do not retry; tell the caller there was a technical problem."


@tool
def get_studio_info(topic: str) -> str:
    """Return static studio information based on the topic.

    Args:
        topic: One of "hours", "pricing", "drop_in", "birthday_party", or "general".

    Returns:
        Human-readable studio information string.
    """
    info_map = {
        "hours": "We're open Monday through Saturday, 6 AM to 8 PM IST.",
        "pricing": "Drop-in classes are $30, and a 10-class pack is $200.",
        "drop_in": "Drop-ins are welcome at $30 per class — just show up a few minutes early.",
        "birthday_party": "We do offer birthday party bookings — please contact the studio directly for group rates.",
        "general": "Solstice Pilates is open Monday through Saturday, 6 AM to 8 PM. Drop-ins are $30.",
    }
    return info_map.get(topic, info_map["general"])


@tool
def log_caller_note(caller_phone: str, caller_name: str, topic: str, notes: str) -> str:
    """Log or update a caller's record in Google Sheets.

    Args:
        caller_phone: E.164 phone number, e.g. "+15551234567".
        caller_name:  Full name of the caller.
        topic:        Short topic of the call, e.g. "booking" or "cancellation".
        notes:        Any additional notes to store.

    Returns:
        "Logged." on success, or an empty string on failure.
    """
    caller_phone = _normalize_phone(caller_phone)
    try:
        _sheets.upsert_caller(caller_phone, caller_name, topic, notes)
        return "Logged."
    except Exception as exc:
        log.error("log_caller_note failed: %s", exc, exc_info=True)
        return ""


@tool
def escalate_to_human(reason: str, caller_name: str, caller_phone: str) -> str:
    """Save caller details to the CRM and trigger an immediate live phone transfer.

    ONLY call this tool for:
    - Billing complaints or charge disputes
    - Refund requests
    - Aggressive or abusive callers

    NEVER call this tool for bookings, cancellations, rescheduling, class
    inquiries, pricing questions, or general studio information.

    IMPORTANT: You MUST have the caller's name and phone before calling this tool.

    Args:
        reason:       Short description of why escalation is needed.
        caller_name:  Caller's full name.
        caller_phone: Caller's phone number (any format).

    Returns:
        Internal status string for the LLM (not spoken to the caller).
    """
    caller_phone = _normalize_phone(caller_phone or "")
    try:
        _sheets.upsert_caller(
            phone=caller_phone or "unknown",
            name=caller_name,
            topic="escalation",
            notes=reason,
        )
    except Exception as exc:
        log.error("escalate_to_human: sheets write failed: %s", exc, exc_info=True)

    name_part = f" {caller_name.split()[0]}" if caller_name else ""
    return (
        f"Saved.{name_part}'s details ({caller_phone}) and reason ({reason!r}) are "
        "logged. Now say exactly: 'Let me transfer you to our team now — please hold.' "
        "The call will be transferred immediately after you say that."
    )


ALL_TOOLS = [
    list_upcoming_classes,
    check_class_availability,
    book_class,
    reschedule_booking,
    cancel_booking,
    get_studio_info,
    log_caller_note,
    escalate_to_human,
]
