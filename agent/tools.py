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
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def _validate_phone(phone: str) -> str | None:
    """Return None if phone looks valid, or an error string if it's clearly invented."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 10:
        return (
            "ERROR: the phone number looks incomplete. "
            "Ask the caller for their full phone number, then try again."
        )
    return None


_PLACEHOLDER_NAMES = frozenset(
    {
        "unknown",
        "unknown caller",
        "unknown user",
        "n/a",
        "na",
        "none",
        "caller",
        "user",
        "customer",
        "client",
        "person",
        "someone",
        "name",
        "full name",
        "your name",
    }
)


def _validate_name(name: str) -> str | None:
    """Return None if name looks real, or an error string if it's a placeholder."""
    if not name or not name.strip():
        return (
            "ERROR: caller_name is empty. "
            "Ask the caller for their full name before calling this tool."
        )
    if name.strip().lower() in _PLACEHOLDER_NAMES:
        return (
            f"ERROR: '{name}' is not a real name. "
            "Ask the caller for their full name, then try again."
        )
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
            day_word = date[len(prefix) :].strip()
            if day_word in _WEEKDAYS:
                target_dow = _WEEKDAYS[day_word]
                days_ahead = (target_dow - now.weekday()) % 7
                if prefix == "next " and days_ahead == 0:
                    days_ahead = 7
                if days_ahead == 0 and prefix != "next ":
                    days_ahead = 0
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
        date: When to look. Accepts ISO dates, month+day, weekday names,
              or relative words. Examples:
                "today", "tomorrow",
                "Monday", "this Friday", "next Saturday",
                "June 1", "06-01", "2026-06-01".
              The current year is assumed when no year is given.

    Returns:
        Human-readable summary of classes and open spots.
        Example: "We have a Reformer at 6 AM with 4 spots, Mat at 9 AM with 8 spots."
    """
    log.info("TOOL list_upcoming_classes | raw_date=%r", date)
    try:
        date = _normalize_date(date)
        log.info("TOOL list_upcoming_classes | normalized=%r", date)
        slots = _calendar.list_classes(date)
        log.info("TOOL list_upcoming_classes | found %d slots", len(slots))
        if not slots:
            return "There are no classes scheduled for that day."
        parts = []
        for s in slots:
            if s.spots_left == 0:
                parts.append(f"{s.title} at {s.time} (id:{s.class_id}) is fully booked")
            else:
                label = "spot" if s.spots_left == 1 else "spots"
                parts.append(
                    f"{s.title} at {s.time} (id:{s.class_id}) has {s.spots_left} {label} left"
                )
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
        Example: "That class has 3 spots left." or "That class is fully booked."
    """
    log.info("TOOL check_class_availability | class_id=%r", class_id)
    try:
        spots = _calendar.check_availability(class_id)
        log.info("TOOL check_class_availability | spots_left=%d", spots)
        if spots == 0:
            return "That class is fully booked."
        label = "spot" if spots == 1 else "spots"
        return f"That class has {spots} {label} left."
    except Exception as exc:
        log.error(
            "check_class_availability failed for %s: %s", class_id, exc, exc_info=True
        )
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
        Example: "You're all set for Reformer at 6 PM on Monday June 1."
    """
    log.info(
        "TOOL book_class | class_id=%r  name=%r  phone=%r",
        class_id,
        caller_name,
        caller_phone,
    )
    if not caller_name or not caller_phone:
        return "ERROR: caller name and phone number are required before booking. Ask the caller for both, then try again."
    name_err = _validate_name(caller_name)
    if name_err:
        return name_err
    phone_err = _validate_phone(caller_phone)
    if phone_err:
        return phone_err
    try:
        confirmation = _calendar.book_class(class_id, caller_name, caller_phone)
        log.info("TOOL book_class | success: %r", confirmation)
        return confirmation
    except ValueError as exc:
        log.warning("TOOL book_class | ValueError: %s", exc)
        return str(exc)
    except Exception as exc:
        log.error("book_class failed for event %s: %s", class_id, exc, exc_info=True)
        return f"ERROR: booking failed — {exc}. Do not retry; tell the caller there was a technical problem."


@tool
def reschedule_booking(
    old_class_id: str,
    new_class_id: str,
    caller_name: str,
    caller_phone: str,
) -> str:
    """Move a caller's existing booking from one class to another.

    Args:
        old_class_id: Event ID of the class to cancel.
        new_class_id: Event ID of the class to book.
        caller_name:  Full name of the caller.
        caller_phone: E.164 phone number.

    Returns:
        Confirmation string on success, or an error string on failure.
    """
    log.info(
        "TOOL reschedule_booking | old=%r  new=%r  name=%r  phone=%r",
        old_class_id,
        new_class_id,
        caller_name,
        caller_phone,
    )
    if not caller_name or not caller_phone:
        return "ERROR: caller name and phone number are required before rescheduling. Ask the caller for both, then try again."
    name_err = _validate_name(caller_name)
    if name_err:
        return name_err
    phone_err = _validate_phone(caller_phone)
    if phone_err:
        return phone_err
    try:
        confirmation = _calendar.reschedule_booking(
            old_class_id, new_class_id, caller_name, caller_phone
        )
        log.info("TOOL reschedule_booking | success: %r", confirmation)
        return confirmation
    except ValueError as exc:
        log.warning("TOOL reschedule_booking | ValueError: %s", exc)
        return str(exc)
    except Exception as exc:
        log.error(
            "reschedule_booking failed (%s->%s): %s",
            old_class_id,
            new_class_id,
            exc,
            exc_info=True,
        )
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
    log.info(
        "TOOL cancel_booking | class_id=%r  name=%r  phone=%r",
        class_id,
        caller_name,
        caller_phone,
    )
    if not caller_name or not caller_phone:
        return "ERROR: caller name and phone number are required before cancelling. Ask the caller for both, then try again."
    name_err = _validate_name(caller_name)
    if name_err:
        return name_err
    phone_err = _validate_phone(caller_phone)
    if phone_err:
        return phone_err
    try:
        confirmation = _calendar.cancel_booking(class_id, caller_name, caller_phone)
        log.info("TOOL cancel_booking | success: %r", confirmation)
        return confirmation
    except ValueError as exc:
        log.warning("TOOL cancel_booking | ValueError: %s", exc)
        return str(exc)
    except Exception as exc:
        log.error(
            "cancel_booking failed for event %s: %s", class_id, exc, exc_info=True
        )
        return f"ERROR: cancellation failed — {exc}. Do not retry; tell the caller there was a technical problem."


@tool
def get_studio_info(topic: str) -> str:
    """Return static studio information based on the topic.

    Args:
        topic: One of "hours", "pricing", "drop_in", "birthday_party", or "general".

    Returns:
        Human-readable studio information string.
    """
    log.info("TOOL get_studio_info | topic=%r", topic)
    info_map = {
        "hours": "We're open Monday through Saturday, 6 AM to 8 PM IST.",
        "pricing": "Drop-in classes are $30, and a 10-class pack is $200.",
        "drop_in": "Drop-ins are welcome at $30 per class — just show up a few minutes early.",
        "birthday_party": "We do offer birthday party bookings — please contact the studio directly for group rates.",
        "general": "Solstice Pilates is open Monday through Saturday, 6 AM to 8 PM. Drop-ins are $30.",
    }
    result = info_map.get(topic, info_map["general"])
    log.info("TOOL get_studio_info | returning info for topic=%r", topic)
    return result


@tool
def log_caller_note(
    caller_phone: str,
    caller_name: str,
    topic: str,
    notes: str,
) -> str:
    """Log or update a caller's record in Google Sheets.

    Args:
        caller_phone: E.164 phone number, e.g. "+15551234567".
        caller_name:  Full name of the caller.
        topic:        Short topic of the call, e.g. "booking" or "cancellation".
        notes:        Any additional notes to store.

    Returns:
        "Logged." on success, or an empty string on failure.
    """
    log.info(
        "TOOL log_caller_note | phone=%r  name=%r  topic=%r",
        caller_phone,
        caller_name,
        topic,
    )
    try:
        _sheets.upsert_caller(caller_phone, caller_name, topic, notes)
        log.info("TOOL log_caller_note | saved to sheets")
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
    inquiries, pricing questions, or general studio information — handle
    those yourself using the available tools.

    IMPORTANT: Before calling this tool you MUST have the caller's name and
    phone number. If you do not have them, ask for them first.

    Args:
        reason:       Short description of why escalation is needed.
        caller_name:  Caller's full name.
        caller_phone: Caller's phone number (any format is fine).

    Returns:
        Internal status string for the LLM (not spoken to the caller).
    """
    log.info(
        "TOOL escalate_to_human | name=%r  phone=%r  reason=%r",
        caller_name,
        caller_phone,
        reason,
    )
    try:
        _sheets.upsert_caller(
            phone=caller_phone or "unknown",
            name=caller_name,
            topic="escalation",
            notes=reason,
        )
        log.info(
            "TOOL escalate_to_human | saved to CRM | name=%s  phone=%s  reason=%r",
            caller_name,
            caller_phone,
            reason,
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
