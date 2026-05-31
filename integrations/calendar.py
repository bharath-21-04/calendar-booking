from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import google.oauth2.service_account
from google.auth.transport.requests import AuthorizedSession

from config.settings import get_settings

log = logging.getLogger(__name__)

_CALENDAR_API = "https://www.googleapis.com/calendar/v3"


@dataclass
class ClassSlot:
    class_id: str
    title: str
    time: str
    spots_left: int
    capacity: int


class CalendarClient:
    """
    Google Calendar is the single source of truth for class availability and bookings.

    Architecture — two event types, distinguished by extendedProperties.private.type:

    CLASS events  (type=class):
        summary:  "Reformer — 6 AM"
        extendedProperties.private:
            type     = "class"
            capacity = "10"

    BOOKING events  (type=booking):
        summary:  "Booking: <name>"
        extendedProperties.private:
            type    = "booking"
            classId = "<class_event_id>"
            name    = "<caller name>"
            phone   = "<E.164 phone>"
        start/end: same time window as the parent class

    Availability  = capacity - count(booking events where classId = class_id)
    Book          = create a booking event
    Cancel        = delete the matching booking event
    No JSON is stored in event descriptions; no local database is used.
    """

    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(self) -> None:
        settings = get_settings()
        credentials = (
            google.oauth2.service_account.Credentials.from_service_account_info(
                settings.service_account_info,
                scopes=self.SCOPES,
            )
        )
        self._session = AuthorizedSession(credentials)
        self._cal_id = settings.google_calendar_id
        self._default_capacity = settings.class_capacity_default
        self._tz = ZoneInfo(settings.studio_timezone)
        self._tz_name = settings.studio_timezone

    # -- low-level helpers ----------------------------------------------------

    def _events_url(self) -> str:
        return f"{_CALENDAR_API}/calendars/{self._cal_id}/events"

    def _event_url(self, event_id: str) -> str:
        return f"{_CALENDAR_API}/calendars/{self._cal_id}/events/{event_id}"

    def _list_events(self, params: list[tuple[str, str]]) -> list[dict]:
        """Page through events.list using a list-of-tuples for repeated params."""
        all_events: list[dict] = []
        page_token: str | None = None
        while True:
            p = params.copy()
            if page_token:
                p.append(("pageToken", page_token))
            resp = self._session.get(self._events_url(), params=p)
            resp.raise_for_status()
            data = resp.json()
            all_events.extend(data.get("items", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return all_events

    def _get_event(self, event_id: str) -> dict:
        resp = self._session.get(self._event_url(event_id))
        resp.raise_for_status()
        return resp.json()

    def _create_event(self, body: dict) -> dict:
        resp = self._session.post(self._events_url(), json=body)
        resp.raise_for_status()
        return resp.json()

    def _delete_event(self, event_id: str) -> None:
        resp = self._session.delete(self._event_url(event_id))
        resp.raise_for_status()

    def _format_time(self, iso: str) -> str:
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.strftime("%-I:%M %p on %a %b %-d")
        except Exception:
            return iso

    # -- booking sub-event helpers --------------------------------------------

    def _get_booking_events(self, class_id: str) -> list[dict]:
        """Return all booking sub-events linked to a class."""
        return self._list_events(
            [
                ("privateExtendedProperty", f"classId={class_id}"),
                ("privateExtendedProperty", "type=booking"),
            ]
        )

    def _create_booking_event(
        self,
        class_id: str,
        name: str,
        phone: str,
        start_iso: str,
        end_iso: str,
        timezone: str,
        class_title: str,
    ) -> str:
        """Create a booking sub-event and return its event ID."""
        body = {
            "summary": f"{name} | {class_title}",
            "description": f"Name:  {name}\nPhone: {phone}\nClass: {class_title}",
            "start": {"dateTime": start_iso, "timeZone": timezone},
            "end": {"dateTime": end_iso, "timeZone": timezone},
            "extendedProperties": {
                "private": {
                    "type": "booking",
                    "classId": class_id,
                    "name": name,
                    "phone": phone,
                }
            },
        }
        event = self._create_event(body)
        log.info(
            "calendar CREATE booking | id=%s  class=%s  name=%r  phone=%s",
            event["id"],
            class_id,
            name,
            phone,
        )
        return event["id"]

    def _refresh_class_description(self, class_id: str, capacity: int) -> None:
        """Patch the class event description with a human-readable roster."""
        bookings = self._get_booking_events(class_id)
        names = [
            b.get("extendedProperties", {}).get("private", {}).get("name", "?")
            for b in bookings
        ]
        booked = len(names)
        roster = ", ".join(names) if names else "(none yet)"
        desc = f"Capacity: {capacity}  |  Booked: {booked}/{capacity}\nRoster: {roster}"
        resp = self._session.patch(
            self._event_url(class_id), json={"description": desc}
        )
        resp.raise_for_status()
        log.info(
            "calendar PATCH class desc | id=%s  booked=%d/%d",
            class_id,
            booked,
            capacity,
        )

    # -- public API -----------------------------------------------------------

    def create_event(
        self,
        title: str,
        start_iso: str,
        end_iso: str,
        timezone: str,
        capacity: int,
        pre_bookings: list[dict] | None = None,
    ) -> str:
        """Create a class event (and optional pre-bookings). Returns the class event ID."""
        body = {
            "summary": title,
            "start": {"dateTime": start_iso, "timeZone": timezone},
            "end": {"dateTime": end_iso, "timeZone": timezone},
            "extendedProperties": {
                "private": {
                    "type": "class",
                    "capacity": str(capacity),
                }
            },
        }
        event = self._create_event(body)
        class_id = event["id"]
        log.info(
            "calendar CREATE class | id=%s  title=%r  capacity=%d  pre_booked=%d",
            class_id,
            title,
            capacity,
            len(pre_bookings or []),
        )
        for booking in pre_bookings or []:
            self._create_booking_event(
                class_id,
                booking["name"],
                booking["phone"],
                start_iso,
                end_iso,
                timezone,
                title,
            )
        return class_id

    def list_classes(self, date: str) -> list[ClassSlot]:
        """Return ClassSlot list for a given date (YYYY-MM-DD)."""
        d = datetime.strptime(date, "%Y-%m-%d")
        day_start = datetime(d.year, d.month, d.day, tzinfo=self._tz)
        t_min = day_start.isoformat()
        t_max = (day_start + timedelta(days=1)).isoformat()

        # Two API calls: class events + booking events for the whole day.
        class_events = self._list_events(
            [
                ("timeMin", t_min),
                ("timeMax", t_max),
                ("singleEvents", "true"),
                ("orderBy", "startTime"),
                ("privateExtendedProperty", "type=class"),
            ]
        )
        booking_events = self._list_events(
            [
                ("timeMin", t_min),
                ("timeMax", t_max),
                ("singleEvents", "true"),
                ("privateExtendedProperty", "type=booking"),
            ]
        )

        # Count bookings per class_id.
        booking_counts: dict[str, int] = {}
        for b in booking_events:
            cid = b.get("extendedProperties", {}).get("private", {}).get("classId")
            if cid:
                booking_counts[cid] = booking_counts.get(cid, 0) + 1

        slots: list[ClassSlot] = []
        for ev in class_events:
            ext = ev.get("extendedProperties", {}).get("private", {})
            capacity = int(ext.get("capacity", self._default_capacity))
            class_id = ev["id"]
            booked = booking_counts.get(class_id, 0)
            start_raw = ev.get("start", {}).get(
                "dateTime", ev.get("start", {}).get("date", "")
            )
            slots.append(
                ClassSlot(
                    class_id=class_id,
                    title=ev.get("summary", "Class"),
                    time=self._format_time(start_raw),
                    capacity=capacity,
                    spots_left=max(0, capacity - booked),
                )
            )
        return slots

    def check_availability(self, class_id: str) -> int:
        """Return spots remaining for a class."""
        ev = self._get_event(class_id)
        ext = ev.get("extendedProperties", {}).get("private", {})
        capacity = int(ext.get("capacity", self._default_capacity))
        booked = len(self._get_booking_events(class_id))
        return max(0, capacity - booked)

    def book_class(self, class_id: str, name: str, phone: str) -> str:
        """Book a class for a caller. Returns a confirmation string."""
        class_ev = self._get_event(class_id)
        ext = class_ev.get("extendedProperties", {}).get("private", {})
        capacity = int(ext.get("capacity", self._default_capacity))

        existing = self._get_booking_events(class_id)
        if len(existing) >= capacity:
            raise ValueError("Sorry, that class is fully booked.")
        for b in existing:
            if b.get("extendedProperties", {}).get("private", {}).get("phone") == phone:
                return f"You're already booked for {class_ev.get('summary', 'that class')}."

        start_iso = class_ev["start"].get("dateTime", class_ev["start"].get("date", ""))
        end_iso = class_ev["end"].get("dateTime", class_ev["end"].get("date", ""))
        timezone = class_ev["start"].get("timeZone", self._tz_name)
        title = class_ev.get("summary", "Class")

        self._create_booking_event(
            class_id, name, phone, start_iso, end_iso, timezone, title
        )
        self._refresh_class_description(class_id, capacity)
        return f"You're all set! {name} is booked for {title} at {self._format_time(start_iso)}."

    def cancel_booking(self, class_id: str, name: str, phone: str) -> str:
        """Cancel a booking. Returns a confirmation string."""
        bookings = self._get_booking_events(class_id)
        target = next(
            (
                b
                for b in bookings
                if b.get("extendedProperties", {}).get("private", {}).get("phone")
                == phone
            ),
            None,
        )
        if target is None:
            raise ValueError(f"No booking found for {phone} in that class.")

        self._delete_event(target["id"])
        log.info(
            "calendar DELETE booking | id=%s  class=%s  phone=%s",
            target["id"],
            class_id,
            phone,
        )

        class_ev = self._get_event(class_id)
        ext = class_ev.get("extendedProperties", {}).get("private", {})
        capacity = int(ext.get("capacity", self._default_capacity))
        self._refresh_class_description(class_id, capacity)

        title = class_ev.get("summary", "Class")
        start_iso = class_ev["start"].get("dateTime", class_ev["start"].get("date", ""))
        return f"Done! Your booking for {title} at {self._format_time(start_iso)} has been cancelled."

    def reschedule_booking(
        self, old_class_id: str, new_class_id: str, name: str, phone: str
    ) -> str:
        """Move a booking from one class to another."""
        self.cancel_booking(old_class_id, name, phone)
        return self.book_class(new_class_id, name, phone)
