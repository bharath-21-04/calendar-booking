from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import google.oauth2.service_account
import googleapiclient.discovery
from googleapiclient.errors import HttpError

from config.settings import get_settings

log = logging.getLogger(__name__)


@dataclass
class ClassSlot:
    class_id: str
    title: str
    time: str
    spots_left: int
    capacity: int


class CalendarClient:
    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(self) -> None:
        settings = get_settings()
        credentials = (
            google.oauth2.service_account.Credentials.from_service_account_info(
                settings.service_account_info,
                scopes=self.SCOPES,
            )
        )
        self._service = googleapiclient.discovery.build(
            "calendar", "v3", credentials=credentials, cache_discovery=False
        )
        self._calendar_id = settings.google_calendar_id
        self._default_capacity = settings.class_capacity_default

    def list_classes(self, date: str) -> list[ClassSlot]:
        time_min = f"{date}T08:00:00Z"
        dt = datetime.strptime(date, "%Y-%m-%d")
        next_day = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        time_max = f"{next_day}T07:59:59Z"

        result = (
            self._service.events()
            .list(
                calendarId=self._calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        slots = []
        for event in result.get("items", []):
            data = self._parse_description(event.get("description"))
            capacity = data.get("capacity", self._default_capacity)
            bookings = data.get("bookings", [])
            start = event.get("start", {}).get(
                "dateTime", event.get("start", {}).get("date", "")
            )
            slots.append(
                ClassSlot(
                    class_id=event["id"],
                    title=event.get("summary", "Class"),
                    time=start,
                    spots_left=max(0, capacity - len(bookings)),
                    capacity=capacity,
                )
            )
        return slots

    def check_availability(self, class_id: str) -> int:
        event = (
            self._service.events()
            .get(calendarId=self._calendar_id, eventId=class_id)
            .execute()
        )
        data = self._parse_description(event.get("description"))
        capacity = data.get("capacity", self._default_capacity)
        bookings = data.get("bookings", [])
        return max(0, capacity - len(bookings))

    def book_class(self, class_id: str, name: str, phone: str) -> str:
        event = (
            self._service.events()
            .get(calendarId=self._calendar_id, eventId=class_id)
            .execute()
        )
        etag = event.get("etag", "")
        data = self._parse_description(event.get("description"))
        capacity = data.get("capacity", self._default_capacity)
        bookings = data.get("bookings", [])

        log.info(
            "book_class: event=%s capacity=%d existing_bookings=%d",
            class_id,
            capacity,
            len(bookings),
        )

        if len(bookings) >= capacity:
            raise ValueError("Class is fully booked.")

        bookings.append({"name": name, "phone": phone})
        data["bookings"] = bookings

        self._patch_event(class_id, {"description": json.dumps(data)}, etag)

        updated = (
            self._service.events()
            .get(calendarId=self._calendar_id, eventId=class_id)
            .execute()
        )
        saved = self._parse_description(updated.get("description"))
        saved_count = len(saved.get("bookings", []))
        log.info(
            "book_class: after patch bookings_count=%d (expected %d)",
            saved_count,
            len(bookings),
        )
        if saved_count != len(bookings):
            log.error("book_class: patch did not persist! event=%s", class_id)

        title = event.get("summary", "class")
        start = event.get("start", {}).get(
            "dateTime", event.get("start", {}).get("date", "")
        )
        return f"Booked: {title} at {self._format_time(start)}"

    def reschedule_booking(
        self,
        old_class_id: str,
        new_class_id: str,
        name: str,
        phone: str,
    ) -> str:
        self.cancel_booking(old_class_id, name, phone)
        return self.book_class(new_class_id, name, phone)

    def cancel_booking(self, class_id: str, name: str, phone: str) -> str:
        event = (
            self._service.events()
            .get(calendarId=self._calendar_id, eventId=class_id)
            .execute()
        )
        etag = event.get("etag", "")
        data = self._parse_description(event.get("description"))
        bookings = data.get("bookings", [])

        log.info(
            "cancel_booking: event=%s existing_bookings=%d name=%s phone=%s",
            class_id,
            len(bookings),
            name,
            phone,
        )

        updated = [
            b for b in bookings if b.get("name") != name and b.get("phone") != phone
        ]
        if len(updated) == len(bookings):
            raise ValueError(f"No booking found for {name} / {phone}.")

        data["bookings"] = updated
        self._patch_event(class_id, {"description": json.dumps(data)}, etag)
        log.info(
            "cancel_booking: removed %d booking(s), now %d remain",
            len(bookings) - len(updated),
            len(updated),
        )

        title = event.get("summary", "class")
        start = event.get("start", {}).get(
            "dateTime", event.get("start", {}).get("date", "")
        )
        return f"Cancelled: {title} at {self._format_time(start)}"

    def _parse_description(self, description: str | None) -> dict:

        if not description:
            return {"capacity": self._default_capacity, "bookings": []}
        try:
            return json.loads(description)
        except (json.JSONDecodeError, ValueError):
            return {"capacity": self._default_capacity, "bookings": []}

    def _patch_event(self, event_id: str, body: dict, etag: str) -> dict:
        request = self._service.events().patch(
            calendarId=self._calendar_id,
            eventId=event_id,
            body=body,
        )
        if etag:
            request.headers["If-Match"] = etag
        try:
            result = request.execute()
            log.debug(
                "_patch_event: success event=%s new_etag=%s",
                event_id,
                result.get("etag", ""),
            )
            return result
        except HttpError as exc:
            if exc.resp.status == 412:
                log.warning(
                    "_patch_event: 412 ETag mismatch for %s, retrying", event_id
                )
                fresh = (
                    self._service.events()
                    .get(calendarId=self._calendar_id, eventId=event_id)
                    .execute()
                )
                retry = self._service.events().patch(
                    calendarId=self._calendar_id,
                    eventId=event_id,
                    body=body,
                )
                retry.headers["If-Match"] = fresh.get("etag", "")
                return retry.execute()
            log.error(
                "_patch_event: HttpError %s for event=%s: %s",
                exc.resp.status,
                event_id,
                exc,
            )
            raise

    def _format_time(self, iso: str) -> str:
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.strftime("%I:%M %p on %a %b %-d")
        except Exception:
            return iso
