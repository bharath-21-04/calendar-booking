from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

import google.oauth2.service_account
import googleapiclient.discovery

from config.settings import get_settings
from integrations.db import BookingDB

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
        self._db = BookingDB()

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
            capacity = self._get_capacity(event.get("description"))
            booked = self._db.count(event["id"])
            start = event.get("start", {}).get(
                "dateTime", event.get("start", {}).get("date", "")
            )
            slots.append(
                ClassSlot(
                    class_id=event["id"],
                    title=event.get("summary", "Class"),
                    time=start,
                    spots_left=max(0, capacity - booked),
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
        capacity = self._get_capacity(event.get("description"))
        return max(0, capacity - self._db.count(class_id))

    def book_class(self, class_id: str, name: str, phone: str) -> str:
        event = (
            self._service.events()
            .get(calendarId=self._calendar_id, eventId=class_id)
            .execute()
        )
        capacity = self._get_capacity(event.get("description"))
        booked = self._db.count(class_id)

        log.info(
            "book_class: event=%s capacity=%d booked=%d",
            class_id,
            capacity,
            booked,
        )

        if booked >= capacity:
            raise ValueError("Class is fully booked.")

        try:
            self._db.add(class_id, name, phone)
        except sqlite3.IntegrityError:
            raise ValueError(f"{name} already has a booking for this class.")

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

        removed = self._db.remove(class_id, name, phone)
        if removed == 0:
            raise ValueError(f"No booking found for {name} / {phone}.")

        title = event.get("summary", "class")
        start = event.get("start", {}).get(
            "dateTime", event.get("start", {}).get("date", "")
        )
        return f"Cancelled: {title} at {self._format_time(start)}"

    def _get_capacity(self, description: str | None) -> int:
        if not description:
            return self._default_capacity
        try:
            return json.loads(description).get("capacity", self._default_capacity)
        except (json.JSONDecodeError, ValueError):
            return self._default_capacity

    def _format_time(self, iso: str) -> str:
        try:
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            return dt.strftime("%I:%M %p on %a %b %-d")
        except Exception:
            return iso
