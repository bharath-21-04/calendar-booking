from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from integrations.calendar import CalendarClient
from integrations.db import BookingDB
from config.settings import get_settings

CLASS_TYPES = ["Reformer", "Mat", "Tower"]
SEED_SCHEDULE = [
    (6, 0, "Reformer", 10, 0),
    (9, 0, "Mat", 8, 8),
    (12, 0, "Tower", 10, 5),
    (17, 0, "Reformer", 10, 10),
    (18, 0, "Mat", 8, 3),
    (19, 0, "Tower", 10, 0),
    (6, 0, "Mat", 8, 0),
    (9, 0, "Reformer", 10, 9),
    (18, 0, "Reformer", 10, 0),
    (7, 0, "Tower", 10, 10),
]

DAY_OFFSETS = [1, 1, 1, 1, 1, 1, 2, 2, 3, 4]


def _make_fake_bookings(count: int) -> list[dict]:
    return [
        {"name": f"Test User {i+1}", "phone": f"+1555000{i:04d}"} for i in range(count)
    ]


def seed() -> None:
    client = CalendarClient()
    db = BookingDB()
    settings = get_settings()
    today = date.today()

    for (hour, minute, cls_type, capacity, pre_booked), day_offset in zip(
        SEED_SCHEDULE, DAY_OFFSETS
    ):
        event_date = today + timedelta(days=day_offset)
        start_dt = datetime(
            event_date.year, event_date.month, event_date.day, hour, minute
        )
        end_dt = start_dt + timedelta(hours=1)

        title = f"{cls_type} — {start_dt.strftime('%-I %p')}"

        existing = client.list_classes(event_date.isoformat())
        if any(e.title == title for e in existing):
            print(f"[seed] Skipping (exists): {title} on {event_date}")
            continue

        event_body = {
            "summary": title,
            "description": json.dumps({"capacity": capacity}),
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": settings.studio_timezone,
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": settings.studio_timezone,
            },
        }

        created = (
            client._service.events()
            .insert(calendarId=settings.google_calendar_id, body=event_body)
            .execute()
        )
        class_id = created["id"]

        for booking in _make_fake_bookings(pre_booked):
            db.add(class_id, booking["name"], booking["phone"])

        status = "FULL" if pre_booked >= capacity else f"{capacity - pre_booked} spots"
        print(f"[seed] Created: {title} on {event_date} ({status})")

    print("[seed] Done.")


if __name__ == "__main__":
    seed()
