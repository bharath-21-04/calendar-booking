from __future__ import annotations

from datetime import date, datetime, timedelta

from integrations.calendar import CalendarClient
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
    # Use 415 area code (real NANP area) to avoid the 555-block in _validate_phone.
    return [
        {"name": f"Test User {i + 1}", "phone": f"+14150000{i:04d}"}
        for i in range(count)
    ]


def seed() -> None:
    client = CalendarClient()
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

        pre_bookings = _make_fake_bookings(pre_booked)
        event_id = client.create_event(
            title=title,
            start_iso=start_dt.isoformat(),
            end_iso=end_dt.isoformat(),
            timezone=settings.studio_timezone,
            capacity=capacity,
            pre_bookings=pre_bookings,
        )

        status = "FULL" if pre_booked >= capacity else f"{capacity - pre_booked} spots"
        print(f"[seed] Created: {title} on {event_date} — {status}  (id={event_id})")

    print("[seed] Done.")


if __name__ == "__main__":
    seed()
