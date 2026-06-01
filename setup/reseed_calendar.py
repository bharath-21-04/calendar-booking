"""
Deletes all seeded class events (and their bookings) then re-seeds with
timezone-aware datetimes so class times match their titles.

Review the dry-run output before answering 'yes'.

Run:  python3 setup/reseed_calendar.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from integrations.calendar import CalendarClient
from config.settings import get_settings


def main() -> None:
    client = CalendarClient()

    
    class_events: list[dict] = []
    for offset in range(1, 8):
        d = (date.today() + timedelta(days=offset)).isoformat()
        slots = client.list_classes(d)
        for slot in slots:
            class_events.append(
                {"id": slot.class_id, "title": slot.title, "date": d, "time": slot.time}
            )

    if not class_events:
        print("No class events found — nothing to delete.")
    else:
        print(f"\nFound {len(class_events)} class events to delete:\n")
        for ev in class_events:
            print(
                f"  [{ev['date']}]  {ev['title']:<28}  stored={ev['time']}  id={ev['id'][:16]}..."
            )

    
    print()
    answer = (
        input("Delete all of the above (and their bookings) and re-seed? [yes/no]: ")
        .strip()
        .lower()
    )
    if answer != "yes":
        print("Aborted.")
        sys.exit(0)

    
    for ev in class_events:
        class_id = ev["id"]
        
        bookings = client._get_booking_events(class_id)
        for b in bookings:
            try:
                client._delete_event(b["id"])
                bname = (
                    b.get("extendedProperties", {}).get("private", {}).get("name", "?")
                )
                print(f"  deleted booking: {bname}  (id={b['id'][:16]}...)")
            except Exception as exc:
                print(f"  WARN: could not delete booking {b['id']}: {exc}")
        
        try:
            client._delete_event(class_id)
            print(f"  deleted class:   {ev['title']}  (id={class_id[:16]}...)")
        except Exception as exc:
            print(f"  WARN: could not delete class {class_id}: {exc}")

    
    print("\nRe-seeding…")
    from setup.seed_calendar import seed

    seed()
    print("\nDone — re-seed complete with correct IST times.")


if __name__ == "__main__":
    main()
