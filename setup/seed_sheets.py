"""
Seed script — populate Google Sheets Contacts tab with sample caller records.

Usage:
    python -m setup.seed_sheets
"""

from __future__ import annotations

from integrations.sheets import SheetsClient

SAMPLE_CALLERS = [
    {
        "phone": "+15550001001",
        "name": "Alice Nguyen",
        "topic": "booking",
        "notes": "Booked Reformer 6 PM on Mon",
        "email": "alice@example.com",
    },
    {
        "phone": "+15550001002",
        "name": "Brian Carter",
        "topic": "reschedule",
        "notes": "Moved from Mat 9 AM to Mat 6 PM",
        "email": "",
    },
    {
        "phone": "+15550001003",
        "name": "Carmen Diaz",
        "topic": "cancel",
        "notes": "Cancelled Tower 12 PM",
        "email": "carmen@example.com",
    },
    {
        "phone": "+15550001004",
        "name": "David Kim",
        "topic": "info",
        "notes": "Asked about drop-in pricing",
        "email": "",
    },
    {
        "phone": "+15550001005",
        "name": "Eva Martins",
        "topic": "escalation",
        "notes": "Requested refund — passed to human",
        "email": "eva@example.com",
    },
]


def seed() -> None:
    """Upsert sample caller records into the Contacts sheet."""
    client = SheetsClient()

    for caller in SAMPLE_CALLERS:
        client.upsert_caller(
            phone=caller["phone"],
            name=caller["name"],
            topic=caller["topic"],
            notes=caller["notes"],
            email=caller["email"],
        )
        print(f"[seed] Upserted: {caller['name']} ({caller['phone']})")

    print("[seed] Sheets seeding done.")


if __name__ == "__main__":
    seed()
