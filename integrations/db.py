from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "bookings.db"


class BookingDB:
    def __init__(self, path: str | Path = DB_PATH) -> None:
        self._path = str(path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bookings (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    class_id     TEXT    NOT NULL,
                    caller_name  TEXT    NOT NULL,
                    caller_phone TEXT    NOT NULL,
                    booked_at    TEXT    NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(class_id, caller_phone)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_bookings_class_id ON bookings(class_id)"
            )

    def count(self, class_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM bookings WHERE class_id = ?",
                (class_id,),
            ).fetchone()
        return row["n"] if row else 0

    def get_bookings(self, class_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT caller_name, caller_phone, booked_at FROM bookings WHERE class_id = ? ORDER BY booked_at",
                (class_id,),
            ).fetchall()
        return [
            {
                "name": r["caller_name"],
                "phone": r["caller_phone"],
                "booked_at": r["booked_at"],
            }
            for r in rows
        ]

    def add(self, class_id: str, name: str, phone: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO bookings (class_id, caller_name, caller_phone) VALUES (?, ?, ?)",
                (class_id, name, phone),
            )
        log.info("db.add: booked class_id=%s name=%s phone=%s", class_id, name, phone)

    def remove(self, class_id: str, name: str, phone: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM bookings WHERE class_id = ? AND (caller_name = ? OR caller_phone = ?)",
                (class_id, name, phone),
            )
        removed = cursor.rowcount
        log.info(
            "db.remove: class_id=%s name=%s phone=%s removed=%d",
            class_id,
            name,
            phone,
            removed,
        )
        return removed

    def has_booking(self, class_id: str, phone: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM bookings WHERE class_id = ? AND caller_phone = ?",
                (class_id, phone),
            ).fetchone()
        return row is not None
