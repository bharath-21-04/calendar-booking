from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import gspread
from google.oauth2.service_account import Credentials

from config.settings import get_settings


@dataclass
class CallerRecord:
    phone: str
    name: str = ""
    email: str = ""
    first_seen: str = ""
    last_seen: str = ""
    call_count: int = 0
    last_topic: str = ""
    notes: str = ""


COLUMNS = [
    "phone",
    "name",
    "email",
    "first_seen",
    "last_seen",
    "call_count",
    "last_topic",
    "notes",
]


class SheetsClient:
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    WORKSHEET_NAME = "Contacts"

    def __init__(self) -> None:
        settings = get_settings()
        credentials = Credentials.from_service_account_info(
            settings.service_account_info,
            scopes=self.SCOPES,
        )
        client = gspread.Client(auth=credentials)
        spreadsheet = client.open_by_key(settings.google_sheet_id)
        try:
            self._ws = spreadsheet.worksheet(self.WORKSHEET_NAME)
        except gspread.WorksheetNotFound:
            self._ws = spreadsheet.add_worksheet(
                title=self.WORKSHEET_NAME, rows=1000, cols=len(COLUMNS)
            )
        self._ensure_headers()

    def upsert_caller(
        self,
        phone: str,
        name: str,
        topic: str,
        notes: str,
        email: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        # Single get_all_records() call covers both lookup and row-index resolution.
        records = self._ws.get_all_records()
        row_index: int | None = None
        existing_row: dict | None = None
        for i, r in enumerate(records):
            if str(r.get("phone")) == phone:
                row_index = i + 2  # 1-based index, skip header row
                existing_row = r
                break

        if existing_row is not None and row_index is not None:
            existing = self._row_to_record(existing_row)
            updated = CallerRecord(
                phone=phone,
                name=name or existing.name,
                email=email or existing.email,
                first_seen=existing.first_seen,
                last_seen=now,
                call_count=existing.call_count + 1,
                last_topic=topic,
                notes=notes,
            )
            self._ws.update(
                f"A{row_index}:H{row_index}",
                [self._record_to_row(updated)],
            )
        else:
            new_record = CallerRecord(
                phone=phone,
                name=name,
                email=email,
                first_seen=now,
                last_seen=now,
                call_count=1,
                last_topic=topic,
                notes=notes,
            )
            self._ws.append_row(self._record_to_row(new_record))

    def get_caller(self, phone: str) -> Optional[CallerRecord]:
        records = self._ws.get_all_records()
        for row in records:
            if str(row.get("phone")) == phone:
                return self._row_to_record(row)
        return None

    def log_escalation(self, phone: str, name: str, reason: str) -> None:
        self.upsert_caller(phone=phone, name=name, topic="escalation", notes=reason)

    def _ensure_headers(self) -> None:
        first_row = self._ws.row_values(1)
        if not first_row:
            self._ws.insert_row(COLUMNS, index=1)

    def _row_to_record(self, row: dict) -> CallerRecord:
        return CallerRecord(
            phone=str(row.get("phone", "")),
            name=str(row.get("name", "")),
            email=str(row.get("email", "")),
            first_seen=str(row.get("first_seen", "")),
            last_seen=str(row.get("last_seen", "")),
            call_count=int(row.get("call_count", 0) or 0),
            last_topic=str(row.get("last_topic", "")),
            notes=str(row.get("notes", "")),
        )

    def _record_to_row(self, record: CallerRecord) -> list:
        return [
            record.phone,
            record.name,
            record.email,
            record.first_seen,
            record.last_seen,
            record.call_count,
            record.last_topic,
            record.notes,
        ]
