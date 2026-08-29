"""Local-only calendar tools backed by SQLite and an ICS mirror."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from lsm_harness.coding_agent.tools import ToolDefinition


def _ics_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _write_ics(conn: sqlite3.Connection, path: Path) -> None:
    rows = conn.execute(
        'SELECT id,title,start,"end",attendees,notes FROM calendar_events ORDER BY start'
    ).fetchall()
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//LSM Harness//Local Calendar//CN"]
    for row in rows:
        start = datetime.fromisoformat(row["start"]).strftime("%Y%m%dT%H%M%S")
        end = datetime.fromisoformat(row["end"]).strftime("%Y%m%dT%H%M%S")
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:lsm-{row['id']}@local",
                f"DTSTART:{start}",
                f"DTEND:{end}",
                f"SUMMARY:{_ics_escape(row['title'])}",
                f"DESCRIPTION:{_ics_escape(row['notes'] or '')}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def make_tools(conn: sqlite3.Connection, home: Path) -> list[ToolDefinition]:
    def create_event(
        title: str,
        start: str,
        end: str = "",
        attendees: str = "",
        notes: str = "",
    ) -> str:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end) if end else start_dt + timedelta(hours=1)
        normalized_start = start_dt.isoformat(timespec="minutes")
        normalized_end = end_dt.isoformat(timespec="minutes")
        existing = conn.execute(
            "SELECT id FROM calendar_events WHERE title=? AND start=?",
            (title.strip(), normalized_start),
        ).fetchone()
        if existing:
            return f"Event '{title}' at {normalized_start} already exists (not duplicated)."
        conn.execute(
            'INSERT INTO calendar_events(title,start,"end",attendees,notes) VALUES(?,?,?,?,?)',
            (title.strip(), normalized_start, normalized_end, attendees.strip(), notes.strip()),
        )
        conn.commit()
        _write_ics(conn, home / "calendar.ics")
        return (
            f"Event created locally: '{title}' {normalized_start} → {normalized_end}. "
            f"Saved to {home / 'state.db'} and {home / 'calendar.ics'}; not synced externally."
        )

    def list_events(start: str = "", end: str = "", limit: int = 20) -> str:
        query = 'SELECT title,start,"end",attendees FROM calendar_events'
        clauses, params = [], []
        if start:
            clauses.append("start >= ?")
            params.append(start)
        if end:
            clauses.append("start <= ?")
            params.append(end)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY start LIMIT ?"
        params.append(max(1, min(int(limit), 100)))
        rows = conn.execute(query, params).fetchall()
        if not rows:
            return "No local calendar events found."
        return "\n".join(
            f"- {row['start']}–{row['end']} {row['title']}"
            + (f" ({row['attendees']})" if row["attendees"] else "")
            for row in rows
        )

    return [
        ToolDefinition(
            name="create_event",
            label="创建日历事件",
            description="在 LSM 本地日历中创建事件，不会同步到任何外部服务。",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start": {"type": "string", "description": "ISO 8601 时间"},
                    "end": {"type": "string", "description": "ISO 8601；省略则一小时"},
                    "attendees": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["title", "start"],
            },
            execute=create_event,
            effect="local_write",
            execution_mode="sequential",
        ),
        ToolDefinition(
            name="list_events",
            label="读取日历",
            description="读取 LSM 本地日历事件。",
            parameters={
                "type": "object",
                "properties": {
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "limit": {"type": "integer", "default": 20},
                },
            },
            execute=list_events,
            effect="read",
            execution_mode="parallel",
        ),
    ]
