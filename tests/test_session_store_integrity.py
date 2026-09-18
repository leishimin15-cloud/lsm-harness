"""Strict JSONL loading repairs only a torn tail."""

from __future__ import annotations

import json

import pytest

from lsm_harness.ops.session_store import (
    SessionFileError,
    read_session_entries,
    read_session_header,
)


def _header() -> str:
    return json.dumps({"id": "s", "type": "session", "version": 2})


def _message(entry_id: str) -> str:
    return json.dumps({
        "id": entry_id,
        "type": "message",
        "parent_id": "s",
        "message": {"role": "user", "content": "hello"},
    })


def test_torn_final_record_is_removed_atomically(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(_header() + "\n" + _message("m1") + "\n" + '{"id":', encoding="utf-8")

    entries = read_session_entries(path)

    assert [entry.id for entry in entries] == ["s", "m1"]
    assert path.read_text(encoding="utf-8") == _header() + "\n" + _message("m1") + "\n"


def test_corruption_before_tail_is_never_skipped(tmp_path):
    path = tmp_path / "session.jsonl"
    original = _header() + "\nnot-json\n" + _message("m1") + "\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(SessionFileError, match="line 2"):
        read_session_entries(path)

    assert path.read_text(encoding="utf-8") == original


def test_valid_unterminated_tail_gets_newline(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(_header(), encoding="utf-8")

    assert [entry.id for entry in read_session_entries(path)] == ["s"]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_malformed_header_is_not_repaired_away(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text('{"id":', encoding="utf-8")

    with pytest.raises(SessionFileError, match="line 1"):
        read_session_entries(path)


def test_header_reader_does_not_parse_the_conversation_body(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps({
            "id": "s",
            "type": "session",
            "version": 2,
            "cwd": "/workspace/project",
        })
        + "\nnot-json\n",
        encoding="utf-8",
    )

    header = read_session_header(path)

    assert header is not None
    assert header.cwd == "/workspace/project"
