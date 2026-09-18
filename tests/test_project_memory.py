"""ProjectMemoryStore / render_for_prompt 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

from lsm_harness.tools.memory import (
    ProjectMemoryStore,
    render_for_prompt,
    validate_entry,
)


def _store(tmp_path: Path) -> tuple[ProjectMemoryStore, Path]:
    return ProjectMemoryStore(tmp_path / "projects"), tmp_path / "ws"


class TestPaths:
    def test_path_is_per_project_and_stable(self, tmp_path):
        store, ws = _store(tmp_path)
        p1 = store.path_for(ws)
        assert p1 == store.path_for(ws)
        assert p1.parent.name.startswith("ws-")
        assert p1.name == "memory.json"
        assert store.projects_dir in p1.parents

    def test_same_dirname_different_paths_do_not_collide(self, tmp_path):
        store = ProjectMemoryStore(tmp_path / "projects")
        a = store.path_for(tmp_path / "a" / "ws")
        b = store.path_for(tmp_path / "b" / "ws")
        assert a != b


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        store, ws = _store(tmp_path)
        entry, err = store.save_entry(
            ws, name="db-choice", type="decision",
            content="用 SQLite 而不是 Postgres:单机部署优先",
            tags=["storage"],
        )
        assert err is None
        loaded = store.load(ws)
        assert loaded == [entry]
        assert loaded[0]["created_at"] == loaded[0]["updated_at"]

    def test_upsert_keeps_created_at(self, tmp_path):
        store, ws = _store(tmp_path)
        first, _ = store.save_entry(
            ws, name="n", type="fact", content="v1")
        second, _ = store.save_entry(
            ws, name="n", type="fact", content="v2")
        assert second["content"] == "v2"
        assert second["created_at"] == first["created_at"]
        assert len(store.load(ws)) == 1

    def test_delete(self, tmp_path):
        store, ws = _store(tmp_path)
        store.save_entry(ws, name="n", type="fact", content="c")
        assert store.delete(ws, "n") is True
        assert store.delete(ws, "n") is False
        assert store.load(ws) == []

    def test_load_missing_file_is_empty(self, tmp_path):
        store, ws = _store(tmp_path)
        assert store.load(ws) == []

    def test_corrupt_file_is_empty_not_crash(self, tmp_path):
        store, ws = _store(tmp_path)
        path = store.path_for(ws)
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")
        assert store.load(ws) == []
        path.write_text(json.dumps({"version": 2, "entries": []}),
                        encoding="utf-8")
        assert store.load(ws) == []

    def test_entry_cap(self, tmp_path):
        store, ws = _store(tmp_path)
        path = store.path_for(ws)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "version": 1,
            "entries": [
                {"name": f"e{i}", "type": "fact", "content": "x",
                 "tags": [], "created_at": "t", "updated_at": "t"}
                for i in range(100)
            ],
        }), encoding="utf-8")
        entry, err = store.save_entry(ws, name="new", type="fact", content="x")
        assert entry is None and "上限" in err
        # 同名覆盖不受上限影响
        entry, err = store.save_entry(ws, name="e0", type="fact", content="y")
        assert err is None


class TestValidation:
    def test_valid(self):
        assert validate_entry(
            name="a-b_c", type="gotcha", content="坑", tags=["x"]) is None

    def test_bad_type(self):
        err = validate_entry(name="n", type="random", content="c", tags=[])
        assert "type" in err

    def test_bad_name(self):
        assert validate_entry(
            name="", type="fact", content="c", tags=[]) is not None
        assert validate_entry(
            name="-lead", type="fact", content="c", tags=[]) is not None
        assert validate_entry(
            name="空格 名", type="fact", content="c", tags=[]) is not None
        assert validate_entry(
            name="x" * 65, type="fact", content="c", tags=[]) is not None

    def test_content_and_tags_limits(self):
        assert validate_entry(
            name="n", type="fact", content="", tags=[]) is not None
        assert validate_entry(
            name="n", type="fact", content="x" * 2001, tags=[]) is not None
        assert validate_entry(
            name="n", type="fact", content="c",
            tags=["t"] * 9) is not None
        assert validate_entry(
            name="n", type="fact", content="c",
            tags=["x" * 33]) is not None


class TestRenderForPrompt:
    def test_empty(self):
        assert render_for_prompt([]) == ""

    def test_envelope_and_lines(self):
        text = render_for_prompt([{
            "name": "db", "type": "decision", "content": "用 SQLite",
            "tags": ["storage"],
        }])
        assert text.startswith("<project_memory>")
        assert text.rstrip().endswith("</project_memory>")
        assert "- [decision] db: 用 SQLite #storage" in text

    def test_entry_budget(self):
        entries = [
            {"name": f"e{i}", "type": "fact", "content": "c", "tags": []}
            for i in range(60)
        ]
        text = render_for_prompt(entries, max_entries=50)
        assert "e49" in text and "e50" not in text
        assert "还有 10 条" in text

    def test_char_budget(self):
        entries = [
            {"name": f"e{i}", "type": "fact", "content": "x" * 500,
             "tags": []}
            for i in range(20)
        ]
        text = render_for_prompt(entries, max_chars=2000)
        assert len(text) <= 2100  # 截断 + 收尾标签
