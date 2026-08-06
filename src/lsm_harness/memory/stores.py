"""Chinese-first SQLite memory stores using trigram FTS with LIKE fallback."""

from __future__ import annotations

import re
import sqlite3


def _terms(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"[A-Za-z0-9_-]+|[\u3400-\u9fff]+", query.strip())))


def _fts_expression(query: str) -> str:
    usable = [term for term in _terms(query) if len(term) >= 3]
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in usable)


def _like_clause(columns: tuple[str, ...], query: str) -> tuple[str, list[str]]:
    terms = _terms(query)
    if not terms:
        return "", []
    clauses, params = [], []
    for term in terms:
        for column in columns:
            clauses.append(f"{column} LIKE ?")
            params.append(f"%{term}%")
    return " OR ".join(clauses), params


def _relevance_tokens(text: str) -> set[str]:
    """Transparent fallback tokens for Chinese phrases that are not contiguous.

    Trigram FTS is excellent for exact substrings, but a query such as
    ``当前开发项目`` should still match ``正在开发的 Agent 项目``. Chinese
    bigrams make that relationship visible without an embedding dependency.
    """
    lowered = text.lower()
    tokens = set(re.findall(r"[a-z0-9_-]{2,}", lowered))
    for chunk in re.findall(r"[\u3400-\u9fff]+", text):
        if len(chunk) == 1:
            tokens.add(chunk)
        else:
            tokens.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return tokens


def _rank_rows(rows: list[sqlite3.Row], query: str, fields: tuple[str, ...], top_k: int):
    query_text = query.strip().lower()
    query_tokens = _relevance_tokens(query)
    ranked = []
    for row in rows:
        text = " ".join(str(row[field]) for field in fields if row[field]).lower()
        overlap = len(query_tokens & _relevance_tokens(text))
        score = overlap + (20 if query_text and query_text in text else 0)
        if score:
            ranked.append((score, int(row["id"]), row))
    ranked.sort(key=lambda item: (-item[0], -item[1]))
    return [row for _, _, row in ranked[:top_k]]


class FactStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add(self, subject: str, content: str, source: str = "user") -> int:
        cursor = self.conn.execute(
            "INSERT INTO facts (subject, content, source) VALUES (?,?,?)",
            (subject.strip().lower(), content.strip(), source),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def _rows(self, query: str, top_k: int) -> list[sqlite3.Row]:
        fts = _fts_expression(query)
        rows: list[sqlite3.Row] = []
        if fts:
            rows = self.conn.execute(
                "SELECT f.id, f.subject, f.content, f.source, f.created_at "
                "FROM facts_fts JOIN facts f ON f.id=facts_fts.rowid "
                "WHERE facts_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts, top_k),
            ).fetchall()
        if rows:
            return rows
        clause, params = _like_clause(("subject", "content"), query)
        if not clause:
            return []
        rows = self.conn.execute(
            f"SELECT id, subject, content, source, created_at FROM facts "
            f"WHERE {clause} ORDER BY id DESC LIMIT ?",
            [*params, top_k],
        ).fetchall()
        if rows:
            return rows
        candidates = self.conn.execute(
            "SELECT id, subject, content, source, created_at FROM facts ORDER BY id DESC LIMIT 500"
        ).fetchall()
        return _rank_rows(candidates, query, ("subject", "content"), top_k)

    def search(self, query: str, top_k: int = 4) -> list[str]:
        return [f"[{row['subject']}] {row['content']}" for row in self._rows(query, top_k)]

    def search_with_ids(self, query: str, top_k: int = 8) -> list[dict]:
        return [dict(row) for row in self._rows(query, top_k)] if query.strip() else self.list(top_k)

    def list(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, subject, content, source, created_at FROM facts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def update(self, fact_id: int, content: str, subject: str | None = None) -> bool:
        if subject:
            cursor = self.conn.execute(
                "UPDATE facts SET subject=?, content=? WHERE id=?",
                (subject.strip().lower(), content.strip(), fact_id),
            )
        else:
            cursor = self.conn.execute(
                "UPDATE facts SET content=? WHERE id=?", (content.strip(), fact_id)
            )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete(self, fact_id: int) -> bool:
        cursor = self.conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))
        self.conn.commit()
        return cursor.rowcount > 0


class EpisodeStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def add(self, summary: str, happened_at: str) -> int:
        cursor = self.conn.execute(
            "INSERT INTO episodes (happened_at, summary) VALUES (?,?)",
            (happened_at, summary.strip()),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def _rows(self, query: str, top_k: int) -> list[sqlite3.Row]:
        fts = _fts_expression(query)
        rows: list[sqlite3.Row] = []
        if fts:
            rows = self.conn.execute(
                "SELECT e.id, e.happened_at, e.summary, e.created_at FROM episodes_fts "
                "JOIN episodes e ON e.id=episodes_fts.rowid "
                "WHERE episodes_fts MATCH ? ORDER BY rank, e.happened_at DESC LIMIT ?",
                (fts, top_k),
            ).fetchall()
        if not rows:
            clause, params = _like_clause(("summary",), query)
            if clause:
                rows = self.conn.execute(
                    f"SELECT id, happened_at, summary, created_at FROM episodes WHERE {clause} "
                    "ORDER BY happened_at DESC, id DESC LIMIT ?",
                    [*params, top_k],
                ).fetchall()
        if rows:
            return rows
        candidates = self.conn.execute(
            "SELECT id, happened_at, summary, created_at FROM episodes ORDER BY id DESC LIMIT 500"
        ).fetchall()
        return _rank_rows(candidates, query, ("summary",), top_k)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        return [
            f"({row['happened_at']}) {row['summary']}" for row in self._rows(query, top_k)
        ]

    def search_with_ids(self, query: str, top_k: int = 8) -> list[dict]:
        return [dict(row) for row in self._rows(query, top_k)] if query.strip() else self.list(top_k)

    def list(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, happened_at, summary, created_at FROM episodes ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, episode_id: int) -> bool:
        cursor = self.conn.execute("DELETE FROM episodes WHERE id=?", (episode_id,))
        self.conn.commit()
        return cursor.rowcount > 0
