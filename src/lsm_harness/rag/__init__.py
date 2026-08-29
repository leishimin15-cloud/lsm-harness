"""Hybrid RAG engine: chunk → embed → hybrid search → rerank.

Architecture:
  1. Ingestion:  file → text → semantic chunks → embed → store
  2. Hybrid Search: vector (cosine) + FTS5 (BM25) → RRF fusion
  3. Rerank:       small model scores top candidates
  4. Auto-trigger:  retrieval gate decides whether to search documents

Storage: SQLite (chunks + FTS5) + local sentence-transformers model.
The embedding model (~50 MB) is downloaded once and cached locally.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from lsm_harness.types import ModelClient


# ── chunking ────────────────────────────────────────────────────────

# Split on paragraph boundaries first, then by token budget
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate for mixed Chinese/ASCII text."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    return cjk + math.ceil((len(text) - cjk) / 4)


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping semantic chunks.

    Prefers paragraph boundaries.  Falls back to sentence boundaries,
    then fixed-size windows.
    """
    paragraphs = _PARAGRAPH_RE.split(text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if _estimate_tokens(current + "\n\n" + para) > chunk_size and current:
            chunks.append(current.strip())
            # Overlap: keep last `overlap` chars of previous chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text.strip()]


# ── embedding ───────────────────────────────────────────────────────

def _load_embedding_model(home: Path) -> Any:
    """Load (or download) a local sentence-transformers model.

    Uses BGE-small-zh — a bilingual Chinese/English model:
      - 384-dimensional embeddings
      - ~50 MB download
      - Optimised for retrieval tasks
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "RAG requires sentence-transformers. Install with:\n"
            "  pip install sentence-transformers"
        )

    cache = home / "models"
    cache.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        cache_folder=str(cache),
    )
    return model


def _embed(model: Any, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning normalised vectors."""
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# ── RAG engine ──────────────────────────────────────────────────────


class RAGEngine:
    """Document ingestion, hybrid search, and reranking.

    Usage::

        engine = RAGEngine(conn, client, home, small_model="deepseek-v4-flash")
        engine.ingest_file("README.md")
        results = engine.search("如何配置 MCP？", top_k=5)
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        client: ModelClient,
        home: Path,
        small_model: str = "",
        chunk_size: int = 512,
    ):
        self.conn = conn
        self.client = client
        self.home = home
        self.small_model = small_model
        self.chunk_size = max(64, int(chunk_size))
        self._model: Any = None  # lazy loaded
        self._ensure_schema()

    # ── schema ──────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS rag_docs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                ingested_at TEXT DEFAULT (datetime('now')),
                chunk_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL REFERENCES rag_docs(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                token_count INTEGER DEFAULT 0
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts
                USING fts5(content, content='rag_chunks', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS rag_chunks_ai AFTER INSERT ON rag_chunks BEGIN
                INSERT INTO rag_fts(rowid, content) VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS rag_chunks_ad AFTER DELETE ON rag_chunks BEGIN
                INSERT INTO rag_fts(rag_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS rag_chunks_au AFTER UPDATE ON rag_chunks BEGIN
                INSERT INTO rag_fts(rag_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO rag_fts(rowid, content) VALUES (new.id, new.content);
            END;
        """)
        self.conn.execute("INSERT INTO rag_fts(rag_fts) VALUES('rebuild')")
        self.conn.commit()

    # ── ingestion ────────────────────────────────────────────────

    def ingest_file(self, path: str, title: str = "") -> int:
        """Ingest a single file into the knowledge base.

        Returns the number of chunks created.
        Raises FileNotFoundError if the file doesn't exist.
        """
        filepath = Path(os.path.expanduser(path))
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not filepath.is_file():
            raise IsADirectoryError(f"Not a regular file: {path}")

        try:
            text = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = filepath.read_text(encoding="latin-1")
            except Exception:
                raise ValueError(f"Cannot decode file as text: {path}")

        return self.ingest_text(
            text=text,
            path=str(filepath),
            title=title or filepath.name,
        )

    def ingest_text(self, text: str, *, path: str, title: str = "") -> int:
        """Ingest raw text as a document."""
        # Remove old version if re-ingesting
        old = self.conn.execute(
            "SELECT id FROM rag_docs WHERE path=?", (path,)
        ).fetchone()
        if old:
            self.conn.execute("DELETE FROM rag_docs WHERE id=?", (old[0],))
            self.conn.commit()

        self.conn.execute(
            "INSERT INTO rag_docs(path, title) VALUES(?,?)",
            (path, title),
        )
        doc_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        chunks = _chunk_text(text, chunk_size=self.chunk_size)
        model = self._get_model()
        embeddings = _embed(model, chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            self.conn.execute(
                "INSERT INTO rag_chunks(doc_id, chunk_index, content, embedding, token_count) "
                "VALUES(?,?,?,?,?)",
                (doc_id, i, chunk, json.dumps(emb), _estimate_tokens(chunk)),
            )

        self.conn.execute(
            "UPDATE rag_docs SET chunk_count=? WHERE id=?",
            (len(chunks), doc_id),
        )
        self.conn.commit()
        return len(chunks)

    def ingest_directory(
        self, directory: str, glob_pattern: str = "*.{py,md,txt,js,ts,toml,yaml,yml,json}"
    ) -> dict[str, int]:
        """Ingest all matching files in a directory.

        Returns a dict of {path: chunk_count}.
        """
        import fnmatch
        import glob as glob_mod

        dirpath = Path(os.path.expanduser(directory))
        if not dirpath.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Parse glob patterns like *.{py,md}
        patterns = []
        if "," in glob_pattern:
            base, _, ext_list = glob_pattern.partition(".{")
            ext_list = ext_list.rstrip("}")
            for ext in ext_list.split(","):
                patterns.append(f"{base}.{ext}")
        else:
            patterns = [glob_pattern]

        results: dict[str, int] = {}
        for pattern in patterns:
            for filepath in dirpath.rglob(pattern):
                # Skip hidden dirs and files
                if any(part.startswith(".") for part in filepath.parts):
                    continue
                if filepath.stat().st_size > 2 * 1024 * 1024:  # skip >2MB
                    continue
                try:
                    n = self.ingest_file(str(filepath))
                    results[str(filepath.relative_to(dirpath))] = n
                except Exception:
                    continue

        return results

    # ── search ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Hybrid search: vector + FTS5 → RRF fusion → optional rerank.

        Returns a list of {chunk_id, content, title, path, score, ...}.
        """
        model = self._get_model()
        query_emb = _embed(model, [query])[0]

        # ── vector search ────────────────────────────────────
        rows = self.conn.execute(
            "SELECT c.id, c.content, c.embedding, d.title, d.path, c.chunk_index "
            "FROM rag_chunks c JOIN rag_docs d ON c.doc_id = d.id"
        ).fetchall()

        vector_results: list[dict[str, Any]] = []
        for row in rows:
            emb = json.loads(row["embedding"])
            score = _cosine(query_emb, emb)
            vector_results.append({
                "id": row["id"],
                "content": row["content"],
                "title": row["title"],
                "path": row["path"],
                "chunk_index": row["chunk_index"],
                "vector_score": score,
            })
        vector_results.sort(key=lambda x: x["vector_score"], reverse=True)

        # ── FTS5 search ──────────────────────────────────────
        fts5_results: list[dict[str, Any]] = []
        try:
            fts_rows = self.conn.execute(
                "SELECT c.id, c.content, d.title, d.path, c.chunk_index, "
                "bm25(rag_fts, 0, 0, 1) as bm25_score "
                "FROM rag_fts f JOIN rag_chunks c ON f.rowid = c.id "
                "JOIN rag_docs d ON c.doc_id = d.id "
                "WHERE rag_fts MATCH ? ORDER BY bm25_score LIMIT ?",
                (query, top_k * 3),
            ).fetchall()
            for row in fts_rows:
                fts5_results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "title": row["title"],
                    "path": row["path"],
                    "chunk_index": row["chunk_index"],
                    "fts5_score": float(row["bm25_score"]),
                })
        except Exception:
            # FTS5 may fail on malformed queries — just skip
            pass

        # ── RRF fusion ───────────────────────────────────────
        fused = self._rrf_fusion(vector_results[:top_k * 3], fts5_results, k=60)

        # ── rerank (optional) ────────────────────────────────
        if rerank and len(fused) > top_k and self.small_model:
            fused = self._rerank(query, fused[:top_k * 2], top_k)

        return fused[:top_k]

    def _rrf_fusion(
        self,
        list_a: list[dict],
        list_b: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion — merge two ranked lists."""
        scores: dict[int, float] = {}
        items: dict[int, dict] = {}

        for rank, item in enumerate(list_a):
            scores[item["id"]] = scores.get(item["id"], 0) + 1.0 / (k + rank + 1)
            items[item["id"]] = item

        for rank, item in enumerate(list_b):
            scores[item["id"]] = scores.get(item["id"], 0) + 1.0 / (k + rank + 1)
            items[item["id"]] = item

        fused = [(items[cid], s) for cid, s in scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        return [{**item, "fused_score": score} for item, score in fused]

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Use the small model to rerank candidate chunks.

        Sends all candidates in one call, asks for relevance scores.
        Falls back to original order on failure.
        """
        if not candidates:
            return candidates

        # Build a compact scoring prompt
        chunks_text = "\n\n---\n\n".join(
            f"[{i}] {c['content'][:300]}" for i, c in enumerate(candidates)
        )
        prompt = (
            f"查询: {query}\n\n"
            f"以下是从文档中检索到的 {len(candidates)} 个候选片段。"
            f"请选出最相关的 {top_k} 个，按相关性从高到低排列。"
            f"只返回 JSON 数组，包含片段编号，例如 [3, 0, 7]:\n\n"
            f"{chunks_text}"
        )

        try:
            response = self.client.complete(
                model=self.small_model,
                system="你是文档检索重排序器。只返回 JSON 数组。",
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                max_tokens=300,
            )
            # Parse the JSON array
            text = response.text.strip()
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                indices = json.loads(text[start:end + 1])
                reranked = []
                seen: set[int] = set()
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                        reranked.append(candidates[idx])
                        seen.add(idx)
                # Append any candidates not mentioned
                for i, c in enumerate(candidates):
                    if i not in seen:
                        reranked.append(c)
                return reranked[:top_k]
        except Exception:
            pass

        return candidates[:top_k]

    # ── list / stats ────────────────────────────────────────────

    def list_documents(self) -> list[dict[str, Any]]:
        """List all ingested documents."""
        rows = self.conn.execute(
            "SELECT id, path, title, ingested_at, chunk_count "
            "FROM rag_docs ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def delete_document(self, path: str) -> bool:
        """Remove a document and its chunks from the index."""
        cursor = self.conn.execute(
            "DELETE FROM rag_docs WHERE path=?", (path,)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        docs = self.conn.execute("SELECT COUNT(*) FROM rag_docs").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        tokens = self.conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM rag_chunks"
        ).fetchone()[0]
        return {"documents": docs, "chunks": chunks, "total_tokens": tokens}

    # ── auto-trigger integration ────────────────────────────────

    def gated_search(
        self,
        user_message: str,
        top_k: int = 5,
        emit=None,
    ) -> str:
        """Search documents only when the retrieval gate says so.

        Uses the small model to decide whether the user message requires
        document retrieval.  Returns formatted results or empty string.
        """
        from lsm_harness.memory.retrieval import should_retrieve

        should, query, reason = should_retrieve(
            self.client,
            self.small_model,
            user_message,
            emit=emit,
        )

        if emit:
            emit("rag.gate.decided", {
                "decision": "search" if should else "skip",
                "reason": reason,
                "query": query,
            })

        if not should:
            return ""

        results = self.search(query, top_k=top_k)

        if emit:
            emit("rag.retrieved", {
                "query": query,
                "results": len(results),
            })

        if not results:
            return ""

        return self._format_results(results)

    @staticmethod
    def _format_results(results: list[dict]) -> str:
        """Format search results as a Markdown-like string for the model."""
        lines = ["## 相关文档片段\n"]
        for i, r in enumerate(results):
            title = r.get("title", "unknown")
            path = r.get("path", "")
            content = r.get("content", "")[:500]
            lines.append(
                f"### [{i + 1}] {title}\n"
                f"**来源**: {path}\n\n"
                f"{content}\n"
            )
        return "\n".join(lines)

    # ── internal ─────────────────────────────────────────────────

    def _get_model(self) -> Any:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = _load_embedding_model(self.home)
        return self._model
