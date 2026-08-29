"""RAG tools for document ingestion and retrieval."""

from __future__ import annotations

from lsm_harness.agent.tools import ToolResult
from lsm_harness.coding_agent.tools import ToolDefinition


def make_tools(engine) -> list[ToolDefinition]:
    """Build RAG-related tools bound to a RAGEngine instance."""

    return [
        ToolDefinition(
            name="ingest_document",
            label="摄入文档",
            description=(
                "将本地文件摄入知识库，使其内容可被 Agent 检索。"
                "支持 .txt .md .py .js .ts .toml .yaml .json 等文本文件。"
                "摄入后，相关查询将自动检索文件内容。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "要摄入的文件路径。",
                    },
                    "title": {
                        "type": "string",
                        "description": "文档标题（可选，默认使用文件名）。",
                    },
                },
                "required": ["path"],
            },
            execute=lambda path, title="": _ingest(engine, path, title),
            effect="local_write",
            execution_mode="sequential",
            timeout=120.0,  # embedding can take time
        ),
        ToolDefinition(
            name="ingest_directory",
            label="摄入目录",
            description=(
                "将整个目录下的文本文件批量摄入知识库。"
                "默认摄入 .py .md .txt .js .ts .toml .yaml .yml .json 文件。"
                "跳过大于 2MB 的文件和隐藏目录。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "要摄入的目录路径。",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "glob 过滤模式，默认 *.py *.md 等常用文本格式。",
                    },
                },
                "required": ["directory"],
            },
            execute=lambda directory, pattern="*.{py,md,txt,js,ts,toml,yaml,yml,json}": (
                _ingest_dir(engine, directory, pattern)
            ),
            effect="local_write",
            execution_mode="sequential",
            timeout=300.0,
        ),
        ToolDefinition(
            name="search_documents",
            label="搜索知识库",
            description=(
                "在已摄入的文档中搜索相关内容。"
                "使用混合检索（语义向量 + 关键词）和重排序。"
                "返回最相关的文档片段。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，可以是自然语言问题。",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数，默认 5，上限 10。",
                    },
                },
                "required": ["query"],
            },
            execute=lambda query, top_k=5: _search(engine, query, top_k),
            effect="read",
            execution_mode="parallel",
            timeout=30.0,
        ),
        ToolDefinition(
            name="list_rag_documents",
            label="列出知识库文档",
            description="列出知识库中所有已摄入的文档及其统计信息。",
            parameters={
                "type": "object",
                "properties": {},
            },
            execute=lambda: _list(engine),
            effect="read",
            execution_mode="parallel",
        ),
    ]


def _ingest(engine, path: str, title: str) -> str | ToolResult:
    try:
        n = engine.ingest_file(path, title=title)
        return f"已摄入 '{path}': {n} 个文本块。"
    except FileNotFoundError:
        return ToolResult(output=f"文件不存在: {path}", is_error=True)
    except Exception as exc:
        return ToolResult(output=f"摄入失败: {type(exc).__name__}: {exc}", is_error=True)


def _ingest_dir(engine, directory: str, pattern: str) -> str | ToolResult:
    try:
        results = engine.ingest_directory(directory, pattern)
        if not results:
            return f"在 '{directory}' 中没有找到匹配 '{pattern}' 的文本文件。"
        total_chunks = sum(results.values())
        lines = [
            f"已摄入 '{directory}': {len(results)} 个文件，{total_chunks} 个文本块。",
            "",
        ]
        for path, n in sorted(results.items())[:20]:
            lines.append(f"  {path} ({n} chunks)")
        if len(results) > 20:
            lines.append(f"  ... 还有 {len(results) - 20} 个文件")
        return "\n".join(lines)
    except Exception as exc:
        return ToolResult(
            output=f"摄入目录失败: {type(exc).__name__}: {exc}",
            is_error=True,
        )


def _search(engine, query: str, top_k: int) -> str | ToolResult:
    import json
    top_k = max(1, min(int(top_k), 10))
    try:
        results = engine.search(query, top_k=top_k)
    except ImportError as exc:
        return ToolResult(output=f"RAG 搜索不可用: {exc}", is_error=True)
    except Exception as exc:
        return ToolResult(output=f"搜索失败: {type(exc).__name__}: {exc}", is_error=True)

    if not results:
        return f"在知识库中没有找到与 '{query}' 相关的内容。试试先摄入一些文档。"

    return engine._format_results(results)
