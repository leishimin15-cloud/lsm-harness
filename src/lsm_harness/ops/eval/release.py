"""Create a reproducible release manifest after an eval gate passes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lsm_harness.ops.eval.gate import GateReport


def write_release_manifest(root: str | Path, gate: GateReport) -> Path:
    if not gate.passed:
        raise ValueError("cannot create a release manifest for a failed gate")
    root = Path(root).expanduser().resolve()
    records = _read_jsonl(root / "runs.jsonl")
    revisions = sorted({str(row.get("git_revision", "")) for row in records if row.get("git_revision")})
    configs: list[dict[str, Any]] = []
    seen_configs: set[str] = set()
    for row in records:
        config = row.get("config")
        if not isinstance(config, dict):
            continue
        key = json.dumps(
            config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if key in seen_configs:
            continue
        seen_configs.add(key)
        configs.append(config)
    canonical = json.dumps(configs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest = {
        "schema_version": 1,
        "release_id": root.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "artifact_dir": str(root),
        "git_revisions": revisions,
        "config_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "gate": gate.as_dict(),
        "configurations": configs,
    }
    path = root / "release_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows
