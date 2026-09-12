"""Eval fixtures and golden comparisons must be independent from assertions."""

from __future__ import annotations

import json

import pytest

from lsm_harness.ai.types import ModelResponse
from lsm_harness.config import Settings
from lsm_harness.ops.eval import (
    EvalCase,
    EvalResult,
    _compare_latest_golden,
    _queue_client_for_case,
)


def test_deterministic_case_requires_explicit_script():
    case = EvalCase(name="not-self-fulfilling", expect_tools=["write_file"])
    with pytest.raises(ValueError, match="no scripted responses"):
        _queue_client_for_case(case)


def test_script_does_not_derive_from_expectations():
    case = EvalCase(
        name="independent",
        expect_tools=["write_file"],
        expect_in_reply=["must appear"],
        scripted_responses=[ModelResponse(text="independent output")],
    )
    client = _queue_client_for_case(case)
    response = client.complete()
    assert response.text == "independent output"
    assert response.tool_calls == []


def test_latest_golden_is_actually_compared(tmp_path):
    settings = Settings(home=tmp_path)
    golden = tmp_path / "evals" / "golden"
    golden.mkdir(parents=True)
    (golden / "golden-2026.json").write_text(
        json.dumps([
            {
                "name": "case",
                "tools_called": ["read_file"],
                "iterations": 1,
                "reply": "old",
            }
        ]),
        encoding="utf-8",
    )
    failures = _compare_latest_golden(
        settings,
        [EvalResult(
            case_name="case",
            passed=True,
            tools_called=["write_file"],
            iterations=2,
            reply="new",
        )],
    )
    assert len(failures) == 3
