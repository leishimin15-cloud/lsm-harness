"""Compatibility imports for the AI-layer contracts."""

from lsm_harness.ai.types import *  # noqa: F403

# TraceResult/TurnResult moved to the Agent layer (refactor plan §9.1);
# re-export here so existing ``lsm_harness.types`` users keep working.
from lsm_harness.agent.types import TraceResult, TraceStatus, TurnResult  # noqa: F401
