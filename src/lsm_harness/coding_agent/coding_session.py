"""Public coding-session boundary.

``Harness`` remains the compatibility entry used by existing frontends. New
code should depend on ``CodingSession``: it owns the Agent, durable Session,
tools, runtime model state, and the typed product event stream.
"""

from __future__ import annotations

from lsm_harness.coding_agent.app import CodingSession


__all__ = ["CodingSession"]
