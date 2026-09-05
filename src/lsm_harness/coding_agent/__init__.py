"""Product layer built on top of the reusable AI and Agent packages.

The package initializer intentionally stays dependency-free. Import concrete
product services from ``coding_agent.app``, ``coding_agent.session``, or
``coding_agent.subagent`` so lower-level modules can use the package without
initializing the full application graph.
"""

from lsm_harness.coding_agent.tools import ToolDefinition, wrap_tool_definition

__all__ = ["ToolDefinition", "wrap_tool_definition"]
