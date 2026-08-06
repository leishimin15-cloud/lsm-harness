"""Build the v1.1 local-only tool surface."""

from lsm_harness.tools import calendar, memory_admin, notes
from lsm_harness.tools.registry import ToolRegistry


def build_registry(conn, settings, memory) -> ToolRegistry:
    registry = ToolRegistry({"read", "local_write"})
    for tool in calendar.make_tools(conn, settings.home):
        registry.register(tool)
    registry.register(notes.make_tool(memory))
    for tool in memory_admin.make_tools(settings, memory):
        registry.register(tool)
    return registry
