"""Build the v1.1 local-only tool surface."""

from lsm_harness.tools import calendar, delegate, memory_admin, notes
from lsm_harness.tools.registry import ToolRegistry


def build_registry(conn, settings, memory) -> ToolRegistry:
    allowed = {"read", "local_write"}

    # Enable pi delegation if pi is installed
    from lsm_harness.tools.delegate import _find_pi
    pi_available = _find_pi() is not None
    if pi_available:
        allowed.add("external_write")

    registry = ToolRegistry(allowed)
    for tool in calendar.make_tools(conn, settings.home):
        registry.register(tool)
    registry.register(notes.make_tool(memory))
    for tool in memory_admin.make_tools(settings, memory):
        registry.register(tool)

    if pi_available:
        registry.register(delegate.make_tool(settings.home))

    return registry
