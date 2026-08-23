"""Guard the MCP SDK API used by Odysseus built-in stdio servers."""

from mcp.server import Server


def test_builtin_mcp_servers_have_required_lowlevel_api():
    assert hasattr(Server, "list_tools")
    assert hasattr(Server, "call_tool")
