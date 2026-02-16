"""Tests for /memu command migration."""
from nanobot.agent.loop import AgentLoop


def test_memu_command_specs_no_menu() -> None:
    """Test that /memu command specs don't contain /menu commands."""
    # Create a minimal mock AgentLoop to access command specs
    # We just need to verify the specs don't contain /menu

    # Read the source file and check that _memu_base_command_specs uses /memu
    import inspect
    source = inspect.getsource(AgentLoop._memu_base_command_specs)

    # Verify all commands in _memu_base_command_specs use /memu prefix
    assert "/memu help" in source
    assert "/menu help" not in source
    assert "/memu list" in source
    assert "/menu list" not in source
    assert "/memu status" in source
    assert "/menu status" not in source
    assert "/memu tune" in source
    assert "/menu tune" not in source
    assert "/memu restart" in source
    assert "/menu restart" not in source


def test_memu_command_specs_contain_expected_commands() -> None:
    """Test that /memu command specs contain expected commands."""
    import inspect
    source = inspect.getsource(AgentLoop._memu_base_command_specs)

    # Verify expected commands are present
    expected_commands = [
        "/memu help",
        "/memu list",
        "/memu all",
        "/memu categories",
        "/memu status",
        "/memu tune",
        "/memu restart now",
        "/memu version",
    ]

    for cmd in expected_commands:
        assert cmd in source, f"Missing command: {cmd}"


def test_memu_command_specs_no_model_routing() -> None:
    """Test that /memu command specs don't include model routing."""
    import inspect

    # Check _memu_command_specs_full doesn't include routed model commands
    source = inspect.getsource(AgentLoop._memu_command_specs_full)

    # Should not have /memu model routes
    assert "/memu model" not in source


def test_memu_handler_accepts_memu_prefix() -> None:
    """Test that _handle_memu_command_full accepts /memu prefix."""
    import inspect
    source = inspect.getsource(AgentLoop._handle_memu_command_full)

    # Should check for /memu prefix
    assert 'raw.startswith("/memu")' in source


def test_memu_handler_backward_compatible_with_menu() -> None:
    """Test that _handle_memu_command_full is backward compatible with /menu."""
    import inspect
    source = inspect.getsource(AgentLoop._handle_memu_command_full)

    # Should have backward compatibility for /menu
    assert 'raw.startswith("/menu")' in source or "/menu" in source
