"""Unit tests for Inversion of Control MCP Registry."""

from grogu_copilot.schemas import ViewContext, UIComponent, UIAction, ActionType
from grogu_copilot.registry import MCPRegistry


def test_mcp_registry_custom_validator():
    registry = MCPRegistry()
    ctx = ViewContext(
        screen_id="admin_panel",
        title="Admin Settings",
        components=[
            UIComponent(
                id="production_db_drop_btn",
                type="button",
                label="Drop Production Database",
                enabled=True,
                allowed_actions=["click"]
            )
        ]
    )
    registry.update_context(ctx)

    # Register custom safety validator forbidding dangerous operations
    def custom_safety_validator(action: UIAction, current_ctx: ViewContext):
        if action.target_id == "production_db_drop_btn":
            return False, "Dangerous action blocked by host security policy"
        return True, None

    registry.register_action_validator(ActionType.CLICK_BUTTON.value, custom_safety_validator)

    # Validate blocked action
    blocked_action = UIAction(
        action_type=ActionType.CLICK_BUTTON,
        target_id="production_db_drop_btn",
        description="Attempt drop database"
    )
    is_valid, err = registry.validate_action(blocked_action)
    assert is_valid is False
    assert "blocked by host security policy" in err


def test_mcp_registry_custom_action_hook():
    registry = MCPRegistry()
    ctx = ViewContext(
        screen_id="app_home",
        title="App Home",
        components=[
            UIComponent(
                id="analytics_opt_in",
                type="switch",
                label="Opt-in Analytics",
                enabled=True,
                allowed_actions=["toggle"]
            )
        ]
    )
    registry.update_context(ctx)

    hook_called = []

    def on_toggle_hook(action: UIAction, current_ctx: ViewContext):
        hook_called.append(action.target_id)
        return {"logged": True}

    registry.register_action_hook(ActionType.TOGGLE_SWITCH.value, on_toggle_hook)

    action = UIAction(
        action_type=ActionType.TOGGLE_SWITCH,
        target_id="analytics_opt_in",
        payload={"state": True},
        description="Toggle analytics"
    )
    is_valid, _ = registry.validate_action(action)
    assert is_valid is True

    hook_result = registry.execute_hook_if_present(action)
    assert hook_result == {"logged": True}
    assert hook_called == ["analytics_opt_in"]
