"""Model Context Protocol (MCP) & Semantic Tool Registry.

Provides Inversion-of-Control (IoC) registration for UI components, custom tools,
and action validation for third-party host applications.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from .schemas.context import ViewContext, UIComponent
from .schemas.actions import UIAction, ActionType

logger = logging.getLogger(__name__)

ActionValidatorFn = Callable[[UIAction, ViewContext], Tuple[bool, Optional[str]]]
ActionHookFn = Callable[[UIAction, ViewContext], Optional[Dict[str, Any]]]


class MCPRegistry:
    """Inversion of Control Registry for UI Tools and Semantic Action Handlers."""

    def __init__(self):
        self._current_context: Optional[ViewContext] = None
        self._custom_validators: Dict[str, ActionValidatorFn] = {}
        self._custom_hooks: Dict[str, ActionHookFn] = {}
        self._extra_tools: List[Dict[str, Any]] = []

    def update_context(self, context: ViewContext) -> None:
        """Update active ViewContext received from frontend."""
        self._current_context = context
        logger.debug(f"[MCPRegistry] Updated context for screen: {context.screen_id} ({len(context.components)} components)")

    def get_context(self) -> Optional[ViewContext]:
        """Fetch current ViewContext snapshot."""
        return self._current_context

    def register_tool(self, tool_def: Dict[str, Any]) -> None:
        """Register an extra MCP tool definition."""
        self._extra_tools.append(tool_def)

    def register_action_validator(self, action_type: str, validator_fn: ActionValidatorFn) -> None:
        """Register a custom validation function for a specific action type."""
        self._custom_validators[action_type] = validator_fn

    def register_action_hook(self, action_type: str, hook_fn: ActionHookFn) -> None:
        """Register a server-side execution hook triggered when an action is executed."""
        self._custom_hooks[action_type] = hook_fn

    def validate_action(self, action: UIAction) -> Tuple[bool, Optional[str]]:
        """Validate whether a requested UIAction is semantically permissible."""
        if not self._current_context:
            return False, "No active ViewContext available"

        # Check if custom validator is registered
        if action.action_type.value in self._custom_validators:
            return self._custom_validators[action.action_type.value](action, self._current_context)

        # Global actions not requiring specific target component
        if action.action_type in [ActionType.RESET_FILTERS, ActionType.NOTIFY_USER]:
            return True, None

        if action.action_type == ActionType.NAVIGATE:
            destination = action.payload.get("screen_id") or action.payload.get("destination")
            if not destination:
                return False, "NAVIGATE action requires 'screen_id' or 'destination' in payload"
            return True, None

        # Targeted actions
        if not action.target_id:
            return False, f"Action {action.action_type} requires target_id"

        component = self._current_context.get_component(action.target_id)
        if not component:
            return False, f"Target component '{action.target_id}' not found in active ViewContext ({self._current_context.screen_id})"

        if not component.enabled:
            return False, f"Target component '{action.target_id}' is currently disabled"

        # Check allowed_actions permissions
        action_name = action.action_type.value.lower()
        allowed_normalized = [a.lower() for a in component.allowed_actions]

        # Standard action type alias map
        action_mapping = {
            ActionType.CLICK_BUTTON: ["click", "click_button", "press"],
            ActionType.TOGGLE_SWITCH: ["toggle", "toggle_switch", "click", "set_value"],
            ActionType.SET_INPUT_VALUE: ["set_value", "type", "set_input_value", "input"],
            ActionType.SELECT_OPTION: ["select_option", "select", "set_value", "choose"],
            ActionType.OPEN_MODAL: ["open_modal", "open", "click", "show"],
            ActionType.CLOSE_MODAL: ["close_modal", "close", "click", "hide"],
            ActionType.FILTER_TABLE: ["filter", "filter_table", "set_value"],
            ActionType.CUSTOM: ["custom", "*"],
        }

        expected_perms = action_mapping.get(action.action_type, [action_name])
        # If component allows '*' or any of expected permissions
        if "*" in allowed_normalized:
            return True, None

        has_permission = any(perm in allowed_normalized for perm in expected_perms)
        if not has_permission:
            return False, f"Action '{action.action_type}' is not in allowed_actions {component.allowed_actions} for component '{action.target_id}'"

        return True, None

    def execute_hook_if_present(self, action: UIAction) -> Optional[Dict[str, Any]]:
        """Run server-side hook if registered for this action."""
        if action.action_type.value in self._custom_hooks and self._current_context:
            return self._custom_hooks[action.action_type.value](action, self._current_context)
        return None

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return standardized MCP Tool definitions for the LLM."""
        base_tools = [
            {
                "name": "get_active_view_context",
                "description": "Get the current semantic UI state, active screen, and interactive components.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "execute_ui_action",
                "description": "Execute a semantic UI action on an element without DOM scraping.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_type": {
                            "type": "string",
                            "enum": [a.value for a in ActionType],
                            "description": "Semantic action to perform",
                        },
                        "target_id": {
                            "type": "string",
                            "description": "Semantic ID of component",
                        },
                        "payload": {
                            "type": "object",
                            "description": "Arguments for action",
                        },
                        "description": {
                            "type": "string",
                            "description": "Explanation of action purpose",
                        },
                    },
                    "required": ["action_type", "description"],
                },
            },
        ]
        return base_tools + self._extra_tools
