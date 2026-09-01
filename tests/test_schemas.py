"""Unit tests for Pydantic schemas in grogu_copilot."""

import pytest
from pydantic import ValidationError
from grogu_copilot.schemas import (
    UIComponent,
    ViewContext,
    ActionType,
    UIAction,
    AgentResponse,
)


def test_ui_component_schema():
    comp = UIComponent(
        id="sample_button",
        type="button",
        label="Click Me",
        allowed_actions=["click"]
    )
    assert comp.id == "sample_button"
    assert comp.allowed_actions == ["click"]


def test_agent_response_json_schema():
    schema = AgentResponse.model_json_schema()
    assert "properties" in schema
    assert "thought" in schema["properties"]
    assert "speech_output" in schema["properties"]
    assert "actions" in schema["properties"]
