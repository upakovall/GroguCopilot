"""Generic Semantic ViewContext & UIComponent Schemas.

Declarative UI models allowing any host application to expose its UI state
without DOM scraping or CSS selector dependencies.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UIComponent(BaseModel):
    """Generic description of an actionable interactive UI element."""
    id: str = Field(..., description="Unique semantic identifier for the element")
    type: str = Field(..., description="Semantic type (e.g., 'button', 'switch', 'input', 'select', 'table', 'modal', 'card')")
    label: str = Field(..., description="Human-readable title or description of element purpose")
    value: Optional[Any] = Field(None, description="Current value, text, checked state, or active selection")
    enabled: bool = Field(True, description="Whether the component is currently interactive")
    options: Optional[List[str]] = Field(None, description="Available options if component is a select or dropdown")
    allowed_actions: List[str] = Field(
        default_factory=list,
        description="Allowed semantic actions (e.g., ['click', 'set_value', 'toggle', 'select_option'])"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional domain-specific attributes (e.g. min/max bounds, data counts, keywords)"
    )


class ViewContext(BaseModel):
    """Declarative snapshot of the active application screen and interactive components."""
    screen_id: str = Field(..., description="Identifier for current view/page")
    title: str = Field(..., description="Human-readable title of current view")
    active_modal: Optional[str] = Field(None, description="ID of open modal if one is currently displayed")
    focused_element_id: Optional[str] = Field(None, description="ID of element currently focused by user")
    components: List[UIComponent] = Field(
        default_factory=list,
        description="List of all visible and actionable semantic components"
    )
    state_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="High-level domain state snapshot"
    )
    timestamp: Optional[float] = Field(None, description="Timestamp when ViewContext snapshot was generated")

    def get_component(self, component_id: str) -> Optional[UIComponent]:
        """Look up a component by semantic ID."""
        for comp in self.components:
            if comp.id == component_id:
                return comp
        return None
