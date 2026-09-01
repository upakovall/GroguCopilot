"""Unit tests for Dynamic Semantic Reasoner (Zero Hardcoding)."""

import asyncio
from grogu_copilot.schemas import ViewContext, UIComponent, ActionType
from grogu_copilot.registry import MCPRegistry
from grogu_copilot.llm.dynamic_reasoner import DynamicReasoner


def test_dynamic_reasoner_arbitrary_e_commerce_components():
    """Verify that DynamicReasoner maps prompts to completely arbitrary 3rd-party UI schemas."""
    async def _run():
        registry = MCPRegistry()
        reasoner = DynamicReasoner()

        # Completely custom e-commerce view context
        context = ViewContext(
            screen_id="checkout_flow",
            title="E-Commerce Shopping Cart",
            components=[
                UIComponent(
                    id="express_shipping_toggle",
                    type="switch",
                    label="Express Shipping Option",
                    value=False,
                    allowed_actions=["toggle", "click"]
                ),
                UIComponent(
                    id="discount_code_input",
                    type="input",
                    label="Promo Discount Code",
                    allowed_actions=["set_value"]
                ),
                UIComponent(
                    id="payment_method_select",
                    type="select",
                    label="Payment Method Provider",
                    value="credit_card",
                    options=["credit_card", "paypal", "crypto"],
                    allowed_actions=["select_option", "set_value"]
                ),
                UIComponent(
                    id="item_quantity_input",
                    type="input",
                    label="Item Quantity Count",
                    value=1,
                    allowed_actions=["set_value"]
                ),
                UIComponent(
                    id="coupon_modal",
                    type="modal",
                    label="Available Coupons Modal Dialog",
                    allowed_actions=["open_modal", "close_modal"]
                ),
            ]
        )
        registry.update_context(context)

        # Test 1: Switch toggle on arbitrary component
        resp1 = await reasoner.generate_response("Enable express shipping", context, registry)
        assert len(resp1.actions) == 1
        assert resp1.actions[0].target_id == "express_shipping_toggle"
        assert resp1.actions[0].action_type == ActionType.TOGGLE_SWITCH

        # Test 2: Select option on arbitrary dropdown
        resp2 = await reasoner.generate_response("Switch payment method to paypal", context, registry)
        assert len(resp2.actions) == 1
        assert resp2.actions[0].target_id == "payment_method_select"
        assert resp2.actions[0].payload.get("value") == "paypal"

        # Test 3: Set numeric value on arbitrary input
        resp3 = await reasoner.generate_response("Change quantity to 4", context, registry)
        assert len(resp3.actions) == 1
        assert resp3.actions[0].target_id == "item_quantity_input"
        assert resp3.actions[0].payload.get("value") == 4

        # Test 4: Modal open on arbitrary modal
        resp4 = await reasoner.generate_response("Open the coupon modal", context, registry)
        assert len(resp4.actions) == 1
        assert resp4.actions[0].target_id == "coupon_modal"
        assert resp4.actions[0].action_type == ActionType.OPEN_MODAL

    asyncio.run(_run())
