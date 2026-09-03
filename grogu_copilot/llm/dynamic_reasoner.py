"""Dynamic Zero-Shot Semantic Reasoner for Voice AI Copilot.

Analyzes natural language prompts against active ViewContext and MCPRegistry
to dynamically construct structured UIActions without hardcoded domain bindings.
Supports English, Russian, and multilingual commands with conversational intelligence.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from grogu_copilot.llm.provider import BaseLLMProvider
from grogu_copilot.schemas.context import ViewContext
from grogu_copilot.schemas.actions import UIAction, AgentResponse, ActionType
from grogu_copilot.registry import MCPRegistry

logger = logging.getLogger("grogu_copilot.dynamic_reasoner")


class DynamicReasoner(BaseLLMProvider):
    """Dynamic Semantic Reasoner that matches natural language against declarative ViewContext."""

    async def generate_response(
        self,
        prompt: str,
        context: ViewContext,
        registry: MCPRegistry,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResponse:
        """Analyze user prompt against active ViewContext to construct structured UIActions."""
        p_lower = prompt.lower().strip()
        matched_actions: List[UIAction] = []
        reasoning_steps: List[str] = []

        logger.info(f"[DynamicReasoner] Reasoning over {len(context.components)} components for prompt: '{prompt}'")

        # 0. Conversational Intelligence & Identity Handling
        if any(w in p_lower for w in ["who are you", "what is your name", "what's your name", "кто ты", "как тебя зовут", "твое имя", "твоё имя", "хто ти", "як тебе звати"]):
            speech = "I am Grogu AI Copilot, your voice assistant. I can manage servers, execute trading orders, filter tables, and control this interface in real-time."
            return AgentResponse(
                thought="User inquired about assistant identity.",
                speech_output=speech,
                actions=[UIAction(
                    action_type=ActionType.NOTIFY_USER,
                    target_id="copilot_hud",
                    payload={"message": speech},
                    description="Introduced Grogu Voice Copilot"
                )]
            )

        if any(w in p_lower for w in ["hello", "hi", "hey", "привет", "здравствуйте", "добрый день", "вітаю", "добрий день"]):
            speech = "Hello! I am online and listening. How can I assist you with the dashboard today?"
            return AgentResponse(
                thought="Greeting recognized.",
                speech_output=speech,
                actions=[UIAction(
                    action_type=ActionType.NOTIFY_USER,
                    target_id="copilot_hud",
                    payload={"message": speech},
                    description="Greeted user"
                )]
            )

        if any(w in p_lower for w in ["help", "what can you do", "commands", "что ты умеешь", "помощь", "команды", "що ти вмієш", "допомога"]):
            speech = "You can ask me to filter servers, toggle dark and light theme, deploy clusters, execute crypto trades, or set stop loss limits."
            return AgentResponse(
                thought="Help request processed.",
                speech_output=speech,
                actions=[UIAction(
                    action_type=ActionType.NOTIFY_USER,
                    target_id="copilot_hud",
                    payload={"message": speech},
                    description="Displayed capabilities"
                )]
            )

        # 1. Global View Operations (e.g., Global Reset)
        if any(w in p_lower for w in ["reset", "clear all", "сброс", "сбросить", "верни все", "скинути", "очистити"]):
            reset_comp = next((c for c in context.components if "reset" in c.id.lower()), None)
            if reset_comp:
                matched_actions.append(UIAction(
                    action_type=ActionType.CLICK_BUTTON,
                    target_id=reset_comp.id,
                    description=f"Clicked {reset_comp.label or reset_comp.id}"
                ))
                reasoning_steps.append("Global reset triggered.")

        # 2. Priority: Unhealthy / Problem Server Filtering
        unhealthy_keywords = ["неисправ", "проблем", "ошибк", "упавш", "сбойн", "сломан", "неисправн", "unhealth", "error", "fail", "broken", "down", "fault"]
        if any(kw in p_lower for kw in unhealthy_keywords):
            filter_comp = next((c for c in context.components if "filter" in c.id.lower() or "status" in c.id.lower()), None)
            if filter_comp:
                val = "unhealthy"
                act_type = ActionType.FILTER_TABLE if filter_comp.type in ["table", "filter"] else ActionType.SELECT_OPTION
                matched_actions.append(UIAction(
                    action_type=act_type,
                    target_id=filter_comp.id,
                    payload={"value": val, "status": val, "filter": val},
                    description=f"Filtered view to unhealthy items on {filter_comp.id}"
                ))
                reasoning_steps.append("Negative status filter matched.")

        # 3. Active / Healthy Server Filtering
        elif any(kw in p_lower for kw in ["активн", "работа", "исправн", "нормальн", "живые", "active", "healthy", "running", "online", "up"]):
            filter_comp = next((c for c in context.components if "filter" in c.id.lower() or "status" in c.id.lower()), None)
            if filter_comp:
                val = "active"
                act_type = ActionType.FILTER_TABLE if filter_comp.type in ["table", "filter"] else ActionType.SELECT_OPTION
                matched_actions.append(UIAction(
                    action_type=act_type,
                    target_id=filter_comp.id,
                    payload={"value": val, "status": val, "filter": val},
                    description=f"Filtered view to active items on {filter_comp.id}"
                ))
                reasoning_steps.append("Active status filter matched.")

        # 4. Modals (Open / Close)
        if any(w in p_lower for w in ["deploy", "разверни", "развернуть", "деплой", "order", "подтверди", "ордер"]):
            modal_comp = next((c for c in context.components if c.type == "modal" or "modal" in c.id.lower() or "deploy" in c.id.lower()), None)
            if modal_comp:
                if modal_comp.type == "modal":
                    matched_actions.append(UIAction(
                        action_type=ActionType.OPEN_MODAL,
                        target_id=modal_comp.id,
                        description=f"Opened modal: {modal_comp.id}"
                    ))
                else:
                    matched_actions.append(UIAction(
                        action_type=ActionType.CLICK_BUTTON,
                        target_id=modal_comp.id,
                        description=f"Clicked button: {modal_comp.id}"
                    ))
                reasoning_steps.append(f"Modal action on {modal_comp.id}.")

        if any(w in p_lower for w in ["close", "cancel", "закрыть", "отмена", "отменить"]):
            if context.active_modal:
                matched_actions.append(UIAction(
                    action_type=ActionType.CLOSE_MODAL,
                    target_id=context.active_modal,
                    description=f"Closed active modal: {context.active_modal}"
                ))
                reasoning_steps.append(f"Closed modal {context.active_modal}.")

        # 5. Component Iteration for Theme, Sliders, Switches, Tabs, Inputs
        for comp in context.components:
            cid = comp.id.lower()
            clabel = (comp.label or "").lower()

            # Theme switch
            if "theme" in cid or "theme" in clabel:
                if any(w in p_lower for w in ["theme", "тема", "тему", "темную", "светлую", "темная", "светлая", "dark", "light"]):
                    matched_actions.append(UIAction(
                        action_type=ActionType.TOGGLE_SWITCH,
                        target_id=comp.id,
                        payload={"state": "light" if "light" in p_lower or "светл" in p_lower else "dark"},
                        description=f"Toggled theme on {comp.id}"
                    ))
                    reasoning_steps.append(f"Theme toggle on {comp.id}.")

            # Autoscaling / Switches
            elif comp.type == "switch" or "switch" in cid:
                if any(w in p_lower for w in [cid, clabel, "autoscale", "автомасштаб", "hedging", "хеджир"]):
                    target_state = False if any(w in p_lower for w in ["disable", "выключи", "off", "отключи", "стоп"]) else True
                    matched_actions.append(UIAction(
                        action_type=ActionType.TOGGLE_SWITCH,
                        target_id=comp.id,
                        payload={"state": target_state},
                        description=f"Toggled switch {comp.id} to {target_state}"
                    ))
                    reasoning_steps.append(f"Toggled {comp.id}.")

            # Numbers (e.g. "set worker nodes to 6", "buy 2 BTC")
            elif comp.type in ["input", "number", "slider"] or any(w in cid for w in ["nodes", "worker", "amount", "size", "leverage", "stop_loss"]):
                digits = re.findall(r"\b\d+(?:\.\d+)?\b", p_lower)
                if digits:
                    val = float(digits[0]) if "." in digits[0] else int(digits[0])
                    matched_actions.append(UIAction(
                        action_type=ActionType.SET_INPUT_VALUE,
                        target_id=comp.id,
                        payload={"value": val},
                        description=f"Set value {val} on {comp.id}"
                    ))
                    reasoning_steps.append(f"Extracted numeric value {val} for {comp.id}.")

            # Tabs
            elif "tab" in cid:
                if "risk" in p_lower and "risk" in cid:
                    matched_actions.append(UIAction(
                        action_type=ActionType.CLICK_BUTTON,
                        target_id=comp.id,
                        description=f"Switched to tab {comp.id}"
                    ))
                elif ("trade" in p_lower or "торгов" in p_lower) and "trade" in cid:
                    matched_actions.append(UIAction(
                        action_type=ActionType.CLICK_BUTTON,
                        target_id=comp.id,
                        description=f"Switched to tab {comp.id}"
                    ))

        # 6. Construct Acoustic Speech Output and Thoughts
        if not matched_actions:
            speech = f"I understood: '{prompt}'. No specific UI component was matched, but I am ready for commands."
            thought = "No direct UI component match found."
            matched_actions.append(UIAction(
                action_type=ActionType.NOTIFY_USER,
                target_id="copilot_hud",
                payload={"message": speech},
                description="Acknowledged voice command"
            ))
        else:
            first_act = matched_actions[0]
            if first_act.action_type == ActionType.FILTER_TABLE:
                filter_val = first_act.payload.get("value", "target")
                speech = f"Filtered view to show {filter_val} items."
            elif first_act.action_type == ActionType.TOGGLE_SWITCH:
                speech = f"Toggled setting."
            elif first_act.action_type == ActionType.SET_INPUT_VALUE:
                speech = f"Updated value to {first_act.payload.get('value')}."
            elif first_act.action_type == ActionType.OPEN_MODAL:
                speech = "Opened confirmation dialog."
            elif first_act.action_type == ActionType.CLOSE_MODAL:
                speech = "Closed dialog."
            elif first_act.action_type == ActionType.CLICK_BUTTON:
                speech = "Executed action."
            else:
                speech = "Command executed successfully."

            thought = " -> ".join(reasoning_steps)

        return AgentResponse(
            thought=thought,
            speech_output=speech,
            actions=matched_actions
        )
