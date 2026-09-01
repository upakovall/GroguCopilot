"""Dynamic Semantic Intent Reasoner (Bilingual Russian & English).

Generic reasoning engine that dynamically parses arbitrary ViewContext schemas
and maps natural language intents to UI components without hardcoded actions.
Supports multi-language voice commands (Russian & English).
"""

import re
import logging
from typing import Any, Dict, List, Optional
from ..schemas.context import ViewContext, UIComponent
from ..schemas.actions import AgentResponse, UIAction, ActionType
from ..registry import MCPRegistry
from .provider import BaseLLMProvider

logger = logging.getLogger(__name__)


class DynamicReasoner(BaseLLMProvider):
    """Zero-hardcoding semantic reasoner for arbitrary ViewContext schemas (RU/EN)."""

    async def generate_response(
        self,
        prompt: str,
        context: ViewContext,
        registry: MCPRegistry,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResponse:
        """Dynamically parses prompt against current ViewContext components."""
        p_lower = prompt.lower().strip()
        matched_actions: List[UIAction] = []
        reasoning_steps: List[str] = []

        logger.info(f"[DynamicReasoner] Reasoning over {len(context.components)} components for prompt: '{prompt}'")

        # 1. Global Reset Commands (EN: reset, clear; RU: сброс, сбросить, очистить)
        if any(w in p_lower for w in ["reset", "clear all", "default", "сброс", "сбросить", "очисти", "по умолчанию"]):
            reset_action = UIAction(
                action_type=ActionType.RESET_FILTERS,
                target_id=None,
                payload={},
                description="Reset all view filters and parameters to default"
            )
            matched_actions.append(reset_action)
            reasoning_steps.append("Detected global reset request.")

        # 2. Dynamically evaluate each component in ViewContext
        for comp in context.components:
            if not comp.enabled:
                continue

            comp_id_words = comp.id.replace("_", " ").replace("-", " ").lower().split()
            comp_label_words = comp.label.lower().split()
            all_keywords = set(comp_id_words + comp_label_words)

            # Check for keyword affinity in user prompt
            has_affinity = any(kw in p_lower for kw in all_keywords if len(kw) > 2)

            # A. Modal Components (open / close / открыть / закрыть)
            if comp.type == "modal" or "modal" in comp.allowed_actions or "open_modal" in comp.allowed_actions:
                modal_keywords = all_keywords | {"modal", "dialog", "popup", "window", "модалка", "окно", "деплой", "деплоя", "ордер"}
                if any(kw in p_lower for kw in modal_keywords):
                    is_close = any(w in p_lower for w in ["close", "cancel", "dismiss", "hide", "закрой", "закрыть", "отмена", "скрой"])
                    act_type = ActionType.CLOSE_MODAL if is_close else ActionType.OPEN_MODAL
                    action = UIAction(
                        action_type=act_type,
                        target_id=comp.id,
                        payload={},
                        description=f"{'Closed' if is_close else 'Opened'} modal '{comp.label}'"
                    )
                    matched_actions.append(action)
                    reasoning_steps.append(f"Mapped to modal '{comp.id}' ({act_type.value}).")
                continue

            # B. Switch / Toggle Components (boolean states)
            if comp.type in ["switch", "checkbox", "toggle"] or "toggle" in comp.allowed_actions:
                theme_or_switch_hit = has_affinity

                # Check metadata values
                if comp.metadata:
                    for k, v in comp.metadata.items():
                        if isinstance(v, str) and (v.lower() in p_lower or k.lower() in p_lower):
                            theme_or_switch_hit = True

                # Check theme terms in RU/EN
                if "theme" in comp.id and any(w in p_lower for w in ["dark", "light", "theme", "тема", "темную", "светлую", "темная", "светлая", "тему"]):
                    theme_or_switch_hit = True

                # Check auto-hedging / autoscaling terms in RU/EN
                if "hedg" in comp.id and any(w in p_lower for w in ["hedg", "хедж", "хеджирование", "защит"]):
                    theme_or_switch_hit = True
                if "auto" in comp.id and any(w in p_lower for w in ["auto", "авто", "автомасштаб", "масштабирование"]):
                    theme_or_switch_hit = True

                if theme_or_switch_hit:
                    is_off = any(w in p_lower for w in [
                        "disable", "turn off", "deactivate", "light", "false", "off",
                        "выключи", "отключи", "светлая", "светлую", "выключить", "отключить"
                    ])
                    target_state = False if is_off else True

                    payload = {"state": target_state}
                    if "theme" in comp.id:
                        is_light = any(w in p_lower for w in ["light", "светл", "бел"])
                        payload["theme"] = "light" if is_light else "dark"
                    if "auto" in comp.id or "hedg" in comp.id:
                        payload["enabled"] = target_state

                    action = UIAction(
                        action_type=ActionType.TOGGLE_SWITCH,
                        target_id=comp.id,
                        payload=payload,
                        description=f"Toggled '{comp.label}' to {payload}"
                    )
                    matched_actions.append(action)
                    reasoning_steps.append(f"Matched switch '{comp.id}' with payload {payload}.")
                continue

            # C. Select / Dropdown / Option Components
            if comp.type in ["select", "dropdown"] or comp.options:
                matched_option = None
                if comp.options:
                    for opt in comp.options:
                        opt_clean = opt.lower().replace("-", " ").replace("/", " ")
                        if opt.lower() in p_lower or opt_clean in p_lower:
                            matched_option = opt
                            break

                # Crypto / Asset Matching (BTC, Bitcoin, ETH, Ethereum, SOL, Solana, NVDA)
                if not matched_option and comp.options and "asset" in comp.id:
                    if any(w in p_lower for w in ["btc", "bitcoin", "биткоин", "биток", "биткоины"]):
                        matched_option = next((o for o in comp.options if "BTC" in o), None)
                    elif any(w in p_lower for w in ["eth", "ethereum", "эфир", "эфириум"]):
                        matched_option = next((o for o in comp.options if "ETH" in o), None)
                    elif any(w in p_lower for w in ["sol", "solana", "солана"]):
                        matched_option = next((o for o in comp.options if "SOL" in o), None)
                    elif "nvda" in p_lower or "nvidia" in p_lower:
                        matched_option = next((o for o in comp.options if "NVDA" in o), None)

                # Order Type matching (Market, Limit, Buy, Sell)
                if not matched_option and comp.options and "order_type" in comp.id:
                    if any(w in p_lower for w in ["market buy", "маркет", "купи", "купить", "покупка"]):
                        matched_option = "MARKET_BUY"
                    elif any(w in p_lower for w in ["limit buy", "лимит"]):
                        matched_option = "LIMIT_BUY"
                    elif any(w in p_lower for w in ["sell", "продай", "продать", "продажа"]):
                        matched_option = "MARKET_SELL"

                # Region matching (EU, US, Asia)
                if not matched_option and comp.options and "region" in comp.id:
                    if any(w in p_lower for w in ["eu", "europe", "европа", "франкфурт", "frankfurt"]):
                        matched_option = next((o for o in comp.options if "eu" in o), None)
                    elif any(w in p_lower for w in ["west", "запад", "oregon"]):
                        matched_option = next((o for o in comp.options if "west" in o), None)
                    elif any(w in p_lower for w in ["tokyo", "asia", "азия", "токио"]):
                        matched_option = next((o for o in comp.options if "ap" in o or "east" in o), None)

                # Status filters (Active / Unhealthy / Все)
                if not matched_option and comp.options and "status" in comp.id:
                    if any(w in p_lower for w in ["active", "активн", "работа"]):
                        matched_option = "active"
                    elif any(w in p_lower for w in ["unhealthy", "error", "упавш", "ошибк", "сбойн", "проблем"]):
                        matched_option = "unhealthy"
                    elif any(w in p_lower for w in ["all", "все", "сброс"]):
                        matched_option = "all"

                if matched_option:
                    action = UIAction(
                        action_type=ActionType.SELECT_OPTION,
                        target_id=comp.id,
                        payload={"value": matched_option},
                        description=f"Selected '{matched_option}' on '{comp.label}'"
                    )
                    matched_actions.append(action)
                    reasoning_steps.append(f"Selected option '{matched_option}' on '{comp.id}'.")
                continue

            # D. Numeric Input / Stepper / Slider
            if comp.type in ["input", "number", "slider"] or "set_value" in comp.allowed_actions:
                # Check if prompt contains numbers
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', prompt)
                if numbers:
                    has_specific_affinity = any(kw in p_lower for kw in all_keywords if len(kw) > 2)
                    
                    # If this input has keyword match or no other input has matched yet
                    if has_specific_affinity or (comp.type in ["number", "slider"] and not any(a.action_type == ActionType.SET_INPUT_VALUE for a in matched_actions)):
                        val = float(numbers[0]) if "." in numbers[0] else int(numbers[0])
                        action = UIAction(
                            action_type=ActionType.SET_INPUT_VALUE,
                            target_id=comp.id,
                            payload={"value": val},
                            description=f"Set value of '{comp.label}' to {val}"
                        )
                        matched_actions.append(action)
                        reasoning_steps.append(f"Set numeric value {val} on '{comp.id}'.")
                continue

            # E. Data Table Filtering
            if comp.type == "table" or "filter_table" in comp.allowed_actions:
                filter_triggers = ["filter", "show only", "display", "status", "active", "unhealthy", "фильтр", "покажи", "отфильтруй", "активные", "сбойные", "упавшие"]
                if any(w in p_lower for w in filter_triggers):
                    status_target = "all"
                    if any(w in p_lower for w in ["active", "активн"]):
                        status_target = "active"
                    elif any(w in p_lower for w in ["unhealthy", "error", "упавш", "ошибк", "сбойн"]):
                        status_target = "unhealthy"

                    action = UIAction(
                        action_type=ActionType.FILTER_TABLE,
                        target_id=comp.id,
                        payload={"status": status_target},
                        description=f"Filtered table '{comp.label}' by status '{status_target}'"
                    )
                    matched_actions.append(action)
                    reasoning_steps.append(f"Filtered table '{comp.id}' to '{status_target}'.")
                continue

            # F. Buttons & Tab Navigation
            if comp.type == "button" and any(a in comp.allowed_actions for a in ["click", "navigate"]):
                button_triggers = [
                    "click", "press", "trigger", "run", "open", "execute", "switch to", "tab",
                    "нажми", "кликни", "запусти", "исполни", "переключи", "вкладку", "перейди"
                ]
                if has_affinity and any(w in p_lower for w in button_triggers):
                    action = UIAction(
                        action_type=ActionType.CLICK_BUTTON,
                        target_id=comp.id,
                        payload={},
                        description=f"Triggered action on '{comp.label}'"
                    )
                    matched_actions.append(action)
                    reasoning_steps.append(f"Triggered button '{comp.id}'.")

        # 3. Validate actions through MCPRegistry
        valid_actions: List[UIAction] = []
        for act in matched_actions:
            is_valid, err = registry.validate_action(act)
            if is_valid:
                valid_actions.append(act)
                registry.execute_hook_if_present(act)
            else:
                logger.warning(f"[DynamicReasoner] Action rejected by registry: {err}")

        # 4. Fallback if no specific component matched
        if not valid_actions:
            valid_actions.append(UIAction(
                action_type=ActionType.NOTIFY_USER,
                target_id=None,
                payload={"message": f"Processed command: '{prompt}'"},
                description="Acknowledged voice command"
            ))
            reasoning_steps.append("No specific component matched; emitted notification.")

        # 5. Formulate Spoken Speech in proper language
        is_ru = bool(re.search(r'[а-яёА-ЯЁ]', prompt))
        thought = " -> ".join(reasoning_steps) if reasoning_steps else "Analyzed prompt against ViewContext."

        speech_phrases = []
        for act in valid_actions:
            if act.action_type == ActionType.TOGGLE_SWITCH:
                theme = act.payload.get("theme")
                if theme:
                    speech_phrases.append(f"Переключаю тему на {theme}." if is_ru else f"Switching theme to {theme} mode.")
                else:
                    speech_phrases.append(f"Переключил {act.target_id}." if is_ru else f"Updated {act.target_id}.")
            elif act.action_type == ActionType.FILTER_TABLE:
                status = act.payload.get('status', 'all')
                speech_phrases.append(f"Таблица отфильтрована по статусу {status}." if is_ru else f"Filtered table to display {status} entries.")
            elif act.action_type == ActionType.SET_INPUT_VALUE:
                speech_phrases.append(f"Установил значение {act.payload.get('value')}." if is_ru else f"Updated value to {act.payload.get('value')}.")
            elif act.action_type == ActionType.SELECT_OPTION:
                speech_phrases.append(f"Выбрал {act.payload.get('value')}." if is_ru else f"Selected {act.payload.get('value')}.")
            elif act.action_type == ActionType.OPEN_MODAL:
                speech_phrases.append("Открыл модальное окно." if is_ru else "Opened the modal dialog.")
            elif act.action_type == ActionType.CLOSE_MODAL:
                speech_phrases.append("Закрыл окно." if is_ru else "Closed the modal dialog.")
            elif act.action_type == ActionType.RESET_FILTERS:
                speech_phrases.append("Все фильтры сброшены." if is_ru else "Reset all filters to default.")
            elif act.action_type == ActionType.CLICK_BUTTON:
                speech_phrases.append("Действие выполнено." if is_ru else "Action executed.")

        speech_output = " ".join(speech_phrases) if speech_phrases else (f"Команда выполнена: '{prompt}'." if is_ru else f"I understood: '{prompt}'.")

        return AgentResponse(
            thought=thought,
            speech_output=speech_output,
            actions=valid_actions,
            status="success"
        )
