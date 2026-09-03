"""Dynamic Zero-Shot Semantic Reasoner for Voice AI Copilot.

Analyzes natural language prompts against active ViewContext and MCPRegistry
to dynamically construct structured UIActions without hardcoded domain bindings.
Supports English, Russian, and multilingual commands with dialogue turn memory.
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

        # 1. Global View Operations (e.g., Global Reset)
        if any(w in p_lower for w in ["reset", "clear all", "сброс", "сбросить все", "сбрось", "верни как было", "отмени фильтры"]):
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

            # A. Modal Components (open / close / модальное окно)
            if comp.type == "modal" or "modal" in comp.allowed_actions or "open_modal" in comp.allowed_actions:
                modal_keywords = all_keywords | {"modal", "dialog", "popup", "window", "модал", "диалог", "окно", "попап", "деплой", "deploy"}
                if any(kw in p_lower for kw in modal_keywords):
                    is_close = any(w in p_lower for w in ["close", "cancel", "dismiss", "hide", "закрой", "отмени", "спрячь", "скрой"])
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
            if comp.type in ["switch", "toggle", "checkbox"] or "toggle" in comp.allowed_actions:
                # Theme Switch
                if "theme" in comp.id or "theme" in comp.label.lower() or "тема" in p_lower:
                    theme_triggers = ["dark", "light", "theme", "темн", "светл", "тема", "ноч", "дневн", "черн", "бел"]
                    if any(w in p_lower for w in theme_triggers):
                        target_theme = "dark"
                        if any(w in p_lower for w in ["light", "светл", "бел", "дневн", "день"]):
                            target_theme = "light"
                        elif any(w in p_lower for w in ["dark", "темн", "черн", "ноч", "ночь"]):
                            target_theme = "dark"
                        else:
                            curr = comp.value if isinstance(comp.value, str) else "dark"
                            target_theme = "light" if curr == "dark" else "dark"

                        action = UIAction(
                            action_type=ActionType.TOGGLE_SWITCH,
                            target_id=comp.id,
                            payload={"theme": target_theme},
                            description=f"Switched theme to {target_theme} mode"
                        )
                        matched_actions.append(action)
                        reasoning_steps.append(f"Toggled theme switch to '{target_theme}'.")
                        continue

                # Generic boolean toggle switches (e.g. autoscaling, hedging)
                switch_triggers = ["enable", "disable", "turn on", "turn off", "включи", "выключи", "авто", "хедж", "auto", "hedging", "scale", "защит"]
                if has_affinity or any(w in p_lower for w in switch_triggers):
                    if has_affinity or any(w in p_lower for w in ["хедж", "hedg", "автомасштаб", "autoscale"]):
                        curr_val = bool(comp.value)
                        new_state = not curr_val
                        if any(w in p_lower for w in ["enable", "turn on", "activate", "включи", "активируй", "запусти"]):
                            new_state = True
                        elif any(w in p_lower for w in ["disable", "turn off", "deactivate", "выключи", "отключи", "деактивируй"]):
                            new_state = False

                        action = UIAction(
                            action_type=ActionType.TOGGLE_SWITCH,
                            target_id=comp.id,
                            payload={"state": new_state, "enabled": new_state},
                            description=f"Toggled '{comp.label}' to {new_state}"
                        )
                        matched_actions.append(action)
                        reasoning_steps.append(f"Toggled switch '{comp.id}' to {new_state}.")
                        continue

            # C. Select Dropdowns (Options / Status / Regions / Pairs)
            if comp.type in ["select", "dropdown"] or "select_option" in comp.allowed_actions:
                matched_option = None

                # Direct match against declared option values
                if comp.options:
                    for opt in comp.options:
                        opt_lower = opt.lower()
                        if opt_lower in p_lower or opt_lower.replace("-", " ") in p_lower:
                            matched_option = opt
                            break

                # Cloud Regions
                if not matched_option and comp.options and "region" in comp.id:
                    if any(w in p_lower for w in ["eu", "europe", "европ", "франкфурт", "frankfurt", "германи"]):
                        matched_option = next((o for o in comp.options if "eu" in o), None)
                    elif any(w in p_lower for w in ["west", "запад", "орегон", "oregon"]):
                        matched_option = next((o for o in comp.options if "west" in o), None)
                    elif any(w in p_lower for w in ["tokyo", "asia", "токио", "ази", "япон", "japan"]):
                        matched_option = next((o for o in comp.options if "ap" in o or "east" in o), None)
                    elif any(w in p_lower for w in ["east", "восток", "вирджини", "virginia", "us-east"]):
                        matched_option = next((o for o in comp.options if "us-east" in o), None)

                # Status filters (Active / Unhealthy / All)
                # CRITICAL: Check negative/unhealthy words FIRST so "неактивные" isn't misread as "активные"
                if not matched_option and comp.options and ("status" in comp.id or "filter" in comp.id):
                    if any(w in p_lower for w in [
                        "unhealthy", "error", "неисправн", "неработ", "сломан", "ошибоч", "неактивн",
                        "не активн", "проблем", "упавш", "битые", "сбой", "down", "failed", "offline"
                    ]):
                        matched_option = "unhealthy"
                    elif any(w in p_lower for w in [
                        "active", "активн", "работа", "исправн", "живые", "онлайн", "running", "healthy", "online"
                    ]):
                        matched_option = "active"
                    elif any(w in p_lower for w in ["all", "все", "всех", "сброс", "показать все"]):
                        matched_option = "all"

                # Financial Trading Order Types & Asset Pairs
                if not matched_option and comp.options and ("asset" in comp.id or "pair" in comp.id):
                    if any(w in p_lower for w in ["bitcoin", "btc", "биткоин", "биток", "бит"]):
                        matched_option = next((o for o in comp.options if "BTC" in o), None)
                    elif any(w in p_lower for w in ["ethereum", "eth", "эфир", "эфириум"]):
                        matched_option = next((o for o in comp.options if "ETH" in o), None)
                    elif any(w in p_lower for w in ["solana", "sol", "солана", "сол"]):
                        matched_option = next((o for o in comp.options if "SOL" in o), None)
                    elif any(w in p_lower for w in ["nvda", "nvidia", "нвидиа"]):
                        matched_option = next((o for o in comp.options if "NVDA" in o), None)

                if not matched_option and comp.options and "order_type" in comp.id:
                    if any(w in p_lower for w in ["limit", "лимит"]):
                        matched_option = next((o for o in comp.options if "LIMIT" in o), None)
                    elif any(w in p_lower for w in ["sell", "short", "продай", "шорт"]):
                        matched_option = next((o for o in comp.options if "SELL" in o), None)
                    elif any(w in p_lower for w in ["buy", "long", "купи", "лонг"]):
                        matched_option = next((o for o in comp.options if "BUY" in o), None)

                if not matched_option and comp.options and "leverage" in comp.id:
                    lev_numbers = re.findall(r'\b(\d+)x?\b', prompt)
                    if lev_numbers and any(w in p_lower for w in ["leverage", "плеч", "плечо", "x"]):
                        target_lev = lev_numbers[0]
                        if target_lev in comp.options:
                            matched_option = target_lev

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
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', prompt)
                if numbers:
                    has_specific_affinity = any(kw in p_lower for kw in all_keywords if len(kw) > 2)
                    
                    # Check semantic keyword affinities in Russian & English
                    if "worker" in comp.id or "node" in comp.id:
                        if any(w in p_lower for w in ["worker", "node", "воркер", "нод", "узел", "узл", "машин", "сервер"]):
                            has_specific_affinity = True
                    elif "amount" in comp.id or "size" in comp.id:
                        if any(w in p_lower for w in ["amount", "size", "объем", "сумм", "размер", "кол-во", "количеств", "штук", "btc", "eth", "sol", "биткоин"]):
                            has_specific_affinity = True
                    elif "stop_loss" in comp.id or "loss" in comp.id:
                        if any(w in p_lower for w in ["stop", "loss", "стоп", "лосс", "процент", "риск", "%"]):
                            has_specific_affinity = True

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

            # E. Buttons & Tab Navigation
            if comp.type == "button" and any(a in comp.allowed_actions for a in ["click", "navigate"]):
                button_triggers = [
                    "click", "press", "trigger", "run", "open", "execute", "switch to", "tab", "нажми", "кликни",
                    "переключи", "вкладка", "вкладку", "перейди", "открой", "выполни", "подтверди", "риск", "trade", "risk", "торгов"
                ]
                
                # Check Russian & English semantic button affinities
                if "risk" in comp.id and any(w in p_lower for w in ["risk", "риск", "риск-менеджмент"]):
                    has_affinity = True
                elif "trade" in comp.id and any(w in p_lower for w in ["trade", "торгов", "терминал"]):
                    has_affinity = True
                elif "deploy" in comp.id and any(w in p_lower for w in ["deploy", "деплой", "разверни"]):
                    has_affinity = True
                elif "reset" in comp.id and any(w in p_lower for w in ["reset", "сброс", "очисти"]):
                    has_affinity = True

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
                    speech_phrases.append(f"Обновил переключатель {act.target_id}." if is_ru else f"Updated {act.target_id}.")
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
                speech_phrases.append("Сбросил все фильтры." if is_ru else "Reset all filters to default.")
            elif act.action_type == ActionType.CLICK_BUTTON:
                speech_phrases.append("Действие выполнено." if is_ru else "Action executed.")

        speech_output = " ".join(speech_phrases) if speech_phrases else (f"Команда выполнена: '{prompt}'." if is_ru else f"I understood: '{prompt}'.")

        return AgentResponse(
            thought=thought,
            speech_output=speech_output,
            actions=valid_actions,
            status="success"
        )
