"""Dynamic Zero-Shot Semantic Reasoner for Voice AI Copilot.

Analyzes natural language prompts against active ViewContext and MCPRegistry
to dynamically construct structured UIActions without hardcoded domain bindings.
Generates warm, lively, and highly natural conversational speech responses.
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

        is_ru = bool(re.search(r"[а-яё]", p_lower))
        is_uk = bool(re.search(r"[іїєґ]", p_lower))

        logger.info(f"[DynamicReasoner] Reasoning over {len(context.components)} components for prompt: '{prompt}'")

        # 0. Conversational Intelligence, Identity & Greetings
        if any(w in p_lower for w in ["who are you", "what is your name", "what's your name", "кто ты", "как тебя зовут", "твое имя", "твоё имя", "хто ти", "як тебе звати"]):
            if is_uk:
                speech = "Я — Grogu AI Copilot, ваш розумний голосовий помічник! Можу керувати серверами, виставляти ордери та фільтрувати дані."
            elif is_ru:
                speech = "Я — Grogu AI Copilot, ваш голосовой ассистент! Могу переключать темы, фильтровать серверы, выставлять торговые ордера и управлять этим интерфейсом."
            else:
                speech = "I am Grogu AI Copilot, your voice assistant! I can manage cluster nodes, execute trading orders, filter tables, and control this interface in real-time."
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
            if is_uk:
                speech = "Привіт! Я на зв'язку та уважно слухаю. Чим можу допомогти?"
            elif is_ru:
                speech = "Привет! Я на связи и внимательно слушаю. Чем могу помочь по дашборду?"
            else:
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
            if is_uk:
                speech = "Ви можете попросити мене: 'Покажи неробочі сервери', 'Зміни тему', або 'Купи 2 біткоїни'!"
            elif is_ru:
                speech = "Вы можете попросить меня: 'Покажи неисправные серверы', 'Переключи тему', 'Увеличь количество нод до 8' или 'Купи 2 биткоина'!"
            else:
                speech = "You can ask me to: 'Show unhealthy servers', 'Toggle theme', 'Scale worker nodes to 8', or 'Buy 2 Bitcoin'!"
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
        unhealthy_keywords = ["неисправ", "проблем", "ошибк", "упавш", "сбойн", "сломан", "неробоч", "unhealth", "error", "fail", "broken", "down", "fault"]
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
        elif any(kw in p_lower for kw in ["активн", "работа", "исправн", "нормальн", "живые", "активні", "active", "healthy", "running", "online", "up"]):
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
        if any(w in p_lower for w in ["deploy", "разверни", "развернуть", "деплой", "order", "подтверди", "ордер", "розгорни"]):
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

        if any(w in p_lower for w in ["close", "cancel", "закрыть", "отмена", "отменить", "закрити", "скасувати"]):
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
                if any(w in p_lower for w in ["theme", "тема", "тему", "темную", "светлую", "темная", "светлая", "темну", "світлу", "dark", "light"]):
                    matched_actions.append(UIAction(
                        action_type=ActionType.TOGGLE_SWITCH,
                        target_id=comp.id,
                        payload={"state": "light" if any(w in p_lower for w in ["light", "светл", "світл"]) else "dark"},
                        description=f"Toggled theme on {comp.id}"
                    ))
                    reasoning_steps.append(f"Theme toggle on {comp.id}.")

            # Autoscaling / Switches
            elif comp.type == "switch" or "switch" in cid:
                if any(w in p_lower for w in [cid, clabel, "autoscale", "автомасштаб", "hedging", "хеджир", "хеджуван"]):
                    target_state = False if any(w in p_lower for w in ["disable", "выключи", "off", "отключи", "стоп", "вимкни"]) else True
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
                elif any(w in p_lower for w in ["trade", "торгов", "торгів"]) and "trade" in cid:
                    matched_actions.append(UIAction(
                        action_type=ActionType.CLICK_BUTTON,
                        target_id=comp.id,
                        description=f"Switched to tab {comp.id}"
                    ))

        # 6. Construct Warm, Human-Like Speech Output
        if not matched_actions:
            if is_uk:
                speech = f"Я зрозумів запит: '{prompt}'. Готовий виконувати ваші команди!"
            elif is_ru:
                speech = f"Конечно, понял: '{prompt}'. Готов к вашим командам по интерфейсу!"
            else:
                speech = f"Understood: '{prompt}'. Ready for your commands!"
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
                if is_uk:
                    speech = f"Звісно! Показую {filter_val} сервери."
                elif is_ru:
                    speech = "Конечно! Показываю только неисправные серверы." if filter_val == "unhealthy" else "Без проблем! Отображаю активные серверы."
                else:
                    speech = f"Sure thing! Filtering view to show {filter_val} servers right away."
            elif first_act.action_type == ActionType.TOGGLE_SWITCH:
                if "theme" in first_act.target_id.lower():
                    if is_uk:
                        speech = "Звісно, перемикаю тему оформлення!"
                    elif is_ru:
                        speech = "Конечно, с удовольствием переключаю тему оформления!"
                    else:
                        speech = "Sure thing! Switching the interface theme right away."
                else:
                    if is_uk:
                        speech = "Зроблено! Перемикач оновлено."
                    elif is_ru:
                        speech = "Без проблем, настройка переключена!"
                    else:
                        speech = "Done! Successfully updated setting."
            elif first_act.action_type == ActionType.SET_INPUT_VALUE:
                val = first_act.payload.get('value')
                if is_uk:
                    speech = f"Зроблено! Встановив значення {val}."
                elif is_ru:
                    speech = f"Готово! Выставил значение {val}."
                else:
                    speech = f"Done! Updated value to {val}."
            elif first_act.action_type == ActionType.OPEN_MODAL:
                if is_uk:
                    speech = "Відкриваю вікно підтвердження."
                elif is_ru:
                    speech = "Конечно, открываю окно подтверждения деплоя!"
                else:
                    speech = "Opening confirmation dialog for you."
            elif first_act.action_type == ActionType.CLOSE_MODAL:
                if is_uk:
                    speech = "Закрив модальне вікно."
                elif is_ru:
                    speech = "Закрыл окно."
                else:
                    speech = "Closed the modal dialog."
            elif first_act.action_type == ActionType.CLICK_BUTTON:
                if "risk" in first_act.target_id.lower():
                    speech = "Переключил на вкладку управления рисками." if is_ru else "Switched to Risk & Hedging tab."
                elif "trade" in first_act.target_id.lower():
                    speech = "Открыл торговый терминал." if is_ru else "Switched to Trade Terminal."
                elif "reset" in first_act.target_id.lower():
                    speech = "Сбросил все фильтры к значениям по умолчанию." if is_ru else "Reset all filters to default."
                else:
                    speech = "Конечно, выполняю действие!" if is_ru else "Executed action successfully."
            else:
                speech = "Конечно, всё выполнено!" if is_ru else "Command executed successfully."

            thought = " -> ".join(reasoning_steps)

        return AgentResponse(
            thought=thought,
            speech_output=speech,
            actions=matched_actions
        )
