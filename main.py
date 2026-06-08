import copy
import json
import os
from datetime import datetime
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner

try:
    from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
except Exception:  # pragma: no cover
    ProviderOpenAIOfficial = None

try:
    from openai import LengthFinishReasonError
except Exception:  # pragma: no cover
    LengthFinishReasonError = None


DEFAULT_CONFIG = {
    "enable_context_compression": True,
    "compression_threshold": 35000,
    "provider_compression_threshold": 50000,
    "keep_recent_rounds": 4,
    "summary_max_tokens": 512,
    "max_older_messages_for_summary": 20,
    "enable_learning": True,
    "enable_provider_patch": True,
    "enable_retry_on_length_error": True,
    "min_completion_tokens": 2048,
    "retry_completion_tokens": 4096,
    "force_reasoning_effort_low": True,
    "sanitize_tool_call_pairs": True,
    "summary_prompt_prefix": "请用简洁的中文（200字以内）总结以下多轮对话的关键信息、决策和上下文，不要添加主观评价，只保留事实：\n\n",
}


@register(
    "astrbot_plugin_length_error_handler",
    "Kurisu",
    "自动处理 LengthFinishReasonError + 输出预算修正 + Provider 兜底重试（1.0.6）",
    "1.0.6",
    "",
)
class LengthErrorHandlerPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        self.log_dir = os.path.join("data", "plugins_data", "length_error_handler")
        os.makedirs(self.log_dir, exist_ok=True)
        self.error_log_path = os.path.join(self.log_dir, "length_error.jsonl")
        self.config_path = os.path.join(self.log_dir, "config.json")
        self.cfg = self._load_config(context)
        self._activate()
        self._patch_runner()
        self._patch_openai_provider()

    def _activate(self):
        ToolLoopAgentRunner._length_error_handler_active_plugin = self
        if ProviderOpenAIOfficial is not None:
            ProviderOpenAIOfficial._length_error_handler_active_plugin = self

    def _load_config(self, context: Context) -> dict[str, Any]:
        cfg = DEFAULT_CONFIG.copy()
        astrbot_cfg = getattr(context, "config", None)
        if isinstance(astrbot_cfg, dict):
            for key in DEFAULT_CONFIG:
                if key in astrbot_cfg:
                    cfg[key] = astrbot_cfg[key]
            logger.info("[LengthErrorHandler] 已从 AstrBot WebUI 配置加载参数")
            return cfg

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                cfg.update(user_cfg)
                for key, value in DEFAULT_CONFIG.items():
                    user_cfg.setdefault(key, value)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(user_cfg, f, ensure_ascii=False, indent=2)
                logger.info("[LengthErrorHandler] 已从本地 config.json 加载用户配置")
            except Exception as exc:
                logger.warning(f"[LengthErrorHandler] 读取 config.json 失败，使用默认配置: {exc}")
        else:
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.warning(f"[LengthErrorHandler] 写入默认 config.json 失败: {exc}")
        return cfg

    @staticmethod
    def _active_from_runner():
        return getattr(ToolLoopAgentRunner, "_length_error_handler_active_plugin", None)

    @staticmethod
    def _active_from_provider():
        if ProviderOpenAIOfficial is None:
            return None
        return getattr(ProviderOpenAIOfficial, "_length_error_handler_active_plugin", None)

    def _patch_runner(self):
        marker = "1.0.6"
        if getattr(ToolLoopAgentRunner, "_length_error_handler_runner_patch_version", None) == marker:
            logger.info("[LengthErrorHandler] Runner v1.0.6 已注入，刷新活动实例")
            return

        original_iter = getattr(
            ToolLoopAgentRunner,
            "_length_error_handler_original_iter_v106",
            ToolLoopAgentRunner._iter_llm_responses_with_fallback,
        )
        ToolLoopAgentRunner._length_error_handler_original_iter_v106 = original_iter

        async def patched_iter(runner_self, *args, **kwargs):
            plugin = LengthErrorHandlerPlugin._active_from_runner()
            if plugin is not None:
                await plugin._maybe_compress_runner_context(runner_self, force=False)
                plugin._ensure_runner_budget(runner_self, retry=False)
            try:
                async for resp in original_iter(runner_self, *args, **kwargs):
                    yield resp
            except Exception as exc:
                plugin = LengthErrorHandlerPlugin._active_from_runner()
                if plugin is None or not plugin._is_length_error(exc):
                    raise
                logger.warning("[LengthErrorHandler] Runner 捕获 length 截断，压缩并扩大输出预算后重试")
                plugin._log_error(exc, runner_self, "runner")
                plugin._ensure_runner_budget(runner_self, retry=True)
                await plugin._maybe_compress_runner_context(runner_self, force=True)
                async for resp in original_iter(runner_self, *args, **kwargs):
                    yield resp

        ToolLoopAgentRunner._iter_llm_responses_with_fallback = patched_iter
        ToolLoopAgentRunner._length_error_handler_patched = True
        ToolLoopAgentRunner._length_error_handler_runner_patch_version = marker
        logger.info("[LengthErrorHandler] 已注入 Runner v1.0.6")

    def _patch_openai_provider(self):
        if not self.cfg.get("enable_provider_patch", True):
            logger.info("[LengthErrorHandler] Provider patch 已关闭")
            return
        if ProviderOpenAIOfficial is None:
            logger.warning("[LengthErrorHandler] 未找到 ProviderOpenAIOfficial，跳过 Provider patch")
            return

        marker = "1.0.6"
        if getattr(ProviderOpenAIOfficial, "_length_error_handler_provider_patch_version", None) == marker:
            logger.info("[LengthErrorHandler] OpenAI Provider v1.0.6 已注入，刷新活动实例")
            return

        original_query = getattr(
            ProviderOpenAIOfficial,
            "_length_error_handler_original_query_v106",
            ProviderOpenAIOfficial._query,
        )
        original_query_stream = getattr(
            ProviderOpenAIOfficial,
            "_length_error_handler_original_query_stream_v106",
            ProviderOpenAIOfficial._query_stream,
        )
        ProviderOpenAIOfficial._length_error_handler_original_query_v106 = original_query
        ProviderOpenAIOfficial._length_error_handler_original_query_stream_v106 = original_query_stream

        async def patched_query(provider_self, payloads: dict, tools):
            plugin = LengthErrorHandlerPlugin._active_from_provider()
            if plugin is None:
                return await original_query(provider_self, payloads, tools)
            first_payload = plugin._prepare_payload(payloads, retry=False, force_compress=False)
            try:
                return await original_query(provider_self, first_payload, tools)
            except Exception as exc:
                if not plugin._should_retry(exc):
                    raise
                logger.warning("[LengthErrorHandler] OpenAI Provider 捕获 length 截断，压缩 messages + 扩大输出预算后重试")
                plugin._log_error(exc, provider_self, "provider_query")
                retry_payload = plugin._prepare_payload(payloads, retry=True, force_compress=True)
                return await original_query(provider_self, retry_payload, tools)

        async def patched_query_stream(provider_self, payloads: dict, tools):
            plugin = LengthErrorHandlerPlugin._active_from_provider()
            if plugin is None:
                async for resp in original_query_stream(provider_self, payloads, tools):
                    yield resp
                return
            first_payload = plugin._prepare_payload(payloads, retry=False, force_compress=False)
            try:
                async for resp in original_query_stream(provider_self, first_payload, tools):
                    yield resp
            except Exception as exc:
                if not plugin._should_retry(exc):
                    raise
                logger.warning("[LengthErrorHandler] OpenAI Provider Stream 捕获 length 截断，压缩 messages + 扩大输出预算后重试")
                plugin._log_error(exc, provider_self, "provider_stream")
                retry_payload = plugin._prepare_payload(payloads, retry=True, force_compress=True)
                async for resp in original_query_stream(provider_self, retry_payload, tools):
                    yield resp

        ProviderOpenAIOfficial._query = patched_query
        ProviderOpenAIOfficial._query_stream = patched_query_stream
        ProviderOpenAIOfficial._length_error_handler_provider_patch_version = marker
        logger.info("[LengthErrorHandler] 已注入 OpenAI Provider v1.0.6")

    def _is_length_error(self, exc: Exception) -> bool:
        if LengthFinishReasonError is not None and isinstance(exc, LengthFinishReasonError):
            return True
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(
            pat in text
            for pat in (
                "lengthfinishreasonerror",
                "length limit was reached",
                "finish_reason='length'",
                'finish_reason="length"',
                "finish reason length",
                "max_completion_tokens",
                "max_tokens",
            )
        ) and ("length" in text or "token" in text)

    def _should_retry(self, exc: Exception) -> bool:
        return bool(self.cfg.get("enable_retry_on_length_error", True)) and self._is_length_error(exc)

    def _prepare_payload(self, payloads: dict, retry: bool, force_compress: bool) -> dict:
        prepared = copy.deepcopy(payloads)
        self._ensure_payload_budget(prepared, retry=retry)
        self._normalize_reasoning(prepared)
        if self.cfg.get("enable_context_compression", True):
            self._maybe_compress_payload_messages(prepared, force=force_compress)
        return prepared

    def _ensure_payload_budget(self, payloads: dict, retry: bool):
        target = int(self.cfg.get("retry_completion_tokens" if retry else "min_completion_tokens", 4096 if retry else 2048))
        for key in ("max_completion_tokens", "max_tokens"):
            current = payloads.get(key)
            if isinstance(current, int) and current < target:
                payloads[key] = target
                logger.info(f"[LengthErrorHandler] payload.{key} 调整: {current} -> {target}")
                return
        if retry:
            payloads["max_tokens"] = target
            logger.info(f"[LengthErrorHandler] retry payload.max_tokens 设置为 {target}")

    def _normalize_reasoning(self, payloads: dict):
        if not self.cfg.get("force_reasoning_effort_low", True):
            return
        low_values = {"low", "minimal", "none"}
        current = payloads.get("reasoning_effort")
        if isinstance(current, str) and current.lower() not in low_values:
            payloads["reasoning_effort"] = "low"
            logger.info(f"[LengthErrorHandler] reasoning_effort 调整: {current} -> low")
        extra_body = payloads.get("extra_body")
        if isinstance(extra_body, dict):
            current = extra_body.get("reasoning_effort")
            if isinstance(current, str) and current.lower() not in low_values:
                extra_body["reasoning_effort"] = "low"
                logger.info(f"[LengthErrorHandler] extra_body.reasoning_effort 调整: {current} -> low")

    def _maybe_compress_payload_messages(self, payloads: dict, force: bool):
        messages = payloads.get("messages")
        if not isinstance(messages, list):
            return
        estimated = self._estimate_messages_tokens(messages)
        threshold = int(self.cfg.get("provider_compression_threshold", 50000))
        if not force and estimated < threshold:
            return
        keep_recent = max(2, int(self.cfg.get("keep_recent_rounds", 4)))
        compressed = self._compress_messages(messages, keep_recent)
        if len(compressed) < len(messages):
            payloads["messages"] = compressed
            logger.info(f"[LengthErrorHandler] Provider messages 已压缩: {len(messages)} -> {len(compressed)}，估算 tokens≈{estimated}")

    async def _maybe_compress_runner_context(self, runner: Any, force: bool):
        try:
            messages = None
            target_attr = None
            keep_recent = max(2, int(self.cfg.get("keep_recent_rounds", 4)))
            for attr in ("messages", "context", "history", "chat_history"):
                value = getattr(runner, attr, None)
                if isinstance(value, list) and len(value) > keep_recent + 2:
                    messages = value
                    target_attr = attr
                    break
            if messages is None:
                return
            estimated = self._estimate_messages_tokens(messages)
            threshold = int(self.cfg.get("compression_threshold", 35000))
            if not force and estimated < threshold:
                return
            compressed = self._compress_messages(messages, keep_recent)
            if len(compressed) < len(messages):
                setattr(runner, target_attr, compressed)
                logger.info(f"[LengthErrorHandler] Runner 上下文已压缩: {len(messages)} -> {len(compressed)}，估算 tokens≈{estimated}")
        except Exception as exc:
            logger.warning(f"[LengthErrorHandler] Runner 上下文压缩失败: {exc}")

    def _compress_messages(self, messages: list[Any], keep_recent: int) -> list[Any]:
        def _get_role(m):
            return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        def _get_content(m):
            return m.get("content") if isinstance(m, dict) else getattr(m, "content", None)

        system_msgs = [m for m in messages if _get_role(m) == "system"]
        non_system = [m for m in messages if _get_role(m) != "system"]
        if len(non_system) <= keep_recent:
            return messages
        recent = non_system[-keep_recent:]
        older = non_system[:-keep_recent]
        protected_recent = self._include_required_tool_call_messages(older, recent)
        max_older = int(self.cfg.get("max_older_messages_for_summary", 20))
        lines = []
        for msg in older[-max_older:]:
            role = _get_role(msg) or "unknown"
            text = self._content_to_text(_get_content(msg))
            if text:
                lines.append(f"- {role}: {text[:180]}")
        summary = (
            f"[LengthErrorHandler 自动压缩] 已折叠 {len(older)} 条较早消息，仅保留摘要和最近 {len(protected_recent)} 条必要消息。\n"
            + "\n".join(lines)
        )[:4000]
        compressed = system_msgs + [{"role": "system", "content": summary}] + protected_recent
        return self._repair_tool_call_pairs(compressed)

    def _include_required_tool_call_messages(self, older: list[Any], recent: list[Any]) -> list[Any]:
        """Keep assistant tool-call messages needed by retained tool results."""
        if not self.cfg.get("sanitize_tool_call_pairs", True):
            return recent

        required_ids = self._collect_tool_result_ids(recent)
        if not required_ids:
            return recent

        found_ids = self._collect_assistant_tool_call_ids(recent)
        missing_ids = required_ids - found_ids
        if not missing_ids:
            return recent

        protected: list[Any] = []
        for msg in older:
            call_ids = self._assistant_tool_call_ids_from_message(msg)
            if call_ids & missing_ids:
                protected.append(msg)
                missing_ids -= call_ids
            if not missing_ids:
                break

        if protected:
            logger.info(
                "[LengthErrorHandler] 压缩时额外保留 %s 条 assistant tool_call 消息以维持 call_id 配对",
                len(protected),
            )
        return protected + recent

    def _repair_tool_call_pairs(self, messages: Any) -> Any:
        """Pure forward scan: group assistants with their immediately following
        tool messages, keep only complete pairs, discard everything else.

        Chat Completions requires every assistant message with tool_calls to be
        immediately followed by tool messages for each tool_call_id. Context
        compression can cut either side of that pair.

        This implementation uses two passes:
        Pass 1 — forward scan, keep system/user/plain-text + complete pairs.
        Pass 2 — collect retained assistant call_ids, remove orphan tool messages
                 whose call_ids refer to assistants that were dropped in Pass 1.
        """
        if not isinstance(messages, list):
            return messages

        # ---- pass 1: forward walk, keep complete pairs only ----
        repaired: list[Any] = []
        dropped_calls = 0
        dropped_outputs = 0

        idx = 0
        while idx < len(messages):
            msg = messages[idx]

            # system / user / plain assistant → always keep
            if not self._is_assistant_with_tool_calls(msg) and not self._is_tool_output_message(msg):
                repaired.append(msg)
                idx += 1
                continue

            # assistant with tool_calls
            if self._is_assistant_with_tool_calls(msg):
                assistant_call_ids = self._assistant_tool_call_ids_from_message(msg)

                # collect call_ids from immediately following tool messages
                answered_ids: set[str] = set()
                tool_msgs: list[Any] = []
                cursor = idx + 1
                while cursor < len(messages):
                    next_msg = messages[cursor]
                    if not self._is_tool_output_message(next_msg):
                        break
                    answered_ids.update(self._tool_result_ids_from_message(next_msg))
                    tool_msgs.append(next_msg)
                    cursor += 1

                matched = assistant_call_ids & answered_ids
                if not matched:
                    # no matching tool results → drop entire assistant
                    dropped_calls += len(assistant_call_ids)
                    idx = cursor
                    continue

                # keep assistant with only matched tool_calls
                self._filter_assistant_tool_calls(msg, matched)
                repaired.append(msg)

                # keep only tool messages that belong to this assistant
                for tm in tool_msgs:
                    tm_ids = self._tool_result_ids_from_message(tm)
                    if tm_ids and (tm_ids & matched):
                        self._filter_tool_output_parts(tm, matched)
                        repaired.append(tm)
                    else:
                        dropped_outputs += len(tm_ids)

                idx = cursor
                continue

            # orphan tool message (no preceding assistant with tool_calls)
            # defer decision to pass 2 — may belong to an assistant earlier in history
            repaired.append(msg)
            idx += 1

        # ---- pass 2: collect retained assistant call_ids and remove orphan tool messages ----
        retained_assistant_ids: set[str] = set()
        for msg in repaired:
            if self._is_assistant_with_tool_calls(msg):
                retained_assistant_ids.update(self._assistant_tool_call_ids_from_message(msg))

        if retained_assistant_ids:
            final: list[Any] = []
            for msg in repaired:
                if self._is_tool_output_message(msg):
                    tm_ids = self._tool_result_ids_from_message(msg)
                    if not tm_ids or not (tm_ids & retained_assistant_ids):
                        dropped_outputs += max(1, len(tm_ids))
                        continue
                final.append(msg)
            repaired = final

        if dropped_calls:
            logger.warning(
                "[LengthErrorHandler] 已移除 %s 个无匹配 tool 结果的 assistant tool_call",
                dropped_calls,
            )
        if dropped_outputs:
            logger.warning(
                "[LengthErrorHandler] 已移除 %s 个孤立 tool/function 输出，避免 No tool call found 400",
                dropped_outputs,
            )
        messages[:] = repaired
        return messages

    def _fill_unanswered_assistant_tool_calls(self, messages: list[Any]) -> int:
        """Insert a stub tool message after every assistant tool_call that is not
        already followed by a matching tool/function result, so the API always
        sees a complete call→result pair."""
        filled_count = 0
        # Iterate backwards so indices stay stable while inserting.
        for index in range(len(messages) - 1, -1, -1):
            msg = messages[index]
            if not self._is_assistant_with_tool_calls(msg):
                continue
            answered_ids = self._following_tool_result_ids(messages, index)
            missing_ids = self._assistant_tool_call_ids_from_message(msg) - answered_ids
            if not missing_ids:
                continue

            # Build one stub tool message per missing call_id.
            stub_parts = []
            for call_id in sorted(missing_ids):
                stub_parts.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": (
                        "{\"stub\": true, \"note\": \"工具结果已在早前轮次被上下文压缩移除，"
                        "此条为自动补填的占位结果，请 LLM 忽略具体内容并基于上下文继续。\"}"
                    ),
                })
            stub_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": next(iter(missing_ids)),
                "content": stub_parts,
            }
            # Insert right after the assistant message.
            messages.insert(index + 1, stub_msg)
            filled_count += len(missing_ids)

        return filled_count

    @staticmethod
    def _is_assistant_with_tool_calls(msg: Any) -> bool:
        if isinstance(msg, dict):
            if msg.get("role") != "assistant":
                return False
            tool_calls = msg.get("tool_calls")
            content = msg.get("content")
        else:
            if getattr(msg, "role", None) != "assistant":
                return False
            tool_calls = getattr(msg, "tool_calls", None)
            content = getattr(msg, "content", None)
        if isinstance(tool_calls, list) and tool_calls:
            return True
        if isinstance(content, list):
            return any(
                isinstance(part, dict) and part.get("type") in {"function_call", "tool_use"}
                for part in content
            )
        return False

    def _following_tool_result_ids(self, messages: list[Any], index: int) -> set[str]:
        ids: set[str] = set()
        cursor = index + 1
        while cursor < len(messages):
            next_msg = messages[cursor]
            if not self._is_tool_output_message(next_msg):
                break
            ids.update(self._tool_result_ids_from_message(next_msg))
            cursor += 1
        return ids

    def _filter_assistant_tool_calls(self, msg: dict[str, Any], answered_ids: set[str]) -> int:
        dropped = 0
        is_dict = isinstance(msg, dict)
        tool_calls = msg.get("tool_calls") if is_dict else getattr(msg, "tool_calls", None)
        if isinstance(tool_calls, list):
            kept_tool_calls = []
            for tool_call in tool_calls:
                call_id = None
                if isinstance(tool_call, dict):
                    call_id = tool_call.get("id") or tool_call.get("call_id")
                else:
                    call_id = getattr(tool_call, "id", None) or getattr(tool_call, "call_id", None)
                if call_id and str(call_id) in answered_ids:
                    kept_tool_calls.append(tool_call)
                else:
                    dropped += 1
            if kept_tool_calls:
                if is_dict:
                    msg["tool_calls"] = kept_tool_calls
                else:
                    setattr(msg, "tool_calls", kept_tool_calls)
            elif tool_calls:
                if is_dict:
                    msg.pop("tool_calls", None)
                    if msg.get("content") is None:
                        msg["content"] = ""
                else:
                    try:
                        delattr(msg, "tool_calls")
                    except AttributeError:
                        pass
                    if getattr(msg, "content", None) is None:
                        try:
                            setattr(msg, "content", "")
                        except AttributeError:
                            pass

        content = msg.get("content") if is_dict else getattr(msg, "content", None)
        if isinstance(content, list):
            filtered_parts = []
            for part in content:
                if isinstance(part, dict) and self._is_tool_call_part(part):
                    part_ids = set()
                    for key in ("call_id", "id", "tool_call_id"):
                        value = part.get(key)
                        if value:
                            part_ids.add(str(value))
                    if part_ids and not (part_ids & answered_ids):
                        dropped += 1
                        continue
                filtered_parts.append(part)
            if is_dict:
                msg["content"] = filtered_parts
            else:
                try:
                    setattr(msg, "content", filtered_parts)
                except AttributeError:
                    pass
        return dropped

    def _drop_all_tool_outputs(self, messages: list[Any]) -> list[Any]:
        repaired: list[Any] = []
        dropped = 0
        for msg in messages:
            if self._is_tool_output_message(msg):
                dropped += 1
                continue
            repaired.append(msg)
        if dropped:
            logger.warning(
                "[LengthErrorHandler] 未保留任何 assistant tool_call，已移除 %s 条 tool/function 输出消息",
                dropped,
            )
        messages[:] = repaired
        return messages

    def _filter_tool_output_parts(self, msg: Any, valid_ids: set[str]) -> None:
        is_dict = isinstance(msg, dict)
        content = msg.get("content") if is_dict else getattr(msg, "content", None)
        if not isinstance(content, list):
            return
        filtered = []
        dropped = 0
        for part in content:
            if isinstance(part, dict) and self._is_tool_output_part(part):
                part_ids = self._tool_result_ids_from_part(part)
                if not part_ids or not (part_ids & valid_ids):
                    dropped += 1
                    continue
            filtered.append(part)
        if dropped:
            logger.warning(
                "[LengthErrorHandler] 已移除 %s 个孤立 function_call_output 内容块",
                dropped,
            )
        if is_dict:
            msg["content"] = filtered
        else:
            try:
                setattr(msg, "content", filtered)
            except AttributeError:
                pass

    def _collect_assistant_tool_call_ids(self, messages: Any) -> set[str]:
        ids: set[str] = set()
        if not isinstance(messages, list):
            return ids
        for msg in messages:
            ids.update(self._assistant_tool_call_ids_from_message(msg))
        return ids

    def _assistant_tool_call_ids_from_message(self, msg: dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        tool_calls = None
        content = None
        if isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
            content = msg.get("content")
        else:
            tool_calls = getattr(msg, "tool_calls", None)
            content = getattr(msg, "content", None)
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                call_id = None
                if isinstance(tool_call, dict):
                    call_id = tool_call.get("id") or tool_call.get("call_id")
                else:
                    call_id = getattr(tool_call, "id", None) or getattr(tool_call, "call_id", None)
                if self._has_non_empty_id(call_id):
                    ids.add(str(call_id).strip())
        parts = content if isinstance(content, list) else []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if self._is_tool_call_part(part):
                for key in ("call_id", "id", "tool_call_id"):
                    value = part.get(key)
                    if self._has_non_empty_id(value):
                        ids.add(str(value).strip())
        return ids

    def _collect_tool_result_ids(self, messages: Any) -> set[str]:
        ids: set[str] = set()
        if not isinstance(messages, list):
            return ids
        for msg in messages:
            ids.update(self._tool_result_ids_from_message(msg))
        return ids

    def _tool_result_ids_from_message(self, msg: Any) -> set[str]:
        ids: set[str] = set()
        if isinstance(msg, dict):
            for key in ("tool_call_id", "call_id", "tool_use_id"):
                value = msg.get(key)
                if self._has_non_empty_id(value):
                    ids.add(str(value).strip())
            content = msg.get("content")
        else:
            for key in ("tool_call_id", "call_id", "tool_use_id"):
                value = getattr(msg, key, None)
                if self._has_non_empty_id(value):
                    ids.add(str(value).strip())
            content = getattr(msg, "content", None)
        parts = content if isinstance(content, list) else []
        for part in parts:
            if isinstance(part, dict) and self._is_tool_output_part(part):
                ids.update(self._tool_result_ids_from_part(part))
        return ids

    @staticmethod
    def _tool_result_ids_from_part(part: dict[str, Any]) -> set[str]:
        ids: set[str] = set()
        for key in ("call_id", "tool_call_id", "tool_use_id"):
            value = part.get(key)
            if LengthErrorHandlerPlugin._has_non_empty_id(value):
                ids.add(str(value).strip())
        return ids

    @staticmethod
    def _has_non_empty_id(value: Any) -> bool:
        return value is not None and bool(str(value).strip())

    def _is_tool_output_message(self, msg: Any) -> bool:
        if isinstance(msg, dict):
            if msg.get("role") == "tool":
                return True
            if msg.get("type") in {"function_call_output", "tool_result"}:
                return True
            content = msg.get("content")
        else:
            if getattr(msg, "role", None) == "tool":
                return True
            if getattr(msg, "type", None) in {"function_call_output", "tool_result"}:
                return True
            content = getattr(msg, "content", None)
        return isinstance(content, list) and any(
            isinstance(part, dict) and self._is_tool_output_part(part) for part in content
        )

    @staticmethod
    def _is_tool_output_part(part: dict[str, Any]) -> bool:
        return part.get("type") in {"function_call_output", "tool_result"}

    @staticmethod
    def _is_tool_call_part(part: dict[str, Any]) -> bool:
        return part.get("type") in {"function_call", "tool_use"}

    @staticmethod
    def _estimate_messages_tokens(messages: list[Any]) -> int:
        try:
            return len(json.dumps(messages, ensure_ascii=False, default=str)) // 4
        except Exception:
            return len(str(messages)) // 4

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return " ".join(content.split())
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
                    elif item.get("type"):
                        parts.append(f"[{item.get('type')}]")
                else:
                    parts.append(str(item))
            return " ".join(" ".join(parts).split())
        return " ".join(str(content).split())

    def _ensure_runner_budget(self, runner: Any, retry: bool):
        try:
            provider = getattr(runner, "provider", None)
            config = getattr(provider, "curr_model_config", None)
            if config is None:
                return
            target = int(self.cfg.get("retry_completion_tokens" if retry else "min_completion_tokens", 4096 if retry else 2048))
            for attr in ("max_completion_tokens", "max_tokens"):
                current = getattr(config, attr, None)
                if isinstance(current, int) and current < target:
                    setattr(config, attr, target)
                    logger.info(f"[LengthErrorHandler] model_config.{attr} 调整: {current} -> {target}")
        except Exception as exc:
            logger.warning(f"[LengthErrorHandler] 调整 runner 输出预算失败: {exc}")

    def _log_error(self, error: Exception, obj: Any, stage: str):
        if not self.cfg.get("enable_learning", True):
            return
        usage = getattr(getattr(error, "completion", None), "usage", None)
        record = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error)[:2000],
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "model": self._safe_model_name(obj),
        }
        try:
            with open(self.error_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"[LengthErrorHandler] 写入日志失败: {exc}")

    @staticmethod
    def _safe_model_name(obj: Any) -> str:
        try:
            provider = getattr(obj, "provider", None) or obj
            provider_config = getattr(provider, "provider_config", None)
            if isinstance(provider_config, dict) and provider_config.get("model"):
                return str(provider_config["model"])
            for attr in ("model_name", "model", "id"):
                value = getattr(provider, attr, None)
                if value:
                    return str(value)
        except Exception:
            pass
        return "unknown"

    async def terminate(self):
        if getattr(ToolLoopAgentRunner, "_length_error_handler_active_plugin", None) is self:
            ToolLoopAgentRunner._length_error_handler_active_plugin = None
        if ProviderOpenAIOfficial is not None and getattr(ProviderOpenAIOfficial, "_length_error_handler_active_plugin", None) is self:
            ProviderOpenAIOfficial._length_error_handler_active_plugin = None
        logger.info("[LengthErrorHandler] 已停用活动实例")

    @filter.command("length_error_stats")
    async def show_stats(self, event: AstrMessageEvent):
        if not os.path.exists(self.error_log_path):
            yield event.plain_result("暂无 length error 记录")
            return
        try:
            with open(self.error_log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            yield event.plain_result(f"已记录 length error 次数: {len(lines)}")
        except Exception as exc:
            yield event.plain_result(f"读取日志失败: {exc}")

    @filter.command("length_error_clear")
    async def clear_logs(self, event: AstrMessageEvent):
        try:
            if os.path.exists(self.error_log_path):
                os.remove(self.error_log_path)
            yield event.plain_result("Length error 日志已清空")
        except Exception as exc:
            yield event.plain_result(f"清空失败: {exc}")

    @filter.command("length_compress_test")
    async def test_compression(self, event: AstrMessageEvent):
        c = self.cfg
        yield event.plain_result(
            f"智能压缩 v1.0.6\n"
            f"上下文压缩: {c.get('enable_context_compression')}\n"
            f"Runner 阈值: {c.get('compression_threshold')} tokens\n"
            f"Provider 阈值: {c.get('provider_compression_threshold')} tokens\n"
            f"保留最近: {c.get('keep_recent_rounds')} 条非 system 消息\n"
            f"最小输出预算: {c.get('min_completion_tokens')}\n"
            f"重试输出预算: {c.get('retry_completion_tokens')}\n"
            f"Provider patch: {c.get('enable_provider_patch')}\n"
            f"Reasoning 降档: {c.get('force_reasoning_effort_low')}"
        )
