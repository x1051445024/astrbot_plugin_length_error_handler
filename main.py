import copy
import json
import os
import re
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
    "hard_token_budget": 32000,
    "max_system_chars": 12000,
    "max_message_chars": 6000,
    "max_tool_chars": 3000,
    "replace_inline_images_over_chars": 12000,
    "keep_recent_rounds": 4,
    "summary_max_tokens": 512,
    "max_older_messages_for_summary": 20,
    "enable_learning": True,
    "enable_provider_patch": True,
    "enable_retry_on_length_error": True,
    "min_completion_tokens": 2048,
    "retry_completion_tokens": 4096,
    "force_reasoning_effort_low": False,
    "sanitize_tool_call_pairs": True,
    "max_retry_count": 1,
    "summary_prompt_prefix": "请用简洁的中文（200字以内）总结以下多轮对话的关键信息、决策和上下文，不要添加主观评价，只保留事实：\n\n",
}


@register(
    "astrbot_plugin_length_error_handler",
    "牧濑红莉栖（BOT）",
    "自动处理 LengthFinishReasonError + 输出预算修正 + Provider 兜底重试（1.0.9）",
    "1.0.9",
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
        self._retry_counters: dict[int, int] = {}
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
        marker = "1.0.9"
        if getattr(ToolLoopAgentRunner, "_length_error_handler_runner_patch_version", None) == marker:
            logger.info("[LengthErrorHandler] Runner v1.0.9 已注入，刷新活动实例")
            return

        original_iter = getattr(
            ToolLoopAgentRunner,
            "_length_error_handler_original_iter_v106",
            getattr(
                ToolLoopAgentRunner,
                "_length_error_handler_original_iter_v107",
                ToolLoopAgentRunner._iter_llm_responses_with_fallback,
            ),
        )
        ToolLoopAgentRunner._length_error_handler_original_iter_v107 = original_iter

        async def patched_iter(runner_self, *args, **kwargs):
            plugin = LengthErrorHandlerPlugin._active_from_runner()
            # 优化B: 正常路径的预处理包在 try 里，失败则降级为直接调用原函数
            if plugin is not None:
                try:
                    await plugin._maybe_compress_runner_context(runner_self, force=False)
                    plugin._ensure_runner_budget(runner_self, retry=False)
                except Exception as exc_prep:
                    logger.warning(f"[LengthErrorHandler] runner 预处理失败，降级直跑原函数: {exc_prep}")
            try:
                async for resp in original_iter(runner_self, *args, **kwargs):
                    yield resp
            except Exception as exc:
                plugin = LengthErrorHandlerPlugin._active_from_runner()
                if plugin is None or not plugin._is_length_error(exc):
                    raise
                if plugin._retry_budget_exceeded(exc):
                    logger.warning("[LengthErrorHandler] Runner length 重试已达上限(max_retry_count)，交给核心 fallback")
                    raise
                logger.warning("[LengthErrorHandler] Runner 捕获 length 截断，压缩并扩大输出预算后重试")
                # 优化B: 重试路径也包 try，失败则抛原异常，不吞错
                try:
                    plugin._log_error(exc, runner_self, "runner")
                    plugin._ensure_runner_budget(runner_self, retry=True)
                    await plugin._maybe_compress_runner_context(runner_self, force=True)
                except Exception as exc_retry_prep:
                    logger.warning(f"[LengthErrorHandler] runner 重试预处理失败: {exc_retry_prep}")
                    raise
                async for resp in original_iter(runner_self, *args, **kwargs):
                    yield resp
                plugin._clear_retry_counter(exc)

    def _patch_openai_provider(self):
        if not self.cfg.get("enable_provider_patch", True):
            logger.info("[LengthErrorHandler] Provider patch 已关闭")
            return
        if ProviderOpenAIOfficial is None:
            logger.warning("[LengthErrorHandler] 未找到 ProviderOpenAIOfficial，跳过 Provider patch")
            return

        marker = "1.0.9"
        if getattr(ProviderOpenAIOfficial, "_length_error_handler_provider_patch_version", None) == marker:
            logger.info("[LengthErrorHandler] OpenAI Provider v1.0.9 已注入，刷新活动实例")
            return

        original_query = getattr(
            ProviderOpenAIOfficial,
            "_length_error_handler_original_query_v106",
            getattr(
                ProviderOpenAIOfficial,
                "_length_error_handler_original_query_v107",
                ProviderOpenAIOfficial._query,
            ),
        )
        original_query_stream = getattr(
            ProviderOpenAIOfficial,
            "_length_error_handler_original_query_stream_v106",
            getattr(
                ProviderOpenAIOfficial,
                "_length_error_handler_original_query_stream_v107",
                ProviderOpenAIOfficial._query_stream,
            ),
        )
        ProviderOpenAIOfficial._length_error_handler_original_query_v107 = original_query
        ProviderOpenAIOfficial._length_error_handler_original_query_stream_v107 = original_query_stream

        async def patched_query(provider_self, payloads: dict, tools):
            plugin = LengthErrorHandlerPlugin._active_from_provider()
            if plugin is None:
                return await original_query(provider_self, payloads, tools)
            # 优化B: _prepare_payload 失败则降级为直接用原始 payloads
            try:
                first_payload = plugin._prepare_payload(payloads, retry=False, force_compress=False)
            except Exception as exc_prep:
                logger.warning(f"[LengthErrorHandler] provider_query 预处理失败，用原始 payloads: {exc_prep}")
                first_payload = payloads
            try:
                return await original_query(provider_self, first_payload, tools)
            except Exception as exc:
                if not plugin._should_retry(exc):
                    raise
                if plugin._retry_budget_exceeded(exc):
                    logger.warning("[LengthErrorHandler] provider_query length 重试已达上限(max_retry_count)，交给核心 fallback")
                    raise
                logger.warning("[LengthErrorHandler] OpenAI Provider 捕获 length 截断，压缩 messages + 扩大输出预算后重试")
                plugin._log_error(exc, provider_self, "provider_query")
                # 优化B: 重试路径也包 try，失败则抛原异常，不吞错
                try:
                    retry_payload = plugin._prepare_payload(payloads, retry=True, force_compress=True)
                except Exception as exc_retry_prep:
                    logger.warning(f"[LengthErrorHandler] provider_query 重试预处理失败: {exc_retry_prep}")
                    raise
                result = await original_query(provider_self, retry_payload, tools)
                plugin._clear_retry_counter(exc)
                return result

        async def patched_query_stream(provider_self, payloads: dict, tools):
            plugin = LengthErrorHandlerPlugin._active_from_provider()
            if plugin is None:
                async for resp in original_query_stream(provider_self, payloads, tools):
                    yield resp
                return
            # 优化B: _prepare_payload 失败则降级为直接用原始 payloads
            try:
                first_payload = plugin._prepare_payload(payloads, retry=False, force_compress=False)
            except Exception as exc_prep:
                logger.warning(f"[LengthErrorHandler] provider_stream 预处理失败，用原始 payloads: {exc_prep}")
                first_payload = payloads
            try:
                async for resp in original_query_stream(provider_self, first_payload, tools):
                    yield resp
            except Exception as exc:
                if not plugin._should_retry(exc):
                    raise
                if plugin._retry_budget_exceeded(exc):
                    logger.warning("[LengthErrorHandler] provider_stream length 重试已达上限(max_retry_count)，交给核心 fallback")
                    raise
                logger.warning("[LengthErrorHandler] OpenAI Provider Stream 捕获 length 截断，压缩 messages + 扩大输出预算后重试")
                plugin._log_error(exc, provider_self, "provider_stream")
                # 优化B: 重试路径也包 try，失败则抛原异常，不吞错
                try:
                    retry_payload = plugin._prepare_payload(payloads, retry=True, force_compress=True)
                except Exception as exc_retry_prep:
                    logger.warning(f"[LengthErrorHandler] provider_stream 重试预处理失败: {exc_retry_prep}")
                    raise
                async for resp in original_query_stream(provider_self, retry_payload, tools):
                    yield resp
                plugin._clear_retry_counter(exc)

    def _is_length_error(self, exc: Exception) -> bool:
        # 优化: 优先用 SDK 的强类型判断，最可靠。
        if LengthFinishReasonError is not None and isinstance(exc, LengthFinishReasonError):
            return True
        text = f"{type(exc).__name__}: {exc}".lower()
        # 优化: 必须命中明确的 finish_reason=length 信号，避免把中转 API 的
        # "max_tokens parameter is invalid" 之类参数校验错误误判成 length 截断。
        strong_signals = (
            "context_length_exceeded",
            "context window",
            "input exceeds the context",
            "maximum context length",
            "lengthfinishreasonerror",
            "length limit was reached",
            "finish_reason='length'",
            'finish_reason="length"',
            "finish reason length",
            "too many tokens",
            "token limit",
        )
        if not any(pat in text for pat in strong_signals):
            return False
        # 优化: 二次确认必须出现 length/token/context 之一，进一步收紧
        return ("length" in text or "token" in text or "context" in text)

    def _retry_budget_exceeded(self, exc: Exception) -> bool:
        # 优化C: 按异常对象 id 追踪该请求已重试次数，防止与核心 fallback 叠乘。
        # max_retry_count 默认 1：同一 length 错误最多重试一次。
        key = id(exc)
        count = self._retry_counters.get(key, 0)
        limit = int(self.cfg.get("max_retry_count", 1))
        if count >= limit:
            self._retry_counters.pop(key, None)
            return True
        self._retry_counters[key] = count + 1
        return False

    def _clear_retry_counter(self, exc: Exception) -> None:
        self._retry_counters.pop(id(exc), None)

    def _should_retry(self, exc: Exception) -> bool:
        return bool(self.cfg.get("enable_retry_on_length_error", True)) and self._is_length_error(exc)

    def _prepare_payload(self, payloads: dict, retry: bool, force_compress: bool) -> dict:
        # 优化A: 快速短路。正常请求(retry=False, force_compress=False)且上下文较小时，
        # 跳过 deepcopy + json.dumps 估算 + 压缩，只做轻量的预算/推理调整。
        # 这避免每次正常请求都付出 deepcopy(含base64图片) + 两次 json.dumps 全量估算的代价。
        if not retry and not force_compress:
            messages = payloads.get("messages")
            if isinstance(messages, list):
                quick_chars = self._quick_estimate_chars(messages)
                threshold_chars = int(self.cfg.get("provider_compression_threshold", 50000)) * 4
                if quick_chars < int(threshold_chars * 0.6):
                    prepared = dict(payloads)
                    prepared["messages"] = list(messages)
                    self._ensure_payload_budget(prepared, retry=retry)
                    self._normalize_reasoning(prepared)
                    return prepared
        # 完整路径: 上下文较大或需要重试/强制压缩时，走完整的 deepcopy + 估算 + 压缩
        prepared = copy.deepcopy(payloads)
        self._ensure_payload_budget(prepared, retry=retry)
        self._normalize_reasoning(prepared)
        self._sanitize_payload_messages(prepared)
        if self.cfg.get("enable_context_compression", True):
            self._maybe_compress_payload_messages(prepared, force=force_compress)
        self._enforce_payload_hard_budget(prepared)
        self._sanitize_payload_messages(prepared)
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
        if not self.cfg.get("force_reasoning_effort_low", False):
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
        self._enforce_payload_hard_budget(payloads)

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
        messages = self._trim_messages_to_char_limits(messages)
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

    def _enforce_payload_hard_budget(self, payloads: dict):
        messages = payloads.get("messages")
        if not isinstance(messages, list):
            return
        before = self._estimate_messages_tokens(messages)
        budget = int(self.cfg.get("hard_token_budget", 32000))
        if before <= budget:
            self._sanitize_payload_messages(payloads)
            return
        trimmed = self._trim_messages_to_char_limits(messages)
        after = self._estimate_messages_tokens(trimmed)
        if after > budget:
            keep_recent = max(1, int(self.cfg.get("keep_recent_rounds", 4)) // 2)
            trimmed = self._compress_messages(trimmed, keep_recent)
            after = self._estimate_messages_tokens(trimmed)
        if after > budget:
            trimmed = self._shrink_messages_to_budget(trimmed, budget)
            after = self._estimate_messages_tokens(trimmed)
        payloads["messages"] = self._repair_tool_call_pairs(trimmed)
        logger.warning(
            f"[LengthErrorHandler] Provider messages 硬裁剪: tokens≈{before} -> {after}，消息数={len(payloads['messages'])}"
        )

    def _sanitize_payload_messages(self, payloads: dict) -> None:
        messages = payloads.get("messages")
        if not isinstance(messages, list):
            return
        payloads["messages"] = self._repair_tool_call_pairs(messages)

    def _trim_messages_to_char_limits(self, messages: list[Any]) -> list[Any]:
        max_system = int(self.cfg.get("max_system_chars", 12000))
        max_message = int(self.cfg.get("max_message_chars", 6000))
        max_tool = int(self.cfg.get("max_tool_chars", 3000))
        return self._trim_messages_with_limits(messages, max_system, max_message, max_tool)

    def _shrink_messages_to_budget(self, messages: list[Any], budget: int) -> list[Any]:
        max_system = int(self.cfg.get("max_system_chars", 12000))
        max_message = int(self.cfg.get("max_message_chars", 6000))
        max_tool = int(self.cfg.get("max_tool_chars", 3000))
        trimmed = messages
        for factor in (0.7, 0.5, 0.35, 0.25, 0.15):
            trimmed = self._trim_messages_with_limits(
                trimmed,
                max(1200, int(max_system * factor)),
                max(800, int(max_message * factor)),
                max(500, int(max_tool * factor)),
            )
            if self._estimate_messages_tokens(trimmed) <= budget:
                break
        return self._repair_tool_call_pairs(trimmed)

    def _trim_messages_with_limits(
        self,
        messages: list[Any],
        max_system: int,
        max_message: int,
        max_tool: int,
    ) -> list[Any]:
        trimmed: list[Any] = []
        protected_image_indexes = self._protected_inline_image_indexes(messages)
        for index, msg in enumerate(messages):
            item = copy.deepcopy(msg)
            role = self._message_role(item)
            limit = max_tool if role in {"tool", "function"} else max_message
            if role == "system":
                limit = max_system
            self._trim_message_content(
                item,
                limit,
                preserve_inline_images=index in protected_image_indexes,
            )
            trimmed.append(item)
        return trimmed

    @staticmethod
    def _message_role(msg: Any) -> str | None:
        if isinstance(msg, dict):
            return msg.get("role")
        return getattr(msg, "role", None)

    def _trim_message_content(
        self,
        msg: Any,
        limit: int,
        preserve_inline_images: bool = False,
    ) -> None:
        if limit <= 0:
            return
        if isinstance(msg, dict):
            if "content" in msg:
                msg["content"] = self._trim_content_value(
                    msg.get("content"),
                    limit,
                    preserve_inline_images=preserve_inline_images,
                )
            if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                for tool_call in msg["tool_calls"]:
                    if isinstance(tool_call, dict):
                        func = tool_call.get("function")
                        if isinstance(func, dict) and isinstance(func.get("arguments"), str):
                            func["arguments"] = self._truncate_text(func["arguments"], limit)
            return
        content = getattr(msg, "content", None)
        try:
            setattr(
                msg,
                "content",
                self._trim_content_value(
                    content,
                    limit,
                    preserve_inline_images=preserve_inline_images,
                ),
            )
        except Exception:
            pass

    def _trim_content_value(
        self,
        content: Any,
        limit: int,
        preserve_inline_images: bool = False,
    ) -> Any:
        if isinstance(content, str):
            return self._truncate_text(content, limit)
        if hasattr(content, "model_dump_for_context"):
            try:
                return self._trim_content_part(
                    content.model_dump_for_context(),
                    limit,
                    preserve_inline_images=preserve_inline_images,
                )
            except Exception:
                return self._truncate_text(str(content), limit)
        if isinstance(content, list):
            trimmed_parts = []
            for part in content:
                if isinstance(part, dict):
                    trimmed_parts.append(
                        self._trim_content_part(
                            part,
                            limit,
                            preserve_inline_images=preserve_inline_images,
                        )
                    )
                elif hasattr(part, "model_dump_for_context"):
                    try:
                        trimmed_parts.append(
                            self._trim_content_part(
                                part.model_dump_for_context(),
                                limit,
                                preserve_inline_images=preserve_inline_images,
                            )
                        )
                    except Exception:
                        trimmed_parts.append(self._truncate_text(str(part), limit))
                else:
                    trimmed_parts.append(self._truncate_text(str(part), limit))
            return trimmed_parts
        return content

    def _protected_inline_image_indexes(self, messages: list[Any]) -> set[int]:
        protected: set[int] = set()
        for index in range(len(messages) - 1, -1, -1):
            msg = messages[index]
            if self._message_role(msg) != "user":
                continue
            if self._message_has_inline_image(msg):
                protected.add(index)
                break
        return protected

    def _message_has_inline_image(self, msg: Any) -> bool:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        return self._content_has_inline_image(content)

    def _content_has_inline_image(self, content: Any) -> bool:
        if hasattr(content, "model_dump_for_context"):
            try:
                content = content.model_dump_for_context()
            except Exception:
                return False
        if isinstance(content, dict):
            if self._inline_image_data_url(content):
                return True
            return any(self._content_has_inline_image(value) for value in content.values())
        if isinstance(content, list):
            return any(self._content_has_inline_image(item) for item in content)
        return isinstance(content, str) and content.startswith("data:image/") and ";base64," in content[:128]

    def _trim_content_part(
        self,
        part: dict[str, Any],
        limit: int,
        preserve_inline_images: bool = False,
    ) -> dict[str, Any]:
        part_copy = copy.deepcopy(part)
        inline_image = self._inline_image_data_url(part_copy)
        image_limit = int(self.cfg.get("replace_inline_images_over_chars", 12000))
        if inline_image and preserve_inline_images:
            return part_copy
        if inline_image and len(inline_image) > image_limit:
            media_type = inline_image.split(";", 1)[0].removeprefix("data:") or "image"
            return {
                "type": "text",
                "text": (
                    f"[LengthErrorHandler 已省略内联图片 data URL（{media_type}，约 {len(inline_image)} 字符）；"
                    "请根据同一消息中的图片路径或调用读图工具读取原图]"
                ),
            }
        for key in ("text", "output", "content"):
            if isinstance(part_copy.get(key), str):
                part_copy[key] = self._truncate_text(part_copy[key], limit)
        return part_copy

    @staticmethod
    def _inline_image_data_url(part: dict[str, Any]) -> str | None:
        def _is_data_image(value: Any) -> bool:
            return isinstance(value, str) and value.startswith("data:image/") and ";base64," in value[:128]

        image_url = part.get("image_url")
        if isinstance(image_url, dict) and _is_data_image(image_url.get("url")):
            return image_url["url"]
        if _is_data_image(image_url):
            return image_url
        if _is_data_image(part.get("url")):
            return part["url"]

        source = part.get("source")
        if isinstance(source, dict):
            data = source.get("data")
            media_type = str(source.get("media_type") or source.get("mime_type") or "")
            if isinstance(data, str) and len(data) > 12000 and media_type.startswith("image/"):
                return f"data:{media_type};base64,{data}"
        return None

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        keep_head = max(0, limit // 2)
        keep_tail = max(0, limit - keep_head - 80)
        omitted = len(text) - keep_head - keep_tail
        return (
            text[:keep_head]
            + f"\n...[LengthErrorHandler 已截断 {omitted} 字符]...\n"
            + (text[-keep_tail:] if keep_tail else "")
        )

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
        else:
            before_drop = len(repaired)
            repaired = self._drop_all_tool_outputs(repaired)
            dropped_outputs += before_drop - len(repaired)

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

            for call_id in sorted(missing_ids):
                stub_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {
                            "stub": True,
                            "note": "工具结果已在早前轮次被上下文压缩移除，此条为自动补填的占位结果，请忽略具体内容并基于上下文继续。",
                        },
                        ensure_ascii=False,
                    ),
                }
                messages.insert(index + 1, stub_msg)
                filled_count += 1

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

    def _quick_estimate_chars(self, messages: list[Any]) -> int:
        # 优化A: 轻量估算消息总字符数，不做 json.dumps，不 deepcopy。
        # 只用于快速判断是否需要进入完整压缩流程，作为下界估算(实际 tokens 约 = chars/4)。
        total = 0
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                tc = msg.get("tool_calls")
            else:
                content = getattr(msg, "content", None)
                tc = getattr(msg, "tool_calls", None)
            total += self._content_char_len(content)
            if isinstance(tc, list):
                for call in tc:
                    if isinstance(call, dict):
                        fn = call.get("function")
                        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                            total += len(fn["arguments"])
        return total

    def _content_char_len(self, content: Any) -> int:
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            n = 0
            for part in content:
                if isinstance(part, dict):
                    if "text" in part and isinstance(part.get("text"), str):
                        n += len(part["text"])
                    elif part.get("type") == "image_url":
                        n += 200
                    else:
                        n += 50
                else:
                    n += len(str(part))
            return n
        return len(str(content))

    def _estimate_messages_tokens(self, messages: list[Any]) -> int:
        protected_image_indexes = self._protected_inline_image_indexes(messages)
        try:
            total_chars = 0
            for index, msg in enumerate(messages):
                text = json.dumps(msg, ensure_ascii=False, default=str)
                if index in protected_image_indexes:
                    text = self._compact_inline_image_data_urls(text)
                total_chars += len(text)
            return total_chars // 4
        except Exception:
            return len(str(messages)) // 4

    @staticmethod
    def _compact_inline_image_data_urls(text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            header = match.group(1)
            data = match.group(2)
            return f"{header}[inline image omitted for token estimate; chars={len(data)}]"

        return re.sub(
            r"(data:image/[^;\"'\s]+;base64,)([A-Za-z0-9+/=\r\n]+)",
            _replace,
            text,
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return " ".join(content.split())
        if hasattr(content, "model_dump_for_context"):
            try:
                content = content.model_dump_for_context()
            except Exception:
                return " ".join(str(content).split())
        if isinstance(content, list):
            parts = []
            for item in content:
                if hasattr(item, "model_dump_for_context"):
                    try:
                        item = item.model_dump_for_context()
                    except Exception:
                        parts.append(str(item))
                        continue
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url")
                        image_id = image_url.get("id") if isinstance(image_url, dict) else None
                        parts.append(f"[image_url:{image_id or 'image'}]")
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
                    elif item.get("type"):
                        parts.append(f"[{item.get('type')}]")
                else:
                    parts.append(str(item))
            return " ".join(" ".join(parts).split())
        return " ".join(str(content).split())

    def _ensure_runner_budget(self, runner: Any, retry: bool):
        # 优化: 只在 retry=True 时才动 runner 的 model_config，正常请求绝不修改全局配置，
        # 避免把 min_completion_tokens 的副作用残留到后续请求。
        if not retry:
            return
        try:
            provider = getattr(runner, "provider", None)
            config = getattr(provider, "curr_model_config", None)
            if config is None:
                return
            target = int(self.cfg.get("retry_completion_tokens", 4096))
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
            f"智能压缩 v1.0.9\n"
            f"上下文压缩: {c.get('enable_context_compression')}\n"
            f"Runner 阈值: {c.get('compression_threshold')} tokens\n"
            f"Provider 阈值: {c.get('provider_compression_threshold')} tokens\n"
            f"保留最近: {c.get('keep_recent_rounds')} 条非 system 消息\n"
            f"最小输出预算: {c.get('min_completion_tokens')}\n"
            f"重试输出预算: {c.get('retry_completion_tokens')}\n"
            f"Provider patch: {c.get('enable_provider_patch')}\n"
            f"Reasoning 降档: {c.get('force_reasoning_effort_low')}"
        )
