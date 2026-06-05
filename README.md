# 长度错误处理器

自动处理 AstrBot 在调用 OpenAI 兼容 Provider 时可能出现的 `LengthFinishReasonError` / `finish_reason=length` 问题。

本插件会在上下文过长或输出预算不足时，尝试进行上下文压缩、输出 token 预算修正、reasoning 降档，并在捕获 length 截断错误后自动重试一次，降低长对话、工具调用链较长时的失败概率。

## 功能特性

- 自动识别 `LengthFinishReasonError`、`finish_reason=length`、`max_tokens` / `max_completion_tokens` 相关截断错误。
- Runner 层上下文压缩：在对话历史过长时，将较早消息折叠成摘要，只保留最近消息。
- OpenAI Provider 层兜底 patch：在请求发送前检查并修正 payload。
- 自动抬高输出预算：当 `max_tokens` 或 `max_completion_tokens` 过低时，自动提高到配置值。
- length 错误自动重试：捕获截断错误后，强制压缩 messages 并提高重试输出预算。
- reasoning 降档：可将 `reasoning_effort` 调整为 `low`，减少推理 token 挤占正文输出空间。
- 工具调用配对修复：压缩上下文时尽量保留必要的 assistant tool_call 与 tool 输出配对，避免 OpenAI 400 错误。
- 错误日志记录：记录 length error 出现次数、阶段、模型和 token 使用信息，方便排障。
- 支持 WebUI 配置。

## 适用场景

适合以下情况：

- 长对话后模型突然报错，错误中包含 `finish_reason=length`。
- 工具调用很多，历史消息变长后 OpenAI Provider 返回 400 或解析失败。
- 模型输出预算太小，导致 response 被截断。
- reasoning 模型把大量 token 用在推理阶段，正文未完整返回。
- 多轮工具调用压缩后出现 tool_call / tool_result 配对问题。

## 安装方式

将插件目录放入 AstrBot 插件目录：

```text
AstrBot/data/plugins/astrbot_plugin_length_error_handler
```

目录结构示例：

```text
astrbot_plugin_length_error_handler/
├── main.py
├── metadata.yaml
├── _conf_schema.json
└── README.md
```

然后在 AstrBot WebUI 中重载插件，或重启 AstrBot。

## 配置说明

插件支持通过 AstrBot WebUI 配置。主要配置项如下：

| 配置项 | 默认值 | 说明 |
| --- | ---: | --- |
| `enable_context_compression` | `true` | 是否启用上下文压缩。 |
| `compression_threshold` | `35000` | Runner 层触发压缩的粗估 token 阈值。 |
| `provider_compression_threshold` | `50000` | Provider 层发送前触发压缩的粗估 token 阈值。 |
| `keep_recent_rounds` | `4` | 压缩时保留最近的非 system 消息数。 |
| `summary_max_tokens` | `512` | 摘要最大 token 数，当前作为配置保留。 |
| `max_older_messages_for_summary` | `20` | 最多取多少条早期消息参与摘要文本构造。 |
| `enable_learning` | `true` | 是否记录 length error 日志。 |
| `enable_provider_patch` | `true` | 是否启用 OpenAI Provider 层兜底 patch。 |
| `enable_retry_on_length_error` | `true` | 捕获 length 错误后是否自动重试一次。 |
| `min_completion_tokens` | `2048` | 正常请求的最小输出 token 预算。 |
| `retry_completion_tokens` | `4096` | length 错误重试时使用的输出 token 预算。 |
| `force_reasoning_effort_low` | `true` | 是否将 `reasoning_effort` 降为 `low`。 |
| `sanitize_tool_call_pairs` | `true` | 是否修复压缩后的工具调用配对关系。 |

如果没有从 WebUI 读取到配置，插件会尝试在以下位置生成本地配置文件：

```text
data/plugins_data/length_error_handler/config.json
```

## 指令

### 查看 length error 统计

```text
/length_error_stats
```

返回已记录的 length error 次数。

### 清空 length error 日志

```text
/length_error_clear
```

清空插件记录的 length error 日志。

### 查看压缩配置测试信息

```text
/length_compress_test
```

返回当前上下文压缩、阈值、输出预算、Provider patch、reasoning 降档等配置状态。

## 日志位置

当 `enable_learning=true` 时，插件会记录 length error 日志：

```text
data/plugins_data/length_error_handler/length_error.jsonl
```

每条日志包含：

- 时间戳
- 捕获阶段：Runner / Provider Query / Provider Stream
- 错误类型
- 错误摘要
- prompt tokens
- completion tokens
- total tokens
- 模型名称

## 工作机制

插件启动后会注入两层处理逻辑。

### 1. Runner 层

在 `ToolLoopAgentRunner._iter_llm_responses_with_fallback` 前后进行处理：

1. 调用前检查上下文长度。
2. 必要时压缩早期消息。
3. 修正模型输出预算。
4. 如果捕获 length 错误，则记录日志、强制压缩、提高输出预算并重试。

### 2. OpenAI Provider 层

在 `ProviderOpenAIOfficial._query` 和 `_query_stream` 处进行兜底：

1. 深拷贝 payload，避免直接污染原始请求。
2. 修正 `max_tokens` / `max_completion_tokens`。
3. 将 `reasoning_effort` 调低。
4. 根据阈值压缩 `messages`。
5. 修复 tool_call 与 tool 输出配对。
6. 捕获 length 错误后再强制压缩并重试一次。

## 压缩策略

上下文压缩时，插件会：

- 保留所有 system 消息。
- 保留最近 `keep_recent_rounds` 条非 system 消息。
- 将更早的消息转换为一条 system 摘要。
- 摘要最多参考最近 `max_older_messages_for_summary` 条旧消息。
- 尽量补回被保留 tool 输出所依赖的 assistant tool_call 消息。
- 移除孤立的 tool/function 输出，避免 OpenAI 报错。

压缩摘要会带有类似标记：

```text
[LengthErrorHandler 自动压缩] 已折叠 N 条较早消息，仅保留摘要和最近 M 条必要消息。
```

## 调参建议

### 长对话仍然容易失败

可以适当降低：

```text
provider_compression_threshold: 30000~40000
compression_threshold: 25000~35000
```

### 输出仍然被截断

可以适当提高：

```text
min_completion_tokens: 4096
retry_completion_tokens: 8192
```

前提是你的模型和 Provider 支持对应输出上限。

### 工具调用链容易报 400

保持开启：

```text
sanitize_tool_call_pairs: true
```

不要随意关闭该项。

### reasoning 模型正文输出太少

保持开启：

```text
force_reasoning_effort_low: true
```

这可以减少 reasoning token 占用过多输出预算的情况。

## 注意事项

- 本插件只能降低 length 类错误概率，不能突破模型本身的最大上下文窗口或最大输出限制。
- 如果 Provider 或模型不支持较高的 `max_tokens`，请不要把 `retry_completion_tokens` 设置得过大。
- 上下文压缩会丢失部分早期细节，因此关键需求建议在当前对话中重新明确。
- 插件通过 monkey patch 注入 AstrBot 内部 Runner 和 OpenAI Provider，AstrBot 内部 API 大版本变化时可能需要适配。
- `metadata.yaml` 中的 `name` 是插件加载标识，不建议改成中文。

## 兼容性

- 插件标识：`astrbot_plugin_length_error_handler`
- 显示名称：`长度错误处理器`
- 当前版本：`1.0.6`
- 建议 AstrBot 版本：`>=4.0.0`
- Provider：主要针对 OpenAI 官方兼容 Provider。

## 常见问题

### 1. 为什么已经压缩了还是报错？

可能原因：

- 模型上下文窗口本身太小。
- 用户输入、工具结果或系统提示一次性过长。
- Provider 实际支持的输出上限低于配置值。
- 工具调用返回内容太大，单次响应已经超过限制。

可以尝试降低压缩阈值，或减少工具返回内容。

### 2. 为什么历史上下文里出现自动压缩提示？

这是插件为了保留早期对话关键信息写入的摘要消息，不是异常。

### 3. 为什么会移除部分 tool/function 输出？

OpenAI 要求 tool 输出必须能找到对应的 assistant tool_call。上下文压缩后如果配对不完整，保留孤立 tool 输出会导致 400 错误。因此插件会清理孤立输出。

### 4. 为什么不要改插件 name？

`name` 是 AstrBot 识别和加载插件的内部标识。中文显示名请使用 `display_name`，不要修改 `name`。

## 许可证

MIT
