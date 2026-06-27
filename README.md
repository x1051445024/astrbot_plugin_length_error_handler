# 长度错误处理器

自动处理 AstrBot 调用 OpenAI 兼容 Provider 时常见的 `LengthFinishReasonError`、`finish_reason=length`、输出 token 预算不足和长上下文导致的失败。

本插件会在请求发送前或 length 截断错误发生后，尝试压缩上下文、调整输出预算、降低 reasoning 强度，并修复工具调用消息配对，降低长对话和多轮工具调用场景下的失败概率。

## 功能特性

- 自动识别 `LengthFinishReasonError`、`finish_reason=length`、`max_tokens` / `max_completion_tokens` 相关截断错误。
- 自动识别 OpenAI 兼容上游常见的 `context_length_exceeded`、`context window`、`too many tokens` 等上下文超限错误。
- Runner 层上下文压缩：当对话历史过长时，将较早消息折叠为摘要，只保留最近消息。
- OpenAI Provider 层兜底 patch：请求发送前检查并修正 payload。
- Provider 发送前硬预算裁剪：压缩后仍超限时继续收缩单条消息、工具结果和系统提示。
- 内联图片 data URL 保护：历史里保存的巨型 `data:image/...;base64` 会替换成文本占位，避免单张图片把上下文撑爆。
- 自动提高输出预算：当 `max_tokens` 或 `max_completion_tokens` 太低时，按配置提升到更安全的值。
- length 错误自动重试：捕获截断错误后强制压缩 messages，并使用更高输出预算重试。支持通过 `max_retry_count` 配置最大重试次数，避免与核心 fallback 叠乘。
- reasoning 降档：可将 `reasoning_effort` 调整为 `low`，减少推理 token 挤占正文输出空间。
- 工具调用配对修复：压缩上下文时尽量保留必要的 assistant tool_call 与 tool 输出配对，清理孤立输出。
- 空 ID 防护：过滤没有有效 `call_id` / `tool_call_id` 的工具输出，避免上游返回 `empty string` 400。
- 错误日志记录：记录 length error 出现次数、阶段、模型和 token 使用信息，方便排障。
- 支持 AstrBot WebUI 配置。
- **性能优化**：小上下文请求自动走快速路径，跳过 deepcopy 和 json.dumps 估算，降低 CPU 开销。
- **防御性容错**：插件预处理逻辑出错时自动降级为原始请求，不中断用户对话。

## 适用场景

适合以下情况：

- 长对话后模型突然报错，错误中包含 `finish_reason=length`。
- 工具调用很多，历史消息变长后 OpenAI Provider 返回 400 或解析失败。
- 模型输出预算太小，导致 response 被截断。
- reasoning 模型把大量 token 用在推理阶段，正文未完整返回。
- 多轮工具调用压缩后出现 tool_call / tool_result 配对问题。
- 上游提示 `context_length_exceeded`，但 AstrBot 日志只看到 `relay stream ended without a terminal result` 或 502。
- 会话历史里保存了巨型内联图片 base64，导致一次请求达到几十万甚至百万级粗估 tokens。
- 本地 octopus 或 OpenAI 兼容接口提示 `call_id` 为空或工具调用配对错误。

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
| `compression_threshold` | `30000` | Runner 层触发压缩的粗估 token 阈值。 |
| `provider_compression_threshold` | `50000` | Provider 层发送前触发压缩的粗估 token 阈值。 |
| `hard_token_budget` | `32000` | Provider 发送前的 messages 硬预算，压缩后仍超出会继续裁剪。 |
| `max_system_chars` | `12000` | 硬裁剪时单条 system 消息最多保留的字符数。 |
| `max_message_chars` | `6000` | 硬裁剪时单条普通消息最多保留的字符数。 |
| `max_tool_chars` | `3000` | 硬裁剪时单条 tool/function 消息最多保留的字符数。 |
| `replace_inline_images_over_chars` | `12000` | 超过该字符数的内联图片 data URL 会替换为文本占位。 |
| `keep_recent_rounds` | `4` | 压缩时保留最近的非 system 消息数。 |
| `summary_max_tokens` | `1024` | 摘要最大 token 数，当前作为配置保留。 |
| `max_older_messages_for_summary` | `20` | 最多取多少条早期消息参与摘要文本构造。 |
| `enable_learning` | `true` | 是否记录 length error 日志。 |
| `enable_provider_patch` | `true` | 是否启用 OpenAI Provider 层兜底 patch。 |
| `enable_retry_on_length_error` | `true` | 捕获 length 错误后是否自动重试。 |
| `min_completion_tokens` | `512` | 正常请求的最小输出 token 预算。 |
| `retry_completion_tokens` | `8192` | length 错误重试时使用的输出 token 预算。 |
| `force_reasoning_effort_low` | `false` | 是否将 `reasoning_effort` 降为 `low`。默认关闭，不全局降级推理质量。 |
| `sanitize_tool_call_pairs` | `true` | 是否修复压缩后的工具调用配对关系。 |
| `max_retry_count` | `1` | 同一 length 错误的最大重试次数。默认 1 次，超出后交给 AstrBot 核心 fallback。 |

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

### 查看插件配置摘要

```text
/length_compress_test
```

返回当前压缩、重试、Provider patch 等配置状态。

## 工作机制

### Runner 层

插件会 patch `ToolLoopAgentRunner._iter_llm_responses_with_fallback`：

1. 请求前按阈值检查上下文长度。
2. 小上下文请求走快速路径（跳过 deepcopy + json.dumps），降低 CPU 开销。
3. 超阈值时将旧消息折叠为摘要，保留最近消息。
4. 捕获 length 截断错误时记录日志。
5. 强制压缩上下文、提高输出预算后重试（受 `max_retry_count` 限制）。
6. 预处理出错时自动降级为原始请求，不中断用户对话。

### OpenAI Provider 层

插件会 patch `ProviderOpenAIOfficial._query` 和 `_query_stream`：

1. 深拷贝 payload，避免直接污染原始请求对象。
2. 修正 `max_tokens` / `max_completion_tokens`。
3. 可将 `reasoning_effort` 降为 `low`。
4. 根据阈值压缩 `messages`。
5. 修复 assistant tool_call 与 tool 输出配对。
6. 移除孤立或没有有效 ID 的工具输出。
7. 对压缩后仍超出 `hard_token_budget` 的请求继续做硬裁剪。
8. 将巨型内联图片 data URL 替换为占位文本。
9. 捕获 length/context 错误后强制压缩并重试（受 `max_retry_count` 限制）。
10. 预处理出错时自动降级为原始请求，不中断用户对话。

## 与 callid_sanitizer 的关系

`astrbot_plugin_callid_sanitizer` 主要负责请求发送前清理过长或无效的工具调用 ID。本插件主要负责 length 截断和长上下文恢复。

两个插件可以同时启用：

- `callid_sanitizer` 更偏向 ID 卫生和空/超长 ID 防护。
- `length_error_handler` 更偏向上下文压缩、输出预算和 length 错误重试。

如果遇到本地 octopus 报错 `Invalid input[x].call_id: empty string`，建议同时更新两个插件，并清理当前出错会话的历史上下文。

## 排错建议

### 更新后仍然报 `call_id` 为空

旧会话历史里可能已经保存了坏的工具调用记录。请尝试：

1. 清空该会话上下文或新建会话。
2. 重启 AstrBot。
3. 确认 `sanitize_tool_call_pairs=true`。
4. 确认 `astrbot_plugin_callid_sanitizer` 也已更新到最新版本。

### 长对话仍然触发 length 错误

可以尝试降低阈值：

```yaml
compression_threshold: 25000
provider_compression_threshold: 30000
hard_token_budget: 24000
keep_recent_rounds: 3
retry_completion_tokens: 4096
```

### 工具调用链容易报 400

建议保持：

```yaml
sanitize_tool_call_pairs: true
enable_provider_patch: true
```

并尽量避免多个会改写 messages 的插件同时做激进压缩。

### octopus 上游提示 `context_length_exceeded`

如果上游返回 `Your input exceeds the context window`，而 AstrBot 侧显示 `relay stream ended without a terminal result` 或 502，通常是中继层把真实上下文超限错误包装成了流式 502。

建议先确认插件已更新并重启 AstrBot，然后把小上下文模型的配置调低：

```yaml
compression_threshold: 24000
provider_compression_threshold: 24000
hard_token_budget: 24000
replace_inline_images_over_chars: 12000
```

如果旧会话历史里已经保存了大段 `data:image/...;base64`，可以清空该会话上下文或新建会话；新版插件会在后续请求发送前把这类巨型内联图片替换为占位文本。

## 注意事项

- 本插件通过 monkey patch 注入 AstrBot 内部 Runner 和 OpenAI Provider，AstrBot 内部 API 大版本变化时可能需要适配。
- 本插件只能降低 length 类错误概率，不能突破模型本身的最大上下文窗口或最大输出限制。
- 插件会清理孤立工具输出；极少数情况下，过旧工具结果会被摘要替代，不再保留原始完整内容。
- 建议在更新插件后重启 AstrBot。

## 版本信息

- 插件标识：`astrbot_plugin_length_error_handler`
- 当前版本：`1.0.9`
- 作者：牧濑红莉栖（BOT）
- 仓库：https://github.com/x1051445024/astrbot_plugin_length_error_handler

## 更新日志

### v1.0.9

**性能优化：**
- 小上下文请求自动走快速路径，跳过 deepcopy 和 json.dumps 估算，降低 CPU 开销。
- 新增 `_quick_estimate_chars` 和 `_content_char_len` 轻量估算方法。

**防御性容错：**
- 三个 patched 函数（Runner 层 `patched_iter`、Provider 层 `patched_query` 和 `patched_query_stream`）的预处理逻辑加了 try/except 防御。
- 预处理出错时自动降级为原始请求，插件逻辑出错不中断用户对话。

**重试控制：**
- 新增 `max_retry_count` 配置项（默认 1），同一 length 错误最多由本插件重试指定次数。
- 超出重试上限后干净地交给 AstrBot 核心 fallback 机制，避免多层重试叠加。

**其他改进：**
- `force_reasoning_effort_low` 默认值改为 `false`，不再全局降级推理模型质量。
- `_is_length_error` 收紧匹配逻辑，减少误判中转 API 参数校验错误为 length 截断。
- `_ensure_runner_budget` 只在 retry 时修改配置，避免正常请求的副作用残留。
