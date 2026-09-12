# Agent 运行全景图

> 把 LangGraph 的各个知识点串成一条「从用户提问到最终回答」的完整执行链路。
> 核心结论：整个 Agent 本质是一张 LangGraph 状态图，`create_agent` 建好 model / tools 两个节点和一个循环，中间件以「包一层」和「插节点」两种方式挂进去。

---

## 图 1：静态结构（create_agent 生成了什么）

```
用户 (Streamlit 输入 prompt)
   │
   ▼
ReactAgent.execute_stream(prompt)
   │  self.agent.stream(input_dict, stream_mode="values", context={"report": False})
   ▼
┌─────────────────── create_agent 编译出的 CompiledStateGraph ───────────────────┐
│                                                                                │
│  START ──▶ [before_model] ──▶ [model] ──有 tool_calls?──▶ [tools]              │
│              log_before_model     ▲                        │                   │
│                                   │                        │ (Send 并行扇出)    │
│                                   └────────────────────────┘                   │
│                                   没有 tool_calls 了 → 往下                     │
│                                   ▼                                            │
│                                  END ──▶ 返回完整 messages                      │
│                                                                                │
│  外加两个「隐藏钩子」：                                                            │
│   - [dynamic_prompt] report_prompt_switch：每次调 model 前决定用哪套提示词        │
│   - [wrap_tool_call] monitor_tool：包在每个工具执行的外层做监控                   │
└────────────────────────────────────────────────────────────────────────────────┘
   │  stream_mode="values" 每步 yield 最新的 message
   ▼
Streamlit 逐字渲染
```

**记住这个骨架**：一张图、两个核心节点（model / tools）、一个回环、三个钩子（before_model / dynamic_prompt / wrap_tool_call）。

---

## 图 2：ReAct 循环怎么转（动态跳转）

```
        ┌────────────── 进入循环 ──────────────┐
        ▼                                      │
   [before_model] 打日志                        │
        ▼                                      │
   [dynamic_prompt] 读 runtime.context 决定提示词 │
        ▼                                      │
   [model] bind_tools(7个工具) → invoke          │
        │                                      │
        ├─ 输出 tool_calls? ───是──▶ [tools]    │
        │                              │       │
        │                        ToolNode 执行工具│
        │                      结果包装成 ToolMessage│
        │                              │       │
        │                              └──▶ 回到循环入口（可能 Send 并行）
        │                                      │
        └─ 没有 tool_calls ──▶ END（退出条件，factory.py:1670）
```

**核心一句话**：model 节点吐出 `tool_calls` → 图把流程送到 tools → 工具结果回填 messages → 再回 model，直到 model 不再吐 tool_calls。这就是「思考→行动→观察→再思考」的工程实现。

---

## 图 3：单个工具调用在 ToolNode 内部的执行链

```
model 吐出一个 tool_call {name, args, id}
   │
   ▼
ToolNode._run_one
   │  ① 按 name 查工具实例 tools_by_name.get(name)
   │  ② 组装 ToolCallRequest(tool_call, tool, state, runtime)
   │
   ▼
┌────────── wrap_tool_call 洋葱皮（你的 monitor_tool 在这里）──────────┐
│  monitor_tool(request, handler):                                   │
│    记录 request.tool_call["name"] / ["args"]   ← 执行前              │
│    result = handler(request)                   ← 触发真正执行        │
│    记录调用成功 / 失败                          ← 执行后              │
│    若工具是 fill_context_for_report:                                 │
│        request.runtime.context["report"] = True  ← 翻报告标志位       │
└────────────────────────────────────────────────────────────────────┘
   │  handler 内部 ↓
   ▼
_execute_tool_sync
   │  ③ 注入 state/runtime 参数（如果需要）
   │  ④ tool.invoke(args)  ← 真正调用工具函数
   │  ⑤ 结果包装成 ToolMessage(content, tool_call_id=call["id"])
   ▼
ToolMessage 回填到 state["messages"]（靠 tool_call_id 对上号）
```

**对应项目实际代码**：`monitor_tool` 里的 `request` 就是这个 `ToolCallRequest`，`handler(request)` 就是触发 `tool.invoke` 的回调。`fill_context_for_report` 触发标志位翻转，正是发生在第 ② 层洋葱皮里。

---

## 图 4：完整时间线（用「生成使用报告」串起所有知识点）

```
1. 用户问"生成我的使用报告"
2. execute_stream 启动，context={"report": False}  ← runtime 上下文初值
3. [before_model] log_before_model 打日志
4. [dynamic_prompt] 读 context["report"]==False → 用「客服提示词」
5. [model] 模型思考 → 决定先要用户 ID → 输出 tool_call{get_user_id}
6. [tools] monitor_tool 记录 → 执行 get_user_id → 返回 ToolMessage("1001")
7. 回 model：模型继续思考 → tool_call{get_current_month} → 返回 "2025-02"
8. 回 model：tool_call{fill_context_for_report}
   └─▶ monitor_tool 里 context["report"] 被翻成 True  ⭐关键
9. 回 model：tool_call{fetch_external_data(1001, 2025-02)} → 返回使用记录
10. 回 model：此时 [dynamic_prompt] 读到 context["report"]==True
    └─▶ 切换成「报告写手提示词」  ⭐提示词切换生效
11. [model] 生成最终 Markdown 报告，不再吐 tool_calls
12. 路由到 END，execute_stream yield 完整回答 → Streamlit 流式输出
```

**这条时间线串起了全部五个知识点**：图结构（3-5-6 的循环）、ToolNode（6-9 每次工具执行）、Send（9 若一次调多工具）、wrap_tool_call 中间件（8 的标志位翻转）、runtime.context + dynamic_prompt（2 初值 + 8 翻转 + 10 读取切换）。

---

## 代码对照表（面试被问到哪，就指哪）

| 环节 | 项目代码 | 框架源码 |
|---|---|---|
| 图入口 / 流式启动 | `agent/tools/react_agent.py` 的 `execute_stream` | `langchain/agents/factory.py` 的 `create_agent` |
| 工具定义 | `agent/tools/agent_tools.py` | `@tool` → `bind_tools` |
| 工具监控 + 标志位翻转 | `agent/tools/middleware.py` 的 `monitor_tool` | `langgraph/prebuilt/tool_node.py` 的 `_run_one` |
| 模型前日志 | `agent/tools/middleware.py` 的 `log_before_model` | `before_model` 节点 |
| 提示词切换 | `agent/tools/middleware.py` 的 `report_prompt_switch` | `dynamic_prompt` 钩子 |
| 退出循环判断 | （框架自动） | factory.py 的 `_make_model_to_tools_edge` |
| RAG 检索 | `rag/rag_service.py` | `rag_summarize` 工具内部 |

---

## 一句话总述（可接住任何追问）

> 整个 Agent 本质是一张 LangGraph 状态图：`create_agent` 建好 model、tools 两个节点和一个循环。用户提问进来，`before_model` 打日志、`dynamic_prompt` 决定提示词，model 节点用 `bind_tools` 绑上工具去推理，吐出 tool_calls 就路由到 ToolNode 执行——执行被 `wrap_tool_call` 包着做监控，其中一个工具会把 `runtime.context` 的标志位翻转，下一轮 `dynamic_prompt` 读到就切换提示词。工具结果以 ToolMessage 回填，靠 tool_call_id 对上号，循环到 model 不再吐 tool_calls 才结束，最后流式吐给前端。
