# 智扫通智能客服 Agent

面向扫地机器人/扫拖一体机器人的智能客服系统。基于 LangGraph 构建具备自主推理与工具调用能力的 ReAct Agent，结合 RAG 检索增强与场景化动态提示词切换，覆盖日常咨询、故障排查、个性化使用报告等场景，并自建评测体系量化 Agent 质量。

## 技术栈

- **Agent 编排**：LangGraph（`create_agent` + 中间件）
- **组件/模型**：LangChain、通义千问（对话 `qwen3.8-27b`，向量 `qwen3.7-text-embedding`）
- **向量数据库**：ChromaDB
- **前端**：Streamlit
- **配置**：YAML

## 目录结构

```
app.py                    # Streamlit 前端入口
agent/
  react_agent.py          # ReAct Agent 编排
  agent_tools.py          # 7 个工具定义
  middleware.py           # 工具监控 / 模型日志 / 动态提示词
rag/
  rag_service.py          # RAG 总结链
  vector_store.py         # 文档加载 + 分片 + 向量化 + 检索 + MD5 去重
model/factory.py          # 模型工厂（LLM / Embedding）
eval/                     # 评测模块（测试集 + LLM-as-Judge + 指标）
config/                   # YAML 配置（模型名、向量库、提示词路径等）
prompts/                  # 提示词模板
data/                     # 知识库文档 + 外部用户记录
utils/                    # 配置加载 / 日志 / 文件处理 / 路径工具
chroma_db/                # 向量库持久化目录（运行时生成）
```

## 环境准备

### 1. Python 环境

项目依赖安装在 `E:\python3.11` 环境中（注意：本机另有一个 Anaconda 环境，缺少 `langchain-chroma` 等依赖，**务必使用 `E:\python3.11\python.exe`**）。

### 2. 安装依赖

```powershell
E:\python3.11\python.exe -m pip install langchain langgraph langchain-community langchain-chroma langchain-openai langchain-text-splitters chromadb dashscope streamlit pyyaml
```

### 3. 配置 API Key

```powershell
$env:DASHSCOPE_API_KEY = "你的通义千问 API Key"
```

## 启动步骤

### 第一步：建向量库

首次启动、或更换过 embedding 模型后，需要重建向量库（把 `data/` 下的知识库文档分片、向量化、存入 `chroma_db`）：

```powershell
E:\python3.11\python.exe rebuild_vector_store.py
```

> 仅新增知识库文档、未换模型时，可改用增量加载（靠 MD5 去重，只处理新增文件）：
> ```powershell
> E:\python3.11\python.exe rag\vector_store.py
> ```

### 第二步：启动界面

```powershell
E:\python3.11\python.exe -m streamlit run app.py
```

启动后浏览器自动打开（默认 `http://localhost:8501`），即可在聊天框与智能客服对话。

## 配置说明

| 配置文件 | 关键项 | 说明 |
|---|---|---|
| `config/rag.yml` | `chat_model_name` | 对话模型（走 OpenAI 兼容接口） |
| | `embeddings_model_name` | 向量模型（走 DashScope 原生接口） |
| `config/chroma.yml` | `chunk_size` / `chunk_overlap` | 分片参数 |
| | `k` | 检索 Top-K |
| | `persist_directory` / `data_path` | 向量库目录 / 知识库目录 |
| `config/eval.yml` | `judge_model_name` / `judge_temperature` | 评测裁判模型与温度 |

> 切换模型只改 `config/rag.yml`，业务代码零改动；改完对话模型无需重建向量库，改 embedding 模型则需重建。

## 评测（可选）

```powershell
# 全量评测（42 条测试集）
E:\python3.11\python.exe -m eval.run_eval

# 预览单条用例（打印完整链路：工具调用/回答/裁判打分/耗时）
E:\python3.11\python.exe -m eval.run_eval --dry-run --id qa_01

# 只跑前 N 条
E:\python3.11\python.exe -m eval.run_eval --limit 5
```

评测结果输出到 `eval/reports/eval_report.md`（Markdown 报告）与 `results.json`（逐用例原始结果）。


## 相关文档

- [Agent运行全景图](md/Agent运行全景图.md) — LangGraph 执行链路
- [RAG链路全景图](md/RAG链路全景图.md) — RAG 建库/检索链路
- [评测体系全景图](md/评测体系全景图.md) — 评测方法与指标

