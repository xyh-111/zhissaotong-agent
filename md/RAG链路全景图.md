# RAG 链路全景图

> 把 RAG 从「离线建库」到「在线检索」的完整链路串起来，附带真实踩过的坑。
> 核心结论：RAG = 离线把文档切好、向量化、存进 ChromaDB；在线提问时检索 Top-K、拼进 Prompt、让模型「基于参考资料」总结生成。

---

## 图 1：整体架构（两条线）

```
┌─────────────────── 离线建库（一次性 / 增量）───────────────────┐
│                                                                │
│  data/ 下的 PDF / TXT                                           │
│     │  listdir_with_allowed_type 筛选                           │
│     ▼                                                          │
│  算 MD5 ──已处理过?──▶ 跳过（增量去重）                          │
│     │ 否                                                       │
│     ▼                                                          │
│  加载文档（PyPDFLoader / TextLoader）                            │
│     │                                                          │
│     ▼                                                          │
│  递归分片（chunk_size=200, overlap=20）                          │
│     │                                                          │
│     ▼                                                          │
│  向量化（embedding 模型）                                        │
│     │                                                          │
│     ▼                                                          │
│  存入 ChromaDB（persist_directory=chroma_db）                    │
│     │                                                          │
│     ▼                                                          │
│  记录 MD5 到 md5.text（下次跳过）                                 │
└────────────────────────────────────────────────────────────────┘

┌─────────────────── 在线检索（每次提问）─────────────────────────┐
│                                                                │
│  用户提问 query                                                  │
│     │  rag_summarize(query)                                    │
│     ▼                                                          │
│  retriever.invoke(query) ──▶ 向量相似度 Top-K（k=3）             │
│     │                                                          │
│     ▼                                                          │
│  拼接 context："[参考资料1]... [参考资料2]..."                     │
│     │                                                          │
│     ▼                                                          │
│  prompt | model | StrOutputParser  （LCEL 链）                   │
│     │  Prompt 强约束：只基于参考资料、不编造                       │
│     ▼                                                          │
│  总结回答（纯文本字符串）                                          │
└────────────────────────────────────────────────────────────────┘
```

---

## 图 2：离线建库流程（vector_store.py 的 load_document）

```
load_document()
   │
   ▼
listdir_with_allowed_type(data/, ("txt","pdf"))   ← 拿到所有知识库文件
   │
   ▼
for 每个文件:
   │
   ├─ get_file_md5_hex(path)          ← 4KB 分片算 MD5，防大文件爆内存
   │
   ├─ check_md5_hex(md5)？            ← 在 md5.text 里查有没有处理过
   │     └─ 是 → 跳过（logger: "内容已存在知识库，跳过"）
   │
   ├─ get_file_documents(path)        ← txt→TextLoader / pdf→PyPDFLoader
   │
   ├─ splitter.split_documents(docs)  ← 递归分片
   │
   ├─ vctor_store.add_documents(分片)  ← 向量化 + 入库
   │
   └─ save_md5_hex(md5)               ← 记录 MD5，下次跳过
```

**关键设计：MD5 增量去重**

每个文件算 MD5 指纹，和 `md5.text` 里已处理的记录比对，命中就跳过。这样知识库可以**增量加载**——新加文档只处理新增的，不用每次全量重跑。这是「幂等」思想的落地。

---

## 图 3：在线检索流程（rag_service.py 的 rag_summarize）

```
rag_summarize(query)
   │
   ▼
retriever_docs(query) → retriever.invoke(query)
   │    返回 Top-K（k=3）个 Document，每个含 page_content + metadata
   ▼
拼接 context：
   "[参考资料1]：参考资料：{page_content} | 参考元数据：{metadata}\n"
   "[参考资料2]：..."
   │
   ▼
chain.invoke({"input": query, "context": context})
   │
   │  chain = prompt_template | print_prompt | model | StrOutputParser()
   │   ── 这就是 LCEL：Prompt → 模型 → 字符串解析
   ▼
返回总结后的纯文本回答
```

**Prompt 强约束（prompts/rag_summarize.txt 的核心）**

1. **事实准确**：回答必须完全基于参考资料，不编造、不添加、不做主观推断
2. **聚焦提问**：严格围绕原始提问，不扩充范围、不构造新 query
3. **格式**：仅输出概括内容本身，纯文本字符串，不包 JSON/字典

这三条就是 RAG 防幻觉的工程手段——用 Prompt 把模型"锁"在检索到的资料范围内。

---

## 图 4：分片细节（为什么这么设）

配置在 config/chroma.yml：

```yaml
chunk_size: 200        # 每片约 200 字符
chunk_overlap: 20      # 相邻片重叠 20 字符
separators: ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", " "]
k: 3                   # 检索 Top-K
```

- **chunk_size=200**：太小语义不完整、检索碎片化；太大颗粒太粗、相关度低
- **chunk_overlap=20**：切分边界处保留重叠，避免一句话被从中间切断、丢失语义
- **separators 优先级**：先按段落（\n\n）切，再按换行、标点，最后按空格——尽量在"语义边界"处切
- **k=3**：取最相关的 3 段，太少信息不足、太多引入噪声

---

## 真实踩过的坑（面试加分项）

### 坑 1：embedding 模型的接口选择

**现象**：`qwen3.7-text-embedding` 走 OpenAI 兼容接口 `/embeddings` 时报 `input.contents` 格式错误，连标准的 `text-embedding-v4` 也报同样的错。

**定位**：DashScope 的 OpenAI 兼容 `/embeddings` 接口对 embedding 模型有输入格式缺陷；原生 `dashscope.TextEmbedding` 接口才能正确调用。

**解决**：向量模型走**原生 `DashScopeEmbeddings`**，对话模型才走兼容接口。用工厂模式把两种接口抽象掉，切换只改配置。

### 坑 2：embedding 的批大小限制

**现象**：重建向量库时，`qwen3.7-text-embedding` 报 `batch size should not be larger than 20`。

**定位**：该模型单次最多 20 条；而 `DashScopeEmbeddings` 对不认识的模型默认按 25 条一批发。

**解决**：封装 `DashScopeTextEmbeddings`，把 `embed_documents` 手动拆成 ≤20 一批再发。

### 坑 3：换 embedding 模型后旧向量库失效

**现象**：切换 embedding 模型后，旧 `chroma_db` 的向量维度/分布对不上新模型。

**解决**：写了 `rebuild_vector_store.py` 一键重建——删旧向量库 + 清 MD5 记录 + 用新模型重新入库。

---

## 代码对照表

| 环节 | 项目代码 | 说明 |
|---|---|---|
| 文档加载 | `utils/file_handler.py` 的 `pdf_loader`/`txt_loader` | PyPDFLoader / TextLoader |
| MD5 计算 | `utils/file_handler.py` 的 `get_file_md5_hex` | 4KB 分片防大文件 |
| 分片 | `rag/vector_store.py` 的 `RecursiveCharacterTextSplitter` | chunk=200 / overlap=20 |
| 向量库 | `rag/vector_store.py` 的 `Chroma` | collection=agent，持久化 chroma_db |
| 检索 | `rag/vector_store.py` 的 `ger_retriever` | `as_retriever(k=3)` |
| 总结链 | `rag/rag_service.py` 的 `_init_chain` | `prompt | model | parser` |
| 总结 Prompt | `prompts/rag_summarize.txt` | 防幻觉强约束 |
| 增量去重 | `rag/vector_store.py` 的 `load_document` | MD5 指纹比对 |
| embedding 批处理 | `model/factory.py` 的 `DashScopeTextEmbeddings` | 批大小 ≤20 |

---

## 一句话总述（可接住任何追问）

> RAG 分两条线：离线建库时，遍历知识库文件、算 MD5 去重、加载 PDF/TXT、按 chunk=200/overlap=20 递归分片、embedding 向量化后存进 ChromaDB；在线检索时，用 retriever 按向量相似度取 Top-K（k=3），拼成「参考资料」塞进 Prompt，走 `prompt | model | parser` 这条 LCEL 链，靠 Prompt 强约束「只基于参考资料、不编造」来防幻觉。过程中还解决了 embedding 双接口、批大小限制、模型切换后旧库失效这几个工程问题。
