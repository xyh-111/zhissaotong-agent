"""
Agent 评测器
负责：加载测试集 -> 逐个运行 Agent -> 采集工具调用/回答/延迟 -> LLM-as-Judge 打分 -> 汇总指标
"""
import json
import os
import time

from langchain_core.messages import AIMessage

from agent.tools.react_agent import ReactAgent
from agent.tools.agent_tools import rag as rag_service
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from eval.judge import judge_task_completion, judge_hallucination


def _msg_text(message) -> str:
    """把消息的 content 统一转成字符串（兼容 str / list[dict] 两种格式）"""
    content = getattr(message, "content", "") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class AgentEvaluator:
    def __init__(self, test_set_path: str = None):
        self.agent = ReactAgent()
        # 复用 agent_tools 里已实例化的 RAG 服务，避免重复初始化向量库与 embedding
        self.retriever = rag_service.retriever
        self.test_set_path = test_set_path or get_abs_path("eval/test_cases.json")
        self.cases = self._load_cases()
        self.results = []

    def _load_cases(self) -> list:
        with open(self.test_set_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["cases"]

    def _run_case(self, case: dict) -> dict:
        """运行单条用例，返回工具调用列表、最终回答、首字延迟、总耗时"""
        input_dict = {"messages": [{"role": "user", "content": case["query"]}]}
        tool_calls = []
        final_answer = ""
        ttft = None
        start = time.perf_counter()

        # 双模式流式：
        #  - updates 模式取「完整」的工具调用与最终回答（无 token 级拼接问题）
        #  - messages 模式取首字延迟（TTFT，首个模型输出 token 的耗时）
        for mode, chunk in self.agent.agent.stream(
            input_dict,
            stream_mode=["updates", "messages"],
            context={"report": False},
        ):
            if mode == "messages":
                msg_chunk, meta = chunk
                if ttft is None and meta.get("langgraph_node") == "model":
                    if _msg_text(msg_chunk).strip():
                        ttft = time.perf_counter() - start
            elif mode == "updates":
                for _node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    for m in update.get("messages", []):
                        if isinstance(m, AIMessage):
                            for tc in getattr(m, "tool_calls", None) or []:
                                tool_calls.append({
                                    "name": tc.get("name"),
                                    "args": tc.get("args", {}),
                                })
                            # 最终回答 = 不带工具调用的模型输出内容
                            if not getattr(m, "tool_calls", None) and _msg_text(m).strip():
                                final_answer = _msg_text(m)

        total_time = time.perf_counter() - start
        return {
            "tool_calls": tool_calls,
            "final_answer": final_answer.strip(),
            "ttft": ttft,
            "total_time": total_time,
        }

    @staticmethod
    def _check_report_format(answer: str) -> bool:
        """报告格式合规：包含 Markdown 标题，且出现报告/建议/保养相关内容"""
        if not answer:
            return False
        has_heading = any(line.lstrip().startswith("#") for line in answer.splitlines())
        has_content = ("报告" in answer) and ("建议" in answer or "保养" in answer)
        return has_heading and has_content

    def _evaluate_case(self, case: dict, run: dict) -> dict:
        m = {}
        answer = run["final_answer"]
        ref = case.get("reference_points", [])

        # 1. 任务完成率（LLM-as-Judge）
        completed, c_reason = judge_task_completion(case["query"], ref, answer)
        m["completed"] = completed
        m["completion_reason"] = c_reason

        # 2. 幻觉率（LLM-as-Judge）
        has_hallu, h_reason = judge_hallucination(case["query"], ref, answer)
        m["has_hallucination"] = has_hallu
        m["hallucination_reason"] = h_reason

        # 3. 工具选择 / 参数准确率（直接比对，无需裁判）
        expected = case.get("expected_tools", [])
        if expected:
            hit = 0
            for name in expected:
                matched = next((tc for tc in run["tool_calls"] if tc["name"] == name), None)
                if matched is None:
                    continue
                expected_args = (case.get("expected_tool_args") or {}).get(name)
                if expected_args:
                    # 参数做宽松子串匹配
                    if all(str(expected_args[k]) in str(matched["args"].get(k, "")) for k in expected_args):
                        hit += 1
                else:
                    hit += 1
            m["tool_accuracy"] = hit / len(expected)
        else:
            m["tool_accuracy"] = None
        m["tool_call_count"] = len(run["tool_calls"])

        # 4. 检索命中率（Recall@K 代理：Top-K 结果中是否命中标注关键词）
        if case.get("needs_rag"):
            docs = self.retriever.invoke(case["query"])
            keywords = case.get("relevant_keywords", [])
            m["retrieval_hit"] = any(
                any(k in d.page_content for k in keywords) for d in docs
            )
        else:
            m["retrieval_hit"] = None

        # 5. 报告格式合规率
        if case.get("check_report_format"):
            m["report_format_ok"] = self._check_report_format(answer)
        else:
            m["report_format_ok"] = None

        # 6. 延迟
        m["ttft"] = run["ttft"]
        m["total_time"] = run["total_time"]

        return m

    def _find_case(self, case_id: str = None) -> dict | None:
        """按 ID 查找用例；未指定时返回第一条"""
        if case_id:
            for c in self.cases:
                if c["id"] == case_id:
                    return c
            logger.error(f"[eval]未找到用例：{case_id}")
            return None
        return self.cases[0] if self.cases else None

    def run(self, case_ids: list = None, limit: int = None) -> list:
        """跑测试集并评测；可通过 case_ids / limit 只跑子集"""
        cases = self.cases
        if case_ids:
            cases = [c for c in self.cases if c["id"] in case_ids]
        if limit:
            cases = cases[:limit]

        self.results = []
        total = len(cases)
        for idx, case in enumerate(cases, 1):
            logger.info(f"[eval] ({idx}/{total}) {case['id']} [{case['category']}] {case['query']}")
            run = self._run_case(case)
            metrics = self._evaluate_case(case, run)
            self.results.append({"case": case, "run": run, "metrics": metrics})
        return self.results

    def preview(self, case_id: str = None) -> dict | None:
        """单条预览：跑一条用例并打印详细中间过程（工具调用、回答、裁判打分、耗时），用于流程验证与耗时估算"""
        case = self._find_case(case_id)
        if case is None:
            return None

        print("=" * 60)
        print(f"用例：{case['id']} [{case['category']}]")
        print(f"问题：{case['query']}")
        print("=" * 60)

        run = self._run_case(case)
        metrics = self._evaluate_case(case, run)

        print(f"\n[工具调用] 共 {len(run['tool_calls'])} 次：")
        for tc in run["tool_calls"]:
            print(f"  - {tc['name']} {tc['args']}")

        print(f"\n[最终回答]\n{run['final_answer'] or '（空）'}")

        print("\n[裁判打分]")
        print(f"  任务完成：{'是' if metrics['completed'] else '否'} —— {metrics['completion_reason']}")
        print(f"  幻觉：{'是' if metrics['has_hallucination'] else '否'} —— {metrics['hallucination_reason']}")
        if metrics['tool_accuracy'] is not None:
            print(f"  工具准确率：{metrics['tool_accuracy']:.0%}（预期工具 {case.get('expected_tools', [])}）")
        if metrics['retrieval_hit'] is not None:
            print(f"  检索命中：{'是' if metrics['retrieval_hit'] else '否'}")
        if metrics['report_format_ok'] is not None:
            print(f"  报告格式合规：{'是' if metrics['report_format_ok'] else '否'}")

        if metrics['ttft'] is not None:
            print(f"\n[耗时] 首字 {metrics['ttft']:.2f}s / 端到端 {metrics['total_time']:.2f}s")
        else:
            print(f"\n[耗时] 端到端 {metrics['total_time']:.2f}s")

        total_cases = len(self.cases)
        est_min = metrics["total_time"] * total_cases / 60
        print(f"[估算] 全量 {total_cases} 条约需 {est_min:.1f} 分钟（单条约 {metrics['total_time']:.1f}s）")

        return {"case": case, "run": run, "metrics": metrics}

    def _avg(self, values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def aggregate(self) -> dict:
        """汇总整体指标"""
        total = len(self.results)
        if total == 0:
            return {}

        completed = sum(1 for r in self.results if r["metrics"]["completed"])
        hallucinated = sum(1 for r in self.results if r["metrics"]["has_hallucination"])

        tool_cases = [r for r in self.results if r["metrics"]["tool_accuracy"] is not None]
        rag_cases = [r for r in self.results if r["metrics"]["retrieval_hit"] is not None]
        report_cases = [r for r in self.results if r["metrics"]["report_format_ok"] is not None]

        return {
            "总用例数": total,
            "任务完成率": completed / total,
            "幻觉率": hallucinated / total,
            "工具选择/参数准确率": self._avg([r["metrics"]["tool_accuracy"] for r in tool_cases]),
            "检索关键词命中率(Recall@K代理)": sum(1 for r in rag_cases if r["metrics"]["retrieval_hit"]) / len(rag_cases) if rag_cases else None,
            "报告格式合规率": sum(1 for r in report_cases if r["metrics"]["report_format_ok"]) / len(report_cases) if report_cases else None,
            "平均首字延迟(s)": self._avg([r["metrics"]["ttft"] for r in self.results]),
            "平均端到端延迟(s)": self._avg([r["metrics"]["total_time"] for r in self.results]),
            "平均工具调用次数": self._avg([r["metrics"]["tool_call_count"] for r in self.results]),
        }

    def _category_breakdown(self) -> dict:
        """按场景分类统计任务完成率与平均工具调用次数"""
        groups = {}
        for r in self.results:
            cat = r["case"]["category"]
            groups.setdefault(cat, []).append(r)

        out = {}
        for cat, items in groups.items():
            out[cat] = {
                "用例数": len(items),
                "任务完成率": sum(1 for i in items if i["metrics"]["completed"]) / len(items),
                "平均工具调用次数": self._avg([i["metrics"]["tool_call_count"] for i in items]),
            }
        return out

    def print_report(self) -> str:
        """生成并打印评测报告，同时落盘到 eval/reports/"""
        agg = self.aggregate()
        breakdown = self._category_breakdown()

        lines = ["# Agent 评测报告", ""]
        lines.append("## 一、整体指标")
        for k, v in agg.items():
            if v is None:
                lines.append(f"- **{k}**：N/A（无对应用例）")
            elif isinstance(v, float):
                lines.append(f"- **{k}**：{v:.2%}" if ("率" in k) else f"- **{k}**：{v:.2f}")
            else:
                lines.append(f"- **{k}**：{v}")
        lines.append("")

        lines.append("## 二、分场景指标")
        lines.append("| 场景 | 用例数 | 任务完成率 | 平均工具调用次数 |")
        lines.append("| --- | --- | --- | --- |")
        for cat, b in breakdown.items():
            lines.append(f"| {cat} | {b['用例数']} | {b['任务完成率']:.2%} | {b['平均工具调用次数']:.2f} |")
        lines.append("")

        lines.append("## 三、逐用例明细")
        lines.append("| ID | 场景 | 问题 | 完成 | 幻觉 | 工具调用 | 耗时(s) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in self.results:
            c, m = r["case"], r["metrics"]
            tools = ",".join(tc["name"] for tc in r["run"]["tool_calls"]) or "-"
            lines.append(
                f"| {c['id']} | {c['category']} | {c['query']} | {'✅' if m['completed'] else '❌'} "
                f"| {'⚠️' if m['has_hallucination'] else '✅'} | {tools} | {m['total_time']:.2f} |"
            )
        lines.append("")

        report_text = "\n".join(lines)

        # 落盘
        report_dir = get_abs_path("eval/reports")
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "eval_report.md"), "w", encoding="utf-8") as f:
            f.write(report_text)
        with open(os.path.join(report_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)

        print(report_text)
        return report_text
