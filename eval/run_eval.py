"""
评测入口
用法：
  1. 确认已配置通义千问 DashScope API Key（环境变量 DASHSCOPE_API_KEY）
  2. 确认向量库已就绪（首次可先运行 rag/vector_store.py 的 load_document 加载知识库）
  3. 运行：
       python -m eval.run_eval                       # 全量评测
       python -m eval.run_eval --dry-run             # 预览第一条，打印详细中间过程
       python -m eval.run_eval --dry-run --id rpt_01 # 预览指定用例
       python -m eval.run_eval --id qa_01            # 只跑指定用例并出报告
       python -m eval.run_eval --limit 3             # 只跑前 3 条并出报告
结果会打印到控制台，并写入 eval/reports/eval_report.md 与 results.json。
"""
import argparse

from eval.evaluator import AgentEvaluator


def main():
    parser = argparse.ArgumentParser(description="智扫通 Agent 评测")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览单条用例并打印详细中间过程（默认第一条，可配 --id）")
    parser.add_argument("--id", type=str, default=None,
                        help="指定用例 ID，如 qa_01 / rpt_01")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 条用例")
    args = parser.parse_args()

    evaluator = AgentEvaluator()

    if args.dry_run:
        evaluator.preview(args.id)
        return

    case_ids = [args.id] if args.id else None
    evaluator.run(case_ids=case_ids, limit=args.limit)
    evaluator.print_report()


if __name__ == "__main__":
    main()
