"""
RAGAS RAG 质量评估脚本

用法:
    python scripts/evaluate_ragas.py                    # 运行所有指标
    python scripts/evaluate_ragas.py --metrics faithfulness,context_precision
    python scripts/evaluate_ragas.py --test-file path/to/qa.json
    python scripts/evaluate_ragas.py --mode plan         # 指定 RAG 执行模式
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import RagasEvaluator
from core.rag_service import RagService


async def main():
    parser = argparse.ArgumentParser(description="RAGAS RAG 质量评估")
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall",
        help="评估指标，逗号分隔（默认全部 4 项）",
    )
    parser.add_argument(
        "--test-file",
        default=None,
        help="测试数据 JSON 路径（默认 data/qa_pairs/test_qa.json）",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "react", "plan", "multi_agent"],
        help="RAG 执行模式（默认 auto）",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存报告到文件",
    )
    args = parser.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print(f"初始化 RAG 服务...")
    rag = RagService()
    rag.sync_bm25()

    print(f"加载测试数据...")
    test_data = RagasEvaluator.load_test_data(args.test_file)
    if not test_data:
        print("无测试数据。请在 data/qa_pairs/test_qa.json 放置 QA 对，或通过 --test-file 指定路径。")
        return

    print(f"测试数据: {len(test_data)} 条")
    print(f"执行模式: {args.mode}")
    print(f"评估指标: {metrics}")
    print(f"开始评估...\n")

    evaluator = RagasEvaluator()
    report = await evaluator.run(rag_service=rag, test_data=test_data, metrics=metrics)

    # 输出结果
    scores = report.get("scores", {})
    print("\n" + "=" * 50)
    print("RAGAS 评估结果")
    print("=" * 50)
    for m, v in scores.items():
        if v is not None:
            print(f"  {m}: {v:.4f}" if isinstance(v, float) else f"  {m}: {v}")
        else:
            print(f"  {m}: N/A（无法计算）")
    print("=" * 50)

    if not args.no_save:
        evaluator.save_report(report)
        print(f"\n报告已保存到 data/ 目录")


if __name__ == "__main__":
    asyncio.run(main())
