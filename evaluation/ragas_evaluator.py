"""
RAGAS 评估器 — 基于 RAGAS 框架的标准化 RAG 质量评估

与现有 RAGEvaluator 并行工作，提供社区验证的指标：
- Faithfulness:     回答是否忠实于检索上下文（防幻觉）
- AnswerRelevancy:  回答与问题的相关性
- ContextPrecision: 检索上下文的精确度（信号/噪声比）
- ContextRecall:    检索上下文覆盖 ground truth 的程度

使用方式:
    evaluator = RagasEvaluator()
    report = await evaluator.run(rag_service, test_data)

依赖: pip install ragas>=0.4.0
"""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """RAGAS 标准化评估器

    - 复用现有 data/qa_pairs/test_qa.json 测试集
    - 先用 RAG 系统回答问题 → 再跑 RAGAS 指标
    - 与 RAGEvaluator 完全独立，可并行使用
    """

    def __init__(self, llm=None):
        """
        Args:
            llm: LangChain ChatModel（可选，默认用 DashScope ChatTongyi）
        """
        self._llm = llm
        self._results: list[dict] = []

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        from langchain_community.chat_models import ChatTongyi
        from config import settings as config
        return ChatTongyi(
            model=config.chat_model,
            dashscope_api_key=config.dashscope_api_key,
            temperature=0,
        )

    # ── 数据加载 ──

    @staticmethod
    def load_test_data(filepath: str = None) -> list[dict]:
        """加载测试数据（兼容现有 test_qa.json 格式）"""
        from config import settings as config
        filepath = filepath or os.path.join(config.DATA_DIR, "qa_pairs", "test_qa.json")
        if not os.path.exists(filepath):
            logger.warning("测试数据不存在: %s", filepath)
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── RAG 运行 ──

    async def _run_rag(self, rag_service, question: str) -> tuple[str, list[str]]:
        """运行 RAG 系统，返回 (answer, contexts)

        contexts: 从检索结果中提取的文档片段列表
        """
        result = await rag_service.chat(question, user_id="ragas_eval")
        answer = result.get("answer", "")
        context_text = result.get("context", "")

        # 把 context 拆成独立片段列表（RAGAS 需要 list[str]）
        contexts = []
        if context_text and context_text != "无相关资料":
            # 按 [source] 标记拆分
            import re
            chunks = re.split(r"\n(?=\[)", context_text)
            contexts = [c.strip() for c in chunks if c.strip()]

        return answer, contexts

    # ── 主评测流程 ──

    async def run(
        self,
        rag_service,
        test_data: list[dict] = None,
        metrics: list[str] = None,
    ) -> dict:
        """运行 RAGAS 评估

        Args:
            rag_service: RagService 实例
            test_data:  测试数据列表（默认从 test_qa.json 加载）
            metrics:    要评估的指标（默认全部: faithfulness, answer_relevancy,
                        context_precision, context_recall）

        Returns:
            评测报告字典
        """
        test_data = test_data or self.load_test_data()
        if not test_data:
            return {"error": "无测试数据", "timestamp": datetime.now().isoformat()}

        metrics = metrics or [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall",
        ]

        logger.info("RAGAS 评估开始: %d 条测试, 指标=%s", len(test_data), metrics)

        # Step 1: 对每条测试数据跑 RAG，收集 answer + contexts
        samples = []
        for i, case in enumerate(test_data):
            question = case["question"]
            reference = case.get("answer", "")        # ground truth
            expected_source = case.get("source", "")   # 元信息（不在 RAGAS 指标中使用）

            logger.info("RAGAS [{}/{}]: {}", i + 1, len(test_data), question[:50])

            answer, contexts = await self._run_rag(rag_service, question)

            samples.append({
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference,
                "expected_source": expected_source,
            })

            self._results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "reference": reference,
            })

        # Step 2: 构建 RAGAS Dataset + 计算指标
        ragas_report = await self._evaluate_ragas(samples, metrics)

        # Step 3: 合并为报告
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_cases": len(test_data),
            "metrics": metrics,
            "scores": ragas_report,
            "details": [
                {
                    "question": r["question"],
                    "answer": r["answer"][:300],
                    "context_count": len(r["contexts"]),
                    "reference": r.get("reference", "")[:200],
                }
                for r in self._results
            ],
        }
        logger.info("RAGAS 评估完成: scores=%s", ragas_report)
        return report

    async def _evaluate_ragas(self, samples: list[dict], metrics: list[str]) -> dict:
        """调用 RAGAS evaluate() 计算指标"""
        try:
            from ragas import evaluate, EvaluationDataset, SingleTurnSample
            from ragas.metrics import (
                Faithfulness, AnswerRelevancy,
                ContextPrecision, ContextRecall,
            )
        except ImportError:
            logger.warning("RAGAS 未安装，降级为基础评分")
            return self._fallback_scores(samples, metrics)

        # 构建 RAGAS dataset
        ragas_samples = []
        for s in samples:
            # 过滤空值
            contexts = s.get("retrieved_contexts") or []
            if not contexts:
                contexts = ["[无检索上下文]"]

            ragas_samples.append(SingleTurnSample(
                user_input=s["user_input"],
                response=s["response"] or "[空回答]",
                retrieved_contexts=contexts,
                reference=s.get("reference", ""),
            ))

        dataset = EvaluationDataset(samples=ragas_samples)

        # 构建指标列表
        llm = self._get_llm()
        metric_map = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }
        selected = [metric_map[m] for m in metrics if m in metric_map]

        if not selected:
            return {"error": "无有效指标"}

        # 运行评估
        try:
            result = evaluate(dataset, metrics=selected)
            # result 是 EvaluationResult，转 dict
            scores = {}
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                for col in df.columns:
                    if col not in ("user_input", "response", "retrieved_contexts", "reference"):
                        scores[col] = round(float(df[col].mean()), 4)
            elif isinstance(result, dict):
                scores = result
            else:
                scores = {m: None for m in metrics}
            return scores
        except Exception as e:
            logger.error("RAGAS evaluate 失败: %s", e)
            return {"error": str(e)}

    def _fallback_scores(self, samples: list[dict], metrics: list[str]) -> dict:
        """RAGAS 未安装时的降级评分（基于检索上下文存在性）"""
        scores = {}
        for m in metrics:
            if m == "context_precision":
                scores[m] = round(
                    sum(1 for s in samples if s.get("retrieved_contexts")) / max(len(samples), 1), 4
                )
            elif m == "context_recall":
                # 有 ground truth 且检索到上下文的占比
                valid = [s for s in samples if s.get("reference")]
                if valid:
                    scores[m] = round(
                        sum(1 for s in valid if s.get("retrieved_contexts")) / len(valid), 4
                    )
                else:
                    scores[m] = None
            else:
                scores[m] = None
        scores["_note"] = "RAGAS 未安装，以上为降级评分"
        return scores

    # ── 报告持久化 ──

    def save_report(self, report: dict, filepath: str = None):
        """保存评测报告"""
        from config import settings as config
        filepath = filepath or os.path.join(
            config.DATA_DIR,
            f"ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("RAGAS 报告已保存: %s", filepath)
