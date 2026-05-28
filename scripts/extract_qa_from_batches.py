# -*- coding: utf-8 -*-
"""
BATCH QA 提取工具

从 BATCH-*.json 中提取 qa_pairs，生成 questions.json 和 ground_truth.json。

Author: CongCongTian
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


PROJECT_ROOT = Path(__file__).parent.parent
SOP_DIR = PROJECT_ROOT / "data" / "synthetic_dataset" / "sop_documents"
QUESTIONS_OUTPUT = PROJECT_ROOT / "data" / "questions" / "questions.json"
GROUND_TRUTH_OUTPUT = PROJECT_ROOT / "evaluate" / "data" / "ground_truth.json"


def extract_qa_pairs(sop_dir: Path, limit: int = 0):
    """遍历 BATCH-*.json 提取 qa_pairs

    Args:
        sop_dir: BATCH 文件目录
        limit: 最多处理文件数，0 不限制

    Returns:
        (all_questions, ground_truth_dict, stats)
    """
    batch_files = sorted(sop_dir.glob("BATCH-*.json"))
    if limit > 0:
        batch_files = batch_files[:limit]

    print(f"找到 {len(batch_files)} 个 BATCH 文件 (源目录: {sop_dir})")

    all_questions = []
    ground_truth = {}
    duplicate_questions = []
    files_without_qa = []
    stats = {
        "total_batch_files": len(batch_files),
        "files_with_qa": 0,
        "files_without_qa": 0,
        "total_qa_pairs": 0,
        "duplicate_questions": 0,
        "question_types": {},
        "difficulties": {},
    }

    for batch_file in batch_files:
        try:
            with open(batch_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [错误] 读取 {batch_file.name} 失败: {e}")
            continue

        batch_id = data.get("id", batch_file.stem)
        title = data.get("title", "")
        qa_pairs = data.get("qa_pairs", [])

        if not qa_pairs:
            files_without_qa.append(batch_file.name)
            stats["files_without_qa"] += 1
            continue

        stats["files_with_qa"] += 1

        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            if not question:
                continue

            if question in ground_truth:
                duplicate_questions.append(question)
                stats["duplicate_questions"] += 1
                continue

            stats["total_qa_pairs"] += 1

            q_type = qa.get("question_type", "unknown")
            difficulty = qa.get("difficulty", "unknown")
            stats["question_types"][q_type] = stats["question_types"].get(q_type, 0) + 1
            stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1

            all_questions.append(question)

            # 兼容两种关键词字段名
            keywords = qa.get("keywords", [])
            if not keywords:
                keywords = qa.get("source_keywords", [])

            gt_entry = {
                "source_batch": batch_id,
                "source_title": title,
                "id": qa.get("id", ""),
                "answer": qa.get("answer", ""),
                "question_type": q_type,
                "difficulty": difficulty,
                "relevant_nodes": qa.get("relevant_nodes", []),
                "reasoning_path": qa.get("reasoning_path", []),
                "source_chunks": qa.get("source_chunks", []),
                "expected_hops": qa.get("expected_hops", 0),
                "requires_graph_reasoning": qa.get(
                    "requires_graph_reasoning",
                    qa.get("requires_causal_reasoning", False)
                ),
            }

            if "graph_reasoning_info" in qa:
                gt_entry["graph_reasoning_info"] = qa["graph_reasoning_info"]
            if "subgraph_mermaid" in qa:
                gt_entry["subgraph_mermaid"] = qa["subgraph_mermaid"]
            if "subgraph_quality_score" in qa:
                gt_entry["subgraph_quality_score"] = qa["subgraph_quality_score"]
            if "rag_evaluation" in qa:
                gt_entry["rag_evaluation"] = qa["rag_evaluation"]

            gt_entry["keywords"] = keywords
            gt_entry["source_evidence"] = qa.get("source_evidence", [])

            ground_truth[question] = gt_entry

    print(f"\n提取完成！QA 对: {stats['total_qa_pairs']}, "
          f"含 qa_pairs: {stats['files_with_qa']}/{stats['total_batch_files']}, "
          f"重复跳过: {stats['duplicate_questions']}")

    if stats["question_types"]:
        print(f"\n  问题类型分布:")
        for k, v in sorted(stats["question_types"].items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")

    if stats["difficulties"]:
        print(f"\n  难度分布:")
        for k, v in sorted(stats["difficulties"].items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")

    if files_without_qa:
        print(f"\n  无 qa_pairs 的文件 ({len(files_without_qa)} 个):")
        for name in files_without_qa[:10]:
            print(f"    - {name}")
        if len(files_without_qa) > 10:
            print(f"    ... 还有 {len(files_without_qa) - 10} 个")

    if duplicate_questions:
        print(f"\n  重复问题示例 ({len(duplicate_questions)} 个):")
        for q in duplicate_questions[:5]:
            print(f"    - {q[:80]}...")

    return all_questions, ground_truth, stats


def save_outputs(
    all_questions: list,
    ground_truth: dict,
    stats: dict,
    questions_output: Path,
    gt_output: Path,
    dry_run: bool = False,
):
    """保存 questions.json 和 ground_truth.json"""
    gt_with_meta = {
        "description": f"从 sop_documents 目录下所有 BATCH-*.json 文件的 qa_pairs 中提取，"
                       f"共 {len(all_questions)} 条问答对",
        "total_count": len(all_questions),
        "generated_at": datetime.now().isoformat(),
    }
    gt_with_meta.update(ground_truth)

    if dry_run:
        print(f"\n[预览模式] 不写入文件")
        print(f"  questions.json:    {len(all_questions)} 条")
        print(f"  ground_truth.json: {len(ground_truth)} 条")
        print(f"\n  前 3 个问题:")
        for i, q in enumerate(all_questions[:3], 1):
            print(f"    {i}. {q[:100]}...")
        return

    questions_output.parent.mkdir(parents=True, exist_ok=True)
    gt_output.parent.mkdir(parents=True, exist_ok=True)

    for fp in [questions_output, gt_output]:
        if fp.exists():
            bak = fp.with_suffix(".json.bak")
            fp.rename(bak)
            print(f"\n  已备份: {fp.name} → {bak.name}")

    with open(questions_output, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] saved: {questions_output}")
    print(f"     ({len(all_questions)} questions)")

    with open(gt_output, "w", encoding="utf-8") as f:
        json.dump(gt_with_meta, f, ensure_ascii=False, indent=2)
    print(f"\n  [OK] saved: {gt_output}")
    print(f"     ({len(ground_truth)} QA pairs)")


def main():
    parser = argparse.ArgumentParser(
        description="从 BATCH-*.json 提取 QA 对，生成 questions.json 和 ground_truth.json"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="最多处理的 BATCH 文件数（默认 0 = 全部）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式，不写入文件"
    )
    parser.add_argument(
        "--sop-dir", type=str, default=None,
        help=f"BATCH 文件目录（默认 {SOP_DIR}）"
    )
    parser.add_argument(
        "--questions-output", type=str, default=None,
        help=f"questions.json 输出路径（默认 {QUESTIONS_OUTPUT}）"
    )
    parser.add_argument(
        "--gt-output", type=str, default=None,
        help=f"ground_truth.json 输出路径（默认 {GROUND_TRUTH_OUTPUT}）"
    )
    args = parser.parse_args()

    sop_dir = Path(args.sop_dir) if args.sop_dir else SOP_DIR
    questions_output = Path(args.questions_output) if args.questions_output else QUESTIONS_OUTPUT
    gt_output = Path(args.gt_output) if args.gt_output else GROUND_TRUTH_OUTPUT

    print("--- BATCH QA 提取工具 ---")

    all_questions, ground_truth, stats = extract_qa_pairs(sop_dir, args.limit)

    if not all_questions:
        print("\n[错误] 未提取到任何 QA 对，请检查 BATCH 文件")
        return

    save_outputs(
        all_questions, ground_truth, stats,
        questions_output, gt_output,
        dry_run=args.dry_run,
    )

    print(f"\n完成！后续: python examples/quick_start.py --batch --force-rerun --auto-eval")


if __name__ == "__main__":
    main()


