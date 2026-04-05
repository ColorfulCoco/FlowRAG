# -*- coding: utf-8 -*-
"""
问答对生成器

基于SOP文档和知识图谱生成高质量问答对，用于评估数据集构建。

Author: CongCongTian
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field

from src.config import BUILD_LLM_API_KEY, BUILD_LLM_BASE_URL, BUILD_LLM_MODEL, DATA_DIR, PERSIST_DIRECTORY

@dataclass
class QAPair:
    """QA 对数据结构"""
    id: str                                     # 唯一标识
    question: str                               # 问题
    answer: str                                 # 答案
    question_type: str                          # 问题类型
    difficulty: str                             # 难度: easy/medium/hard
    source_doc: str                             # 来源文档
    
    # 验证相关字段
    graph1_can_answer: bool = False             # 传统图谱能否回答
    graph2_can_answer: bool = True              # 我们的图谱能否回答  
    need_text_knowledge: bool = False           # 是否需要文本中的陈述性知识
    
    # 分析字段
    difference_analysis: str = ""               # 差异分析说明
    related_nodes: List[str] = field(default_factory=list)  # 涉及的节点
    evidence: str = ""                          # 答案依据


@dataclass  
class QADataset:
    """QA 数据集"""
    version: str = "1.0"
    created_at: str = ""
    total_count: int = 0
    type_distribution: Dict[str, int] = field(default_factory=dict)
    qa_pairs: List[QAPair] = field(default_factory=list)


# 提示词模板


QA_GENERATION_PROMPT = """
# Role
你是一位工业文档智能处理领域的评测专家。你需要构建一套高区分度的问答数据集，用于验证一篇关于“基于 Mermaid 作为中间逻辑表示构建逻辑骨架图谱”论文的有效性。

# Input Data
请仔细分析以下三组输入数据：

1. **原始文本**：
{text}

2. **Baseline图谱数据**：
*说明：这是由传统三元组抽取生成的图谱，可能存在跨段落逻辑断裂、节点孤立或上下文属性归属不清的问题。*
{graph1}

3. **Mermaid图谱数据(本方法)**：
*说明：这是通过 Text->Mermaid 生成的结构化代码，具有完整的流程骨架、跳转逻辑和挂载的细节属性。*
{mermaid_code}

# Global Constraints (重要)
为了保证便于后续的自动化评测，生成的问题必须满足以下约束：
1. **答案唯一性**：答案必须是原文中明确存在的**实体名称、数值（带单位）、特定步骤名或状态词**。
2. **拒绝长句**：禁止生成需要用整句或段落回答的问题,尽量不要问开放性的问题。

# Task
请对比上述三组数据，生成 3 类具有不同区分度的问答对（每类 2-3 题），输出为 JSON 格式。

## Category 1: 基础事实类 (验证：All Pass)
*   **出题标准**：寻找在 [原始文本] 中有明确定义，且在 [Baseline图谱] 和 [Mermaid图谱] 中都有对应节点和属性的信息。
*   **目标**：确保所有方法（包括纯向量检索）都能回答。
*   **推荐题型**：询问设备的参数值、型号、物理位置、定义名称。

## Category 2: 局部流程类 (验证：Graph & Ours > Vector)
*   **出题标准**：寻找具有明确“先后顺序”的两个相邻步骤。
*   **区分点**：问题必须依赖“顺序”逻辑，纯文本检索可能会因为词语在文中多次出现而找错段落。
*   **推荐题型**：询问“下一步骤的名称”或“前置条件的名称”。

## Category 3: 核心优势类 (验证：Only Ours > Baseline & Vector) —— **最关键部分**
*   **出题标准（满足其一即可，务必挖掘 Mermaid 的结构优势）**：
    1.  **逻辑断裂修复**：请找到在 [Mermaid图谱] 中有连线（如跨越长距离的跳转、条件分支），但在 [Baseline图谱] 中这两个节点是**断开/无连接**的情况。
        *   *提问方式*：针对这一逻辑跳转的后果提问，询问系统下一步会执行的操作名称或进入的状态名称。。
    2.  **结构化上下文**：请找到在 [Mermaid图谱] 中明确挂载在某个 `subgraph` 或节点下的数值（如温度），但在 [Baseline图谱] 中该数值丢失了父节点信息（不知道是哪个阶段的温度）。
        *   *提问方式*：询问特定阶段下的属性值。
*   **目标**：证明你的方法解决了传统图谱的“逻辑断裂”和“信息混淆”问题。
*   **示例**：“如果初始化检测失败，系统将跳转到的模式名称是什么？”（答案：安全锁定模式）

# Output Format (JSON)
请严格遵守以下 JSON 结构：

```json
[
  {
    "category": "Cat_1_All_Pass",
    "question": "...",
    "answer": "...",
    "analysis": "该信息在原文第X段，Baseline三元组(A, 属性, B)完整，Mermaid节点A包含属性B。"
  },
  {
    "category": "Cat_2_Graph_Better",
    "question": "...",
    "answer": "...",
    "analysis": "Baseline和Mermaid均有 A->B 连线。纯向量检索可能因关键词干扰检索到错误的后续步骤。"
  },
  {
    "category": "Cat_3_Only_Ours",
    "question": "...",
    "answer": "...",
    "analysis": {
        "reason": "逻辑断裂 / 上下文丢失",
        "evidence": "Mermaid中存在 `A -->|条件| B` 的直接跳转；而Baseline数据中节点A和B无直接连边，需跨越多个三元组且无法连通。"
    }
  }
]
"""



class QAGenerator:
    """QA 数据集生成器"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        mixed_text_folder: Path,
        mermaid_folder: Path,
        output_file: Path
    ):
        """
        初始化 QA 生成器
        
        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            model_name: 模型名称
            mixed_text_folder: 混合文本目录（扩充后的文档）
            mermaid_folder: Mermaid 流程图目录
            output_file: 输出文件路径
        """
        from src.utils.openai_client import create_openai_client
        self.client = create_openai_client(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.mixed_text_folder = mixed_text_folder
        self.mermaid_folder = mermaid_folder
        self.output_file = output_file
        
        # 统计信息
        self.stats = {
            "total_generated": 0,
            "by_type": {},
            "failed": []
        }
    
    def _load_file(self, file_path: Path) -> Optional[str]:
        """加载文件内容"""
        try:
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            print(f"[警告] 无法读取文件 {file_path}: {e}")
        return None
    
    def _call_llm(self, messages: list, retries: int = None) -> Optional[str]:
        """调用大模型"""
        from src.config import API_RETRY_INTERVAL_SECONDS, API_RETRY_MAX_ATTEMPTS
        _max = API_RETRY_MAX_ATTEMPTS if retries is None else retries
        _interval = API_RETRY_INTERVAL_SECONDS
        for attempt in range(_max + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=8192
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < _max:
                    print(f"  [重试] 第{attempt+1}次失败，{_interval}秒后重试: {e}")
                    time.sleep(_interval)
                else:
                    print(f"  [错误] API调用失败: {e}")
                    return None
    

    
    def _parse_qa_response(self, response: str, source_doc: str) -> List[QAPair]:
        """解析大模型返回的 QA JSON"""
        qa_pairs = []
        
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start == -1 or json_end == 0:
                print(f"  [警告] 无法找到 JSON 数组")
                return qa_pairs
            
            json_str = response[json_start:json_end]
            qa_list = json.loads(json_str)
            
            category_to_type = {
                "Cat_1_All_Pass": "factual",
                "Cat_2_Graph_Better": "procedural",
                "Cat_3_Only_Ours": "structural",
            }
            for i, qa in enumerate(qa_list):
                category = qa.get("category", "")
                q_type = category_to_type.get(category, qa.get("question_type", "unknown"))
                qa_id = f"{source_doc}_{q_type}_{i+1}"

                analysis = qa.get("analysis", "")
                if isinstance(analysis, dict):
                    diff_analysis = analysis.get("reason", "")
                    evidence = analysis.get("evidence", "")
                else:
                    diff_analysis = str(analysis)
                    evidence = ""

                qa_pair = QAPair(
                    id=qa_id,
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    question_type=q_type,
                    difficulty=qa.get("difficulty", "medium"),
                    source_doc=source_doc,
                    graph1_can_answer=(category == "Cat_1_All_Pass" or category == "Cat_2_Graph_Better"),
                    graph2_can_answer=True,
                    need_text_knowledge=(category == "Cat_1_All_Pass"),
                    difference_analysis=diff_analysis,
                    related_nodes=qa.get("related_nodes", []),
                    evidence=evidence
                )
                qa_pairs.append(qa_pair)
                
                if q_type not in self.stats["by_type"]:
                    self.stats["by_type"][q_type] = 0
                self.stats["by_type"][q_type] += 1
                
        except json.JSONDecodeError as e:
            print(f"  [错误] JSON 解析失败: {e}")
        except Exception as e:
            print(f"  [错误] 解析响应失败: {e}")
        
        return qa_pairs
    
    def generate_for_document(self, tg_id: str, count: int = 10) -> List[QAPair]:
        print(f"\n处理文档: {tg_id}")
        
        mixed_text_path = self.mixed_text_folder / f"{tg_id}-new.txt"
        if not mixed_text_path.exists():
            mixed_text_path = self.mixed_text_folder / f"{tg_id}.txt"
        
        mixed_text = self._load_file(mixed_text_path)
        if not mixed_text:
            print(f"  [错误] 无法加载混合文本: {mixed_text_path}")
            self.stats["failed"].append(tg_id)
            return []
        
        mermaid_path = self.mermaid_folder / f"{tg_id}_flowchart.mmd"
        mermaid_code = self._load_file(mermaid_path)
        
        graph1 = ''
        
        prompt = QA_GENERATION_PROMPT.format(
            text=mixed_text,
            graph1=json.dumps(graph1, ensure_ascii=False, indent=2),
            mermaid_code=mermaid_code or "(无 Mermaid 流程图)"
        )
        
        response = self._call_llm([
            {"role": "user", "content": prompt}
        ])
        
        if not response:
            self.stats["failed"].append(tg_id)
            return []
        
        qa_pairs = self._parse_qa_response(response, tg_id)
        self.stats["total_generated"] += len(qa_pairs)
        
        print(f"  生成 {len(qa_pairs)} 个问答对")
        
        return qa_pairs
    
    def generate_all(self, count_per_doc: int = 10, max_documents: Optional[int] = None) -> QADataset:
        """
        为所有文档生成 QA 数据集
        
        Args:
            count_per_doc: 每个文档生成的问题数量
            max_documents: 最大处理文档数
        
        Returns:
            完整的 QA 数据集
        """
        # 获取所有扩充后的文档
        expanded_files = sorted(self.mixed_text_folder.glob("TG-*-new.txt"))
        if not expanded_files:
            # 尝试不带 -new 后缀
            expanded_files = sorted(self.mixed_text_folder.glob("TG-*.txt"))
        
        if max_documents:
            expanded_files = expanded_files[:max_documents]
        
        if not expanded_files:
            print(f"[错误] 在 {self.mixed_text_folder} 中未找到文档")
            return QADataset()
        
        # 提取 TG ID
        tg_ids = []
        for f in expanded_files:
            name = f.stem
            if name.endswith("-new"):
                name = name[:-4]
            tg_ids.append(name)
        tg_ids = list(set(tg_ids))  # 去重
        tg_ids.sort()
        
        print(f"找到 {len(tg_ids)} 个 TG 文档, 每个生成 {count_per_doc} 个问题")
        
        all_qa_pairs = []
        
        for tg_id in tg_ids:
            qa_pairs = self.generate_for_document(tg_id, count_per_doc)
            all_qa_pairs.extend(qa_pairs)
            time.sleep(1)  # API 调用间隔
        
        # 构建数据集
        dataset = QADataset(
            version="1.0",
            created_at=datetime.now().isoformat(),
            total_count=len(all_qa_pairs),
            type_distribution=self.stats["by_type"].copy(),
            qa_pairs=all_qa_pairs
        )
        
        return dataset
    
    def save_dataset(self, dataset: QADataset):
        """保存数据集到文件"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化的字典
        data = {
            "version": dataset.version,
            "created_at": dataset.created_at,
            "total_count": dataset.total_count,
            "type_distribution": dataset.type_distribution,
            "qa_pairs": [asdict(qa) for qa in dataset.qa_pairs]
        }
        
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n数据集已保存: {self.output_file} ({dataset.total_count} 个问题)")



def create_generator(
    output_file: Optional[Path] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None
) -> QAGenerator:
    """创建 QA 生成器"""
    return QAGenerator(
        api_key=api_key or BUILD_LLM_API_KEY,
        base_url=base_url or BUILD_LLM_BASE_URL,
        model_name=model_name or BUILD_LLM_MODEL,
        mixed_text_folder=DATA_DIR / "expand_folder",
        mermaid_folder=PERSIST_DIRECTORY / "mermaid_visualization",
        output_file=output_file or DATA_DIR / "qa_dataset.json"
    )



def main():
    parser = argparse.ArgumentParser(
        description="FlowRAG QA 数据集生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="为所有文档生成 QA"
    )
    
    parser.add_argument(
        "--doc", "-d",
        type=str,
        default=None,
        help="指定单个文档 ID（如 TG-01）"
    )
    
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=10,
        help="每个文档生成的问题数量（默认 10）"
    )
    
    parser.add_argument(
        "--max-docs", "-m",
        type=int,
        default=None,
        help="最大处理文档数量"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    # 创建生成器
    output_file = Path(args.output) if args.output else None
    generator = create_generator(output_file=output_file)
    
    # 执行生成
    if args.doc:
        # 单文档模式
        print(f"为文档 {args.doc} 生成 QA")
        qa_pairs = generator.generate_for_document(args.doc, args.count)
        
        dataset = QADataset(
            version="1.0",
            created_at=datetime.now().isoformat(),
            total_count=len(qa_pairs),
            type_distribution=generator.stats["by_type"].copy(),
            qa_pairs=qa_pairs
        )
        generator.save_dataset(dataset)
    
    elif args.all:
        # 全量生成模式
        dataset = generator.generate_all(args.count, args.max_docs)
        generator.save_dataset(dataset)
    
    else:
        print("请指定生成模式: --all 或 --doc TG-01 (详见 --help)")


if __name__ == "__main__":
    main()
