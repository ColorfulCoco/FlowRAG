# -*- coding: utf-8 -*-
"""
SOP 文档去重脚本

按产品名分组后，对同标题精确去重、对相似流程描述模糊去重，仅保留最长文档。

Author: CongCongTian
"""

import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher


def extract_product_and_procedure(filepath: Path) -> tuple:
    """从文档第一行提取产品名和流程描述"""
    with open(filepath, 'r', encoding='utf-8') as f:
        title = f.readline().strip().lstrip('# ').strip()
    
    if ' - ' in title:
        product = title.split(' - ')[0].strip()
        procedure = title.split(' - ', 1)[1].strip()
    else:
        product = title
        procedure = ""
    
    return product, procedure, title


def get_file_content_length(filepath: Path) -> int:
    """获取文件内容长度（字符数）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return len(f.read())


def text_similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度（0~1）"""
    return SequenceMatcher(None, a, b).ratio()


def dedup_procedures(docs_in_group: list, threshold: float = 0.8) -> tuple:
    """对同一产品下的多个文档做流程级去重

    Args:
        docs_in_group: [(filepath, product, procedure, title, content_length), ...]
        threshold: 流程描述相似度阈值

    Returns:
        (keep_list, remove_list)
    """
    if len(docs_in_group) <= 1:
        return docs_in_group, []
    
    title_groups = defaultdict(list)
    for doc in docs_in_group:
        title_groups[doc[3]].append(doc)
    
    after_exact_dedup = []
    exact_removed = []
    for title, group in title_groups.items():
        group.sort(key=lambda x: x[4], reverse=True)
        after_exact_dedup.append(group[0])
        exact_removed.extend(group[1:])
    
    # 按内容长度降序，优先保留内容最丰富的
    after_exact_dedup.sort(key=lambda x: x[4], reverse=True)
    
    keep = []
    similarity_removed = []
    
    for doc in after_exact_dedup:
        procedure = doc[2]
        is_duplicate = False
        for kept_doc in keep:
            sim = text_similarity(procedure, kept_doc[2])
            if sim > threshold:
                is_duplicate = True
                similarity_removed.append(doc)
                break
        
        if not is_duplicate:
            keep.append(doc)
    
    return keep, exact_removed + similarity_removed


def main():
    parser = argparse.ArgumentParser(description='SOP 文档去重')
    parser.add_argument('--sop-dir', default='data/synthetic_dataset/sop_documents',
                        help='SOP 文档目录')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='流程描述相似度阈值（0~1，默认0.8）')
    parser.add_argument('--execute', action='store_true',
                        help='实际执行删除（默认只预览）')
    parser.add_argument('--backup', action='store_true',
                        help='删除前备份到 _removed 目录')
    args = parser.parse_args()
    
    sop_dir = Path(args.sop_dir)
    txt_files = sorted(sop_dir.glob('*.txt'))
    
    print(f"SOP 文档去重: {len(txt_files)} 个文件, 阈值={args.threshold}, "
          f"{'实际执行' if args.execute else '预览模式'}")
    all_docs = []
    for f in txt_files:
        product, procedure, title = extract_product_and_procedure(f)
        content_length = get_file_content_length(f)
        all_docs.append((f, product, procedure, title, content_length))
    
    product_groups = defaultdict(list)
    for doc in all_docs:
        product_groups[doc[1]].append(doc)
    
    print(f"不同产品名: {len(product_groups)}\n")
    
    total_keep = []
    total_remove = []
    sorted_groups = sorted(product_groups.items(), key=lambda x: -len(x[1]))
    
    print(f"{'产品名':<60} | {'原始':>4} | {'保留':>4} | {'删除':>4}")
    print("-" * 85)
    
    for product, docs in sorted_groups:
        keep, remove = dedup_procedures(docs, threshold=args.threshold)
        total_keep.extend(keep)
        total_remove.extend(remove)
        
        if len(remove) > 0:
            display_name = product[:58] + ".." if len(product) > 60 else product
            print(f"{display_name:<60} | {len(docs):>4} | {len(keep):>4} | {len(remove):>4}")
    
    print("-" * 85)
    print(f"{'合计':<60} | {len(all_docs):>4} | {len(total_keep):>4} | {len(total_remove):>4}")
    print()
    
    reduction_pct = (1 - len(total_keep) / len(all_docs)) * 100
    print(f"\n去重: 保留 {len(total_keep)}, 删除 {len(total_remove)}, 减少 {reduction_pct:.1f}%\n")
    
    if not args.execute:
        print(f"预览模式。执行: python scripts/dedup_sop.py --execute --threshold {args.threshold} [--backup]")
        return
    
    backup_dir = None
    if args.backup:
        backup_dir = sop_dir / "_removed"
        backup_dir.mkdir(exist_ok=True)
    
    removed_count = 0
    for doc in total_remove:
        filepath = doc[0]
        txt_path = filepath
        json_path = filepath.with_suffix('.json')
        
        if args.backup and backup_dir:
            if txt_path.exists():
                shutil.move(str(txt_path), str(backup_dir / txt_path.name))
            if json_path.exists():
                shutil.move(str(json_path), str(backup_dir / json_path.name))
        else:
            if txt_path.exists():
                txt_path.unlink()
            if json_path.exists():
                json_path.unlink()
        
        removed_count += 1
    
    print(f"完成！删除 {removed_count} 个文档, 剩余 {len(total_keep)}")


if __name__ == '__main__':
    main()
