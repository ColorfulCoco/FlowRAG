# -*- coding: utf-8 -*-
"""
SOP 文档重新编号脚本

将 BATCH-XXXX.txt/.json 从 BATCH-0001 起连续编号，同步更新 JSON 内部 id 字段。

Author: CongCongTian
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='SOP 文档重新编号')
    parser.add_argument('--sop-dir', default='data/synthetic_dataset/sop_documents',
                        help='SOP 文档目录')
    parser.add_argument('--start', type=int, default=1,
                        help='起始编号（默认1）')
    parser.add_argument('--execute', action='store_true',
                        help='实际执行重命名（默认只预览）')
    args = parser.parse_args()

    sop_dir = Path(args.sop_dir)

    txt_files = sorted(sop_dir.glob('BATCH-*.txt'))

    print(f"SOP 重新编号: {len(txt_files)} 个文件, {'实际执行' if args.execute else '预览模式'}\n")

    rename_plan = []
    for idx, old_txt in enumerate(txt_files, start=args.start):
        new_id = f"BATCH-{idx:04d}"
        old_stem = old_txt.stem
        old_json = old_txt.with_suffix('.json')

        new_txt = sop_dir / f"{new_id}.txt"
        new_json = sop_dir / f"{new_id}.json"

        rename_plan.append({
            "old_id": old_stem,
            "new_id": new_id,
            "old_txt": old_txt,
            "old_json": old_json,
            "new_txt": new_txt,
            "new_json": new_json,
            "need_rename": (old_stem != new_id),
        })

    need_rename_count = sum(1 for r in rename_plan if r["need_rename"])
    skip_count = len(rename_plan) - need_rename_count
    print(f"需要重命名: {need_rename_count}")
    print(f"无需改动: {skip_count}")
    print()

    changed = [r for r in rename_plan if r["need_rename"]]
    print(f"重命名示例（前10个）:")
    for r in changed[:10]:
        print(f"  {r['old_id']} → {r['new_id']}")
    if len(changed) > 10:
        print(f"  ... 还有 {len(changed) - 10} 个")
    print()

    if not args.execute:
        print(f"预览模式。执行: python scripts/renumber_sop.py --execute")
        return

    # 先改临时名再改目标名，避免 A->B 但 B 已存在的冲突
    print("Step 1/3: 重命名为临时文件名...")
    temp_mapping = []  # (temp_txt, temp_json, plan_item)
    for r in rename_plan:
        temp_txt = sop_dir / f"_TEMP_{r['new_id']}.txt"
        temp_json = sop_dir / f"_TEMP_{r['new_id']}.json"

        if r["old_txt"].exists():
            r["old_txt"].rename(temp_txt)
        if r["old_json"].exists():
            r["old_json"].rename(temp_json)

        temp_mapping.append((temp_txt, temp_json, r))

    print("Step 2/3: 重命名为目标文件名...")
    for temp_txt, temp_json, r in temp_mapping:
        if temp_txt.exists():
            temp_txt.rename(r["new_txt"])
        if temp_json.exists():
            temp_json.rename(r["new_json"])

    print("Step 3/3: 更新 JSON 内部的 id 字段...")
    updated_count = 0
    for r in rename_plan:
        json_path = r["new_json"]
        if not json_path.exists():
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            old_id_value = data.get("id", "")
            data["id"] = r["new_id"]

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            if old_id_value != r["new_id"]:
                updated_count += 1
        except Exception as e:
            print(f"  [警告] 更新 {json_path.name} 失败: {e}")

    print(f"\n完成！重命名 {need_rename_count} 对, 更新 JSON id {updated_count} 个, "
          f"范围 BATCH-0001 ~ BATCH-{len(rename_plan):04d}")


if __name__ == '__main__':
    main()
