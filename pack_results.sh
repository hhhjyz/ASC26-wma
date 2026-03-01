#!/usr/bin/bash
# filepath: pack_results.sh

# 定义源目录和目标目录
BASE_DIR="/pool/nvme/asc26/World_Model/team4/unifolm-world-model-action"
OUTPUT_DIR="results/results_opt_v2"

# 定义要处理的任务目录
TASKS=(
    "unitree_g1_pack_camera"
    "unitree_z1_dual_arm_cleanup_pencils"
    "unitree_z1_dual_arm_stackbox"
    "unitree_z1_dual_arm_stackbox_v2"
    "unitree_z1_stackbox"
)

# 清理并创建输出目录
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "开始打包结果文件..."

# 遍历每个任务目录
for task in "${TASKS[@]}"; do
    task_path="$BASE_DIR/$task"
    
    if [ ! -d "$task_path" ]; then
        echo "警告: 目录不存在 $task_path"
        continue
    fi
    
    echo "处理任务: $task"
    
    # 创建任务输出目录
    mkdir -p "$OUTPUT_DIR/$task"
    
    # 遍历每个case目录
    for case_dir in "$task_path"/case*; do
        if [ ! -d "$case_dir" ]; then
            continue
        fi
        
        case_name=$(basename "$case_dir")
        echo "  处理 $case_name"
        
        # 创建case输出目录
        mkdir -p "$OUTPUT_DIR/$task/$case_name"
        
        # 复制 output.log
        if [ -f "$case_dir/output.log" ]; then
            cp "$case_dir/output.log" "$OUTPUT_DIR/$task/$case_name/"
            echo "    ✓ 复制 output.log"
        else
            echo "    ✗ 未找到 output.log"
        fi
        
        # 复制 mp4 文件 (从 output/inference 目录)
        if [ -d "$case_dir/output/inference" ]; then
            mp4_file=$(find "$case_dir/output/inference" -maxdepth 1 -name "*.mp4" -type f | head -n 1)
            if [ -n "$mp4_file" ]; then
                mp4_name=$(basename "$mp4_file")
                cp "$mp4_file" "$OUTPUT_DIR/$task/$case_name/"
                echo "    ✓ 复制 $mp4_name"
            else
                echo "    ✗ 未找到 mp4 文件"
            fi
        else
            echo "    ✗ output/inference 目录不存在"
        fi
    done
done

# 生成 summary.json
echo "生成 summary.json..."
python3 << 'EOF'
import json
import os
from pathlib import Path
from datetime import datetime

output_dir = "results"
summary = {
    "generated_at": datetime.now().isoformat(),
    "tasks": {}
}

for task_dir in sorted(Path(output_dir).glob("unitree_*")):
    task_name = task_dir.name
    summary["tasks"][task_name] = {
        "cases": {}
    }
    
    for case_dir in sorted(task_dir.glob("case*")):
        case_name = case_dir.name
        files = {
            "output_log": (case_dir / "output.log").exists(),
            "video": None
        }
        
        # 查找 mp4 文件
        mp4_files = list(case_dir.glob("*.mp4"))
        if mp4_files:
            files["video"] = mp4_files[0].name
        
        summary["tasks"][task_name]["cases"][case_name] = files

with open(os.path.join(output_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("summary.json 已生成")
EOF

echo ""
echo "打包完成! 结果保存在 $OUTPUT_DIR 目录"
echo ""
echo "目录结构:"
tree -L 3 "$OUTPUT_DIR" 2>/dev/null || find "$OUTPUT_DIR" -type f

# 显示统计信息
echo ""
echo "统计信息:"
echo "- 总任务数: ${#TASKS[@]}"
echo "- output.log 文件: $(find "$OUTPUT_DIR" -name "output.log" | wc -l)"
echo "- mp4 文件: $(find "$OUTPUT_DIR" -name "*.mp4" | wc -l)"