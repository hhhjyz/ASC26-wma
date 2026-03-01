#!/usr/bin/bash
# filepath: /pool/nvme/asc26/World_Model/team4/unifolm-world-model-action/cal_psnr.sh

# 可配置的结果目录
RESULTS_DIR=${1:-"results/results_opt_v2"}
WORKSPACE_DIR="/pool/nvme/asc26/World_Model/team4/unifolm-world-model-action"

# 优化前的时间数据 (Baseline时间，从results_baseline_v1获取)
declare -A TIME_BEFORE_OPTIM=(
    ["unitree_g1_pack_camera_case1"]=933.181
    ["unitree_g1_pack_camera_case2"]=930.948
    ["unitree_g1_pack_camera_case3"]=932.440
    ["unitree_g1_pack_camera_case4"]=928.839
    ["unitree_z1_stackbox_case1"]=1003.803
    ["unitree_z1_stackbox_case2"]=1006.982
    ["unitree_z1_stackbox_case3"]=1004.082
    ["unitree_z1_stackbox_case4"]=1006.105
    ["unitree_z1_dual_arm_stackbox_case1"]=686.456
    ["unitree_z1_dual_arm_stackbox_case2"]=642.819
    ["unitree_z1_dual_arm_stackbox_case3"]=639.671
    ["unitree_z1_dual_arm_stackbox_case4"]=639.254
    ["unitree_z1_dual_arm_stackbox_v2_case1"]=943.426
    ["unitree_z1_dual_arm_stackbox_v2_case2"]=940.256
    ["unitree_z1_dual_arm_stackbox_v2_case3"]=945.977
    ["unitree_z1_dual_arm_stackbox_v2_case4"]=945.358
    ["unitree_z1_dual_arm_cleanup_pencils_case1"]=704.362
    ["unitree_z1_dual_arm_cleanup_pencils_case2"]=707.568
    ["unitree_z1_dual_arm_cleanup_pencils_case3"]=708.790
    ["unitree_z1_dual_arm_cleanup_pencils_case4"]=740.214
)

current_time="out"

echo "================================================"
echo "计算 PSNR 分数"
echo "结果目录: ${RESULTS_DIR}"
echo "时间戳: ${current_time}"
echo "================================================"
echo ""

# 定义任务配置
declare -A TASKS=(
    ["unitree_g1_pack_camera"]="case1:0_full_fs6 case2:50_full_fs6 case3:100_full_fs6 case4:200_full_fs6"
    ["unitree_z1_stackbox"]="case1:5_full_fs4 case2:15_full_fs4 case3:25_full_fs4 case4:35_full_fs4"
    ["unitree_z1_dual_arm_stackbox"]="case1:5_full_fs4 case2:15_full_fs4 case3:25_full_fs4 case4:35_full_fs4"
    ["unitree_z1_dual_arm_stackbox_v2"]="case1:5_full_fs4 case2:15_full_fs4 case3:25_full_fs4 case4:35_full_fs4"
    ["unitree_z1_dual_arm_cleanup_pencils"]="case1:0_full_fs4 case2:50_full_fs4 case3:100_full_fs4 case4:200_full_fs4"
)

# 初始化结果文件 (JSON数组格式)
SUMMARY_FILE="${RESULTS_DIR}/psnr_summary_${current_time}.json"
mkdir -p "${RESULTS_DIR}/psnr"

# 临时文件存储所有结果
TEMP_RESULTS=""
first_entry=true

# 遍历所有任务
for task in "${!TASKS[@]}"; do
    task_dir="${WORKSPACE_DIR}/${RESULTS_DIR}/${task}"

    if [ ! -d "${task_dir}" ]; then
        echo "警告: 目录不存在 ${task_dir}"
        continue
    fi
    
    echo "处理任务: ${task}"
    
    cases=(${TASKS[$task]})
    
    for case_config in ${cases[@]}; do
        case_name="${case_config%%:*}"
        video_name="${case_config##*:}"
        
        # 提取 case_id (从 case1 -> 1)
        case_id="${case_name#case}"
        
        gt_video="${WORKSPACE_DIR}/${task}/${case_name}/${task}_${case_name}.mp4"
        pred_video="${task_dir}/${case_name}/${video_name}.mp4"
        output_file="${RESULTS_DIR}/psnr/${task}_${case_name}_psnr.json"
        time_log="${task_dir}/${case_name}/time.log"
        
        echo "  处理 ${case_name}..."
        
        if [ ! -f "${gt_video}" ]; then
            echo "    警告: GT视频不存在 ${gt_video}"
            continue
        fi
        
        if [ ! -f "${pred_video}" ]; then
            echo "    警告: 预测视频不存在 ${pred_video}"
            continue
        fi
        
        # 计算PSNR
        python3 psnr_score_for_challenge.py \
            --gt_video "${gt_video}" \
            --pred_video "${pred_video}" \
            --output_file "${output_file}"
        
        # 获取PSNR值
        psnr_value="null"
        if [ -f "${output_file}" ]; then
            psnr_value=$(python3 -c "import json; data=json.load(open('${output_file}')); print(data.get('psnr', 'null'))")
        fi
        
        # 获取优化后的时间 (从output.log读取real时间，格式如: real 15m28.308s)
        output_log="${task_dir}/${case_name}/output.log"
        time_after_optim="null"
        if [ -f "${output_log}" ]; then
            # 提取 "real XXmYY.ZZZs" 格式的时间并转换为秒
            real_time=$(grep "^real" "${output_log}" | tail -1)
            if [ -n "${real_time}" ]; then
                time_after_optim=$(echo "${real_time}" | python3 -c "
import sys
import re
line = sys.stdin.read().strip()
match = re.search(r'real\s+(\d+)m([\d.]+)s', line)
if match:
    minutes = int(match.group(1))
    seconds = float(match.group(2))
    total_seconds = minutes * 60 + seconds
    print(f'{total_seconds:.3f}')
else:
    print('null')
")
            fi
        fi
        
        # 获取优化前的时间
        time_key="${task}_${case_name}"
        time_before_optim="${TIME_BEFORE_OPTIM[$time_key]:-null}"
        
        # 添加到结果
        if [ "$first_entry" = false ]; then
            TEMP_RESULTS="${TEMP_RESULTS},"
        fi
        first_entry=false
        
        TEMP_RESULTS="${TEMP_RESULTS}
{\"scenario\": \"${task}\", \"case_id\": ${case_id}, \"time_before_optim\": ${time_before_optim}, \"time_after_optim\": ${time_after_optim}, \"psnr\": ${psnr_value}}"
        
        echo "    ✓ PSNR: ${psnr_value}, Time: ${time_after_optim}s"
    done
    
    echo ""
done

# 写入最终JSON文件
echo "[${TEMP_RESULTS}
]" > "${SUMMARY_FILE}"

# 格式化JSON文件
python3 << EOF
import json

with open("${SUMMARY_FILE}", "r") as f:
    data = json.load(f)

# 按scenario和case_id排序
data.sort(key=lambda x: (x["scenario"], x["case_id"]))

with open("${SUMMARY_FILE}", "w") as f:
    json.dump(data, f, indent=2)

# 打印统计信息
valid_psnr = [d["psnr"] for d in data if d["psnr"] is not None]
valid_time_before = [d["time_before_optim"] for d in data if d["time_before_optim"] is not None]
valid_time_after = [d["time_after_optim"] for d in data if d["time_after_optim"] is not None]

print("\n" + "="*50)
print("统计信息:")
print("="*50)
print(f"总样本数: {len(data)}")

if valid_psnr:
    print(f"\nPSNR:")
    print(f"  平均: {sum(valid_psnr)/len(valid_psnr):.4f} dB")
    print(f"  最大: {max(valid_psnr):.4f} dB")
    print(f"  最小: {min(valid_psnr):.4f} dB")

if valid_time_before and valid_time_after:
    avg_before = sum(valid_time_before) / len(valid_time_before)
    avg_after = sum(valid_time_after) / len(valid_time_after)
    speedup = avg_before / avg_after if avg_after > 0 else 0
    print(f"\n时间:")
    print(f"  优化前平均: {avg_before:.3f}s")
    print(f"  优化后平均: {avg_after:.3f}s")
    print(f"  加速比: {speedup:.2f}x")

print("="*50)
EOF

echo ""
echo "================================================"
echo "所有PSNR计算完成!"
echo "总结果文件: ${SUMMARY_FILE}"
echo "================================================"