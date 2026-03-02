#!/usr/bin/bash
# filepath: /pool/nvme/asc26/World_Model/team4/unifolm-world-model-action/iter_case.sh

export CUDA_VISIBLE_DEVICES=1
export ATTN_IMPL_TYPE=xformers
export WMA_ENABLE_TB=0
export WMA_SAVE_INTERMEDIATE=0
export WMA_SKIP_INIT=1
export WMA_USE_FP16=1  # 启用 FP16 推理以提升性能，设置为 0 可禁用以提高精度
export WMA_PROFILING=0  # 性能分析开关 (1=启用会增加 cuda.synchronize() 开销，正式测试时关闭)
export WMA_OPENCLIP_LOAD_PRETRAINED=0
export WMA_CUDNN_BENCHMARK=0  # 启用 cuDNN benchmark，针对固定输入尺寸加速
export WMA_USE_TF32=0  # 启用 TF32，A100/A6000 等 Ampere+ GPU 显著加速

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 定义所有 case 的配置
# 格式: "任务目录/case:视频名称"
declare -a CASES=(
    "unitree_g1_pack_camera/case1:0_full_fs6"
    # "unitree_g1_pack_camera/case2:50_full_fs6"
    # "unitree_g1_pack_camera/case3:100_full_fs6"
    # "unitree_g1_pack_camera/case4:200_full_fs6"
    # "unitree_z1_stackbox/case1:5_full_fs4"
    # "unitree_z1_stackbox/case2:15_full_fs4"
    # "unitree_z1_stackbox/case3:25_full_fs4"
    # "unitree_z1_stackbox/case4:35_full_fs4"
    # "unitree_z1_dual_arm_stackbox/case1:5_full_fs4"
    # "unitree_z1_dual_arm_stackbox/case2:15_full_fs4"
    # "unitree_z1_dual_arm_stackbox/case3:25_full_fs4"
    # "unitree_z1_dual_arm_stackbox/case4:35_full_fs4"
    "unitree_z1_dual_arm_stackbox_v2/case1:5_full_fs4"
    # "unitree_z1_dual_arm_stackbox_v2/case2:15_full_fs4"
    # "unitree_z1_dual_arm_stackbox_v2/case3:25_full_fs4"
    # "unitree_z1_dual_arm_stackbox_v2/case4:35_full_fs4"
    # "unitree_z1_dual_arm_cleanup_pencils/case1:0_full_fs4"
    # "unitree_z1_dual_arm_cleanup_pencils/case2:50_full_fs4"
    # "unitree_z1_dual_arm_cleanup_pencils/case3:100_full_fs4"
    # "unitree_z1_dual_arm_cleanup_pencils/case4:200_full_fs4"
)

# 结果保存目录
PSNR_RESULTS_DIR="results/psnr_results"
mkdir -p "${PSNR_RESULTS_DIR}"

# 函数: 计算 PSNR
calculate_psnr() {
    local case_dir=$1
    local video_name=$2
    local task_name=$(echo "$case_dir" | cut -d'/' -f1)
    local case_name=$(echo "$case_dir" | cut -d'/' -f2)
    
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}计算 PSNR: ${case_dir}${NC}"
    echo -e "${BLUE}========================================${NC}\n"
    
    # 定义视频路径
    local gt_video="${case_dir}/${task_name}_${case_name}.mp4"
    local pred_video="${case_dir}/output/inference/${video_name}.mp4"
    local output_file="${PSNR_RESULTS_DIR}/${task_name}_${case_name}_psnr.json"
    
    # 检查 GT 视频是否存在
    if [ ! -f "$gt_video" ]; then
        echo -e "${RED}警告: GT视频不存在 - ${gt_video}${NC}"
        return 1
    fi
    
    # 检查预测视频是否存在
    if [ ! -f "$pred_video" ]; then
        echo -e "${RED}警告: 预测视频不存在 - ${pred_video}${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}GT视频: ${gt_video}${NC}"
    echo -e "${YELLOW}预测视频: ${pred_video}${NC}"
    echo -e "${YELLOW}输出文件: ${output_file}${NC}\n"
    
    # 计算 PSNR
    python3 psnr_score_for_challenge.py \
        --gt_video "$gt_video" \
        --pred_video "$pred_video" \
        --output_file "$output_file"
    
    local psnr_exit_code=$?
    
    if [ $psnr_exit_code -eq 0 ] && [ -f "$output_file" ]; then
        # 读取并显示 PSNR 值
        local psnr_value=$(python3 -c "import json; data=json.load(open('${output_file}')); print(data.get('psnr', 'N/A'))" 2>/dev/null)
        echo -e "\n${GREEN}✓ PSNR 计算成功: ${psnr_value} dB${NC}\n"
        return 0
    else
        echo -e "\n${RED}✗ PSNR 计算失败${NC}\n"
        return 1
    fi
}

# 函数: 运行单个 case
run_case() {
    local case_config=$1
    local case_dir="${case_config%%:*}"
    local video_name="${case_config##*:}"
    local script_path="${case_dir}/run_world_model_interaction.sh"
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}开始运行: ${case_dir}${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    
    # 检查脚本是否存在
    if [ ! -f "$script_path" ]; then
        echo -e "${RED}错误: 脚本不存在 - ${script_path}${NC}"
        return 1
    fi
    
    # 检查数据目录是否存在
    local data_dir="${case_dir}/world_model_interaction_prompts"
    if [ ! -d "$data_dir" ]; then
        echo -e "${RED}错误: 数据目录不存在 - ${data_dir}${NC}"
        return 1
    fi
    
    # 运行脚本
    echo -e "${YELLOW}执行脚本: ${script_path}${NC}\n"
    bash "$script_path"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo -e "\n${GREEN}✓ ${case_dir} 执行成功${NC}\n"
        
        # 计算 PSNR
        calculate_psnr "$case_dir" "$video_name"
        local psnr_code=$?
        
        # 返回综合状态
        return $psnr_code
    else
        echo -e "\n${RED}✗ ${case_dir} 执行失败 (退出码: ${exit_code})${NC}\n"
        return $exit_code
    fi
}

# 主函数
main() {
    local total=${#CASES[@]}
    local success=0
    local failed=0
    declare -a failed_cases
    
    # 创建总结果文件
    local timestamp=$(date "+%Y%m%d-%H%M%S")
    local summary_file="${PSNR_RESULTS_DIR}/psnr_summary_${timestamp}.json"
    
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}开始批量执行 World Model Interaction${NC}"
    echo -e "${GREEN}总共 ${total} 个 case${NC}"
    echo -e "${GREEN}PSNR 结果目录: ${PSNR_RESULTS_DIR}${NC}"
    echo -e "${GREEN}================================================${NC}\n"
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 初始化 summary JSON
    echo "{" > "$summary_file"
    echo "  \"timestamp\": \"${timestamp}\"," >> "$summary_file"
    echo "  \"results\": [" >> "$summary_file"
    
    local first=true
    
    # 遍历所有 case
    for case_config in "${CASES[@]}"; do
        local case_dir="${case_config%%:*}"
        local video_name="${case_config##*:}"
        local task_name=$(echo "$case_dir" | cut -d'/' -f1)
        local case_name=$(echo "$case_dir" | cut -d'/' -f2)
        
        # 运行 case 和计算 PSNR
        run_case "$case_config"
        
        if [ $? -eq 0 ]; then
            ((success++))
            
            # 读取 PSNR 结果并添加到 summary
            local psnr_file="${PSNR_RESULTS_DIR}/${task_name}_${case_name}_psnr.json"
            if [ -f "$psnr_file" ]; then
                if [ "$first" = false ]; then
                    echo "," >> "$summary_file"
                fi
                first=false
                
                echo "    {" >> "$summary_file"
                echo "      \"task\": \"${task_name}\"," >> "$summary_file"
                echo "      \"case\": \"${case_name}\"," >> "$summary_file"
                echo "      \"case_dir\": \"${case_dir}\"," >> "$summary_file"
                
                # 读取 PSNR 值
                local psnr_value=$(python3 -c "import json; data=json.load(open('${psnr_file}')); print(data.get('psnr', 'null'))" 2>/dev/null)
                echo "      \"psnr\": ${psnr_value}" >> "$summary_file"
                echo "    }" >> "$summary_file"
            fi
        else
            ((failed++))
            failed_cases+=("$case_dir")
        fi
        
        # 添加分隔符
        echo -e "\n${YELLOW}----------------------------------------${NC}\n"
    done
    
    # 完成 summary JSON
    echo "  ]" >> "$summary_file"
    echo "}" >> "$summary_file"
    
    # 计算总耗时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(((duration % 3600) / 60))
    seconds=$((duration % 60))
    
    # 计算统计信息
    echo -e "${BLUE}计算统计信息...${NC}"
    python3 << EOF
import json

with open("${summary_file}", "r") as f:
    data = json.load(f)

all_psnr = []
task_stats = {}

for item in data["results"]:
    if "psnr" in item and item["psnr"] is not None:
        psnr = item["psnr"]
        all_psnr.append(psnr)
        
        task = item["task"]
        if task not in task_stats:
            task_stats[task] = []
        task_stats[task].append(psnr)

# 计算统计信息
if all_psnr:
    stats = {
        "total_samples": len(all_psnr),
        "average_psnr": round(sum(all_psnr) / len(all_psnr), 4),
        "max_psnr": round(max(all_psnr), 4),
        "min_psnr": round(min(all_psnr), 4),
        "by_task": {}
    }
    
    for task, psnr_list in task_stats.items():
        stats["by_task"][task] = {
            "count": len(psnr_list),
            "average": round(sum(psnr_list) / len(psnr_list), 4),
            "max": round(max(psnr_list), 4),
            "min": round(min(psnr_list), 4)
        }
    
    data["statistics"] = stats
    
    with open("${summary_file}", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"PSNR 统计信息:")
    print(f"{'='*60}")
    print(f"总样本数: {stats['total_samples']}")
    print(f"平均PSNR: {stats['average_psnr']:.4f} dB")
    print(f"最大PSNR: {stats['max_psnr']:.4f} dB")
    print(f"最小PSNR: {stats['min_psnr']:.4f} dB")
    
    if stats["by_task"]:
        print(f"\n各任务统计:")
        for task, t_stats in sorted(stats["by_task"].items()):
            print(f"\n{task}:")
            print(f"  样本数: {t_stats['count']}")
            print(f"  平均PSNR: {t_stats['average']:.4f} dB")
            print(f"  最大PSNR: {t_stats['max']:.4f} dB")
            print(f"  最小PSNR: {t_stats['min']:.4f} dB")
EOF
    
    # 输出总结
    echo -e "\n${GREEN}================================================${NC}"
    echo -e "${GREEN}执行完成!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo -e "总数: ${total}"
    echo -e "${GREEN}成功: ${success}${NC}"
    echo -e "${RED}失败: ${failed}${NC}"
    echo -e "总耗时: ${hours}h ${minutes}m ${seconds}s"
    
    # 如果有失败的 case，列出它们
    if [ ${#failed_cases[@]} -gt 0 ]; then
        echo -e "\n${RED}失败的 case:${NC}"
        for failed_case in "${failed_cases[@]}"; do
            echo -e "  ${RED}✗ ${failed_case}${NC}"
        done
    fi
    
    echo -e "\n${GREEN}PSNR 汇总文件: ${summary_file}${NC}"
    echo -e "${GREEN}================================================${NC}\n"
    
    # 生成执行报告
    echo -e "${YELLOW}生成执行报告...${NC}"
    {
        echo "执行时间: $(date)"
        echo "总数: ${total}"
        echo "成功: ${success}"
        echo "失败: ${failed}"
        echo "总耗时: ${hours}h ${minutes}m ${seconds}s"
        echo ""
        echo "PSNR 结果目录: ${PSNR_RESULTS_DIR}"
        echo "PSNR 汇总文件: ${summary_file}"
        echo ""
        if [ ${#failed_cases[@]} -gt 0 ]; then
            echo "失败的 case:"
            for failed_case in "${failed_cases[@]}"; do
                echo "  - ${failed_case}"
            done
        fi
    } > execution_report_${timestamp}.txt
    
    echo -e "${GREEN}报告已保存到: execution_report_${timestamp}.txt${NC}\n"
    
    # 返回状态码
    if [ $failed -eq 0 ]; then
        return 0
    else
        return 1
    fi
}

# 运行主函数
main

exit $?