#!/usr/bin/env python3
"""
批量执行多个 case 的推理脚本，支持模型和数据复用。

在同一 Python 进程中连续运行多个 case，避免每个 case 都重复初始化大模型。
"""

import argparse
import copy
import os
import json
import time
import re
import sys

from pytorch_lightning import seed_everything


def get_batch_parser() -> argparse.ArgumentParser:
    """创建批量运行的参数解析器"""
    # 延迟导入以避免循环依赖
    from world_model_interaction import get_parser
    
    parser = get_parser()
    parser.add_argument("--case_root",
                        type=str,
                        required=True,
                        help="case 根目录路径")
    parser.add_argument("--cases_file",
                        type=str,
                        required=True,
                        help="包含 case 路径列表的文本文件")
    parser.add_argument("--gpu_no",
                        type=int,
                        default=0,
                        help="使用的 GPU 索引")
    parser.add_argument("--savedir_name",
                        type=str,
                        default="output",
                        help="每个 case 下的输出目录名")
    parser.add_argument("--batch_report_path",
                        type=str,
                        default=None,
                        help="批量运行报告的 JSON 输出路径")
    parser.add_argument("--parse_case_script",
                        type=int,
                        default=1,
                        choices=[0, 1],
                        help="是否解析每个 case 的脚本获取参数覆盖")
    parser.add_argument("--case_script_name",
                        type=str,
                        default="run_world_model_interaction.sh",
                        help="case 本地脚本文件名")
    return parser


def load_case_list(cases_file: str) -> list[str]:
    """从文件加载 case 列表"""
    cases = []
    with open(cases_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(line.rstrip("/"))
    return cases


def _extract_script_arg(script_content: str, arg_name: str) -> str | None:
    """从 shell 脚本中提取参数值"""
    pattern = rf"--{re.escape(arg_name)}\s+([^\s\\]+)"
    match = re.search(pattern, script_content)
    if not match:
        return None
    token = match.group(1).strip().strip("'").strip('"')
    # 跳过 shell 变量
    if token.startswith("${") or token.startswith("$"):
        return None
    return token


def parse_case_script_args(script_path: str) -> dict:
    """解析 case 脚本中的参数"""
    with open(script_path, "r") as f:
        content = f.read()

    # 需要解析的整数参数
    int_args = [
        "seed", "ddim_steps", "video_length", "frame_stride",
        "n_action_steps", "exe_steps", "n_iter", "save_fps",
        "height", "width", "bs",
    ]
    # 需要解析的浮点参数
    float_args = [
        "ddim_eta", "unconditional_guidance_scale", "guidance_rescale",
    ]
    # 需要解析的字符串参数
    str_args = [
        "timestep_spacing", "ckpt_path", "config", "prompt_dir",
        "dataset", "savedir",
    ]

    overrides = {}
    
    for key in int_args:
        raw = _extract_script_arg(content, key)
        if raw is not None:
            try:
                overrides[key] = int(raw)
            except ValueError:
                pass
    
    for key in float_args:
        raw = _extract_script_arg(content, key)
        if raw is not None:
            try:
                overrides[key] = float(raw)
            except ValueError:
                pass
    
    for key in str_args:
        raw = _extract_script_arg(content, key)
        if raw is not None:
            overrides[key] = raw

    # frame_stride 需要转换为列表
    if "frame_stride" in overrides:
        overrides["frame_stride"] = [int(overrides["frame_stride"])]
    
    return overrides


def main():
    # 延迟导入
    from world_model_interaction import run_inference
    
    parser = get_batch_parser()
    args = parser.parse_args()

    # 初始化随机种子
    base_seed = args.seed
    if base_seed < 0:
        base_seed = int.from_bytes(os.urandom(4), "little")
    seed_everything(base_seed)

    # 加载 case 列表
    case_list = load_case_list(args.cases_file)
    if not case_list:
        raise ValueError(f"未在 {args.cases_file} 中找到有效的 case")

    print(f"[batch] 准备运行 {len(case_list)} 个 case")
    
    report_records = []
    total_start = time.perf_counter()
    
    for case_idx, case_rel in enumerate(case_list):
        case_dir = os.path.join(args.case_root, case_rel)
        if not os.path.isdir(case_dir):
            raise FileNotFoundError(f"Case 目录不存在: {case_dir}")
        
        # 解析 case 信息
        scenario = case_rel.split("/", 1)[0]
        case_name = case_rel.split("/")[-1]
        
        # 复制参数并应用覆盖
        case_args = copy.deepcopy(args)
        
        if args.parse_case_script == 1:
            script_path = os.path.join(case_dir, args.case_script_name)
            if os.path.isfile(script_path):
                overrides = parse_case_script_args(script_path)
                for k, v in overrides.items():
                    setattr(case_args, k, v)
                print(f"[batch] 从脚本解析到 {len(overrides)} 个参数覆盖")

        # 设置路径参数
        case_args.savedir = os.path.join(case_dir, args.savedir_name)
        case_args.prompt_dir = os.path.join(case_dir, "world_model_interaction_prompts")
        case_args.dataset = scenario

        # 关键: 每个 case 重新设置随机种子，确保与单独运行时行为一致
        case_seed = getattr(case_args, 'seed', base_seed)
        seed_everything(int(case_seed))
        
        print(f"\n[batch] ========== Case {case_idx + 1}/{len(case_list)} ==========")
        print(f"[batch] 开始运行: {case_rel}")
        print(f"[batch] dataset={scenario}, seed={case_seed}")
        
        t0 = time.perf_counter()
        try:
            run_inference(case_args, gpu_num=1, gpu_no=args.gpu_no)
            elapsed = time.perf_counter() - t0
            status = "success"
            print(f"[batch] 完成: {case_rel}, 耗时={elapsed:.2f}s")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            status = f"error: {str(e)}"
            print(f"[batch] 失败: {case_rel}, 错误={e}")
        
        report_records.append({
            "case_rel": case_rel,
            "scenario": scenario,
            "case_name": case_name,
            "elapsed_s": float(elapsed),
            "status": status,
            "savedir": case_args.savedir,
        })

    total_elapsed = time.perf_counter() - total_start
    
    # 保存报告
    if args.batch_report_path:
        os.makedirs(os.path.dirname(args.batch_report_path) or ".", exist_ok=True)
        report = {
            "total_cases": len(case_list),
            "total_elapsed_s": float(total_elapsed),
            "records": report_records
        }
        with open(args.batch_report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[batch] 报告已保存到: {args.batch_report_path}")
    
    # 打印总结
    success_count = sum(1 for r in report_records if r["status"] == "success")
    print(f"\n[batch] ========== 运行完成 ==========")
    print(f"[batch] 总数: {len(case_list)}, 成功: {success_count}, 失败: {len(case_list) - success_count}")
    print(f"[batch] 总耗时: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
