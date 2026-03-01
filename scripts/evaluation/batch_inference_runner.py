#!/usr/bin/env python3
"""
批量推理运行器
在单个 Python 进程中连续运行多个 case，复用模型和数据集实例
避免每个 case 都重新初始化大模型
"""

import argparse
import os
import sys
import re
import subprocess
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import torch
from pytorch_lightning import seed_everything


# ===================== 配置解析 =====================
def extract_args_from_shell_script(script_path: str) -> Dict[str, Any]:
    """
    从 run_world_model_interaction.sh 脚本中提取参数
    
    Args:
        script_path: shell 脚本路径
        
    Returns:
        解析出的参数字典
    """
    extracted = {}
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"脚本不存在: {script_path}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # 解析常见参数模式 - 支持引号包裹的值
    arg_patterns = {
        'savedir': r'--savedir[=\s]+["\']?([^"\'\\\s]+)',
        'ckpt_path': r'--ckpt_path[=\s]+["\']?([^"\'\\\s]+)',
        'config': r'--config[=\s]+["\']?([^"\'\\\s]+)',
        'prompt_dir': r'--prompt_dir[=\s]+["\']?([^"\'\\\s]+)',
        'dataset': r'--dataset[=\s]+[\$\{]?([a-zA-Z0-9_]+)',
        'ddim_steps': r'--ddim_steps[=\s]+(\d+)',
        'ddim_eta': r'--ddim_eta[=\s]+([\d.]+)',
        'height': r'--height[=\s]+(\d+)',
        'width': r'--width[=\s]+(\d+)',
        'frame_stride': r'--frame_stride[=\s]+(\d+)',
        'unconditional_guidance_scale': r'--unconditional_guidance_scale[=\s]+([\d.]+)',
        'seed': r'--seed[=\s]+(\d+)',
        'video_length': r'--video_length[=\s]+(\d+)',
        'n_iter': r'--n_iter[=\s]+(\d+)',
        'exe_steps': r'--exe_steps[=\s]+(\d+)',
        'save_fps': r'--save_fps[=\s]+(\d+)',
        'timestep_spacing': r"--timestep_spacing[=\s]+['\"]?([^'\"\s]+)",
        'guidance_rescale': r'--guidance_rescale[=\s]+([\d.]+)',
    }
    
    for arg_name, pattern in arg_patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            # 处理变量引用 (如 ${dataset})
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1]
            # 在脚本中查找变量定义
            if arg_name == 'dataset' and value == 'dataset':
                var_match = re.search(r'dataset="([^"]+)"', content)
                if var_match:
                    value = var_match.group(1)
            
            # 类型转换
            if arg_name in ['ddim_steps', 'height', 'width', 'frame_stride', 
                           'seed', 'video_length', 'n_iter', 'exe_steps', 'save_fps']:
                extracted[arg_name] = int(value)
            elif arg_name in ['ddim_eta', 'unconditional_guidance_scale', 'guidance_rescale']:
                extracted[arg_name] = float(value)
            else:
                extracted[arg_name] = value
    
    # 检查布尔参数 (需要确保不是被注释掉的)
    # 移除注释行
    lines_no_comment = '\n'.join(
        line for line in content.split('\n') 
        if not line.strip().startswith('#')
    )
    
    if '--perframe_ae' in lines_no_comment:
        extracted['perframe_ae'] = True
    if '--zero_pred_state' in lines_no_comment:
        extracted['zero_pred_state'] = True
    
    # DeepCache (检查是否被注释)
    if re.search(r'^\s*--use_deepcache', lines_no_comment, re.MULTILINE):
        extracted['use_deepcache'] = True
        
        dc_interval = re.search(r'--deepcache_interval[=\s]+(\d+)', lines_no_comment)
        if dc_interval:
            extracted['deepcache_interval'] = int(dc_interval.group(1))
        dc_branch = re.search(r'--deepcache_branch_id[=\s]+(\d+)', lines_no_comment)
        if dc_branch:
            extracted['deepcache_branch_id'] = int(dc_branch.group(1))
    
    return extracted


# ===================== Case 定义 =====================
# 格式: "任务目录/case:视频名称"
DEFAULT_CASE_LIST = [
    "unitree_g1_pack_camera/case1:0_full_fs6",
    "unitree_g1_pack_camera/case2:50_full_fs6",
    "unitree_g1_pack_camera/case3:100_full_fs6",
    "unitree_g1_pack_camera/case4:200_full_fs6",
    "unitree_z1_stackbox/case1:5_full_fs4",
    "unitree_z1_stackbox/case2:15_full_fs4",
    "unitree_z1_stackbox/case3:25_full_fs4",
    "unitree_z1_stackbox/case4:35_full_fs4",
    "unitree_z1_dual_arm_stackbox/case1:5_full_fs4",
    "unitree_z1_dual_arm_stackbox/case2:15_full_fs4",
    "unitree_z1_dual_arm_stackbox/case3:25_full_fs4",
    "unitree_z1_dual_arm_stackbox/case4:35_full_fs4",
    "unitree_z1_dual_arm_stackbox_v2/case1:5_full_fs4",
    "unitree_z1_dual_arm_stackbox_v2/case2:15_full_fs4",
    "unitree_z1_dual_arm_stackbox_v2/case3:25_full_fs4",
    "unitree_z1_dual_arm_stackbox_v2/case4:35_full_fs4",
    "unitree_z1_dual_arm_cleanup_pencils/case1:0_full_fs4",
    "unitree_z1_dual_arm_cleanup_pencils/case2:50_full_fs4",
    "unitree_z1_dual_arm_cleanup_pencils/case3:100_full_fs4",
    "unitree_z1_dual_arm_cleanup_pencils/case4:200_full_fs4",
]


class CaseDefinition:
    """单个 case 的定义"""
    
    def __init__(self, case_spec: str, base_dir: str):
        """
        Args:
            case_spec: 格式 "任务目录/case:视频名称"
            base_dir: 工作区根目录
        """
        parts = case_spec.split(':')
        self.case_dir = parts[0]
        self.video_name = parts[1] if len(parts) > 1 else ""
        self.base_dir = base_dir
        
        # 解析任务和 case 名
        dir_parts = self.case_dir.split('/')
        self.task_name = dir_parts[0]
        self.case_name = dir_parts[1] if len(dir_parts) > 1 else ""
        
    @property
    def full_case_dir(self) -> str:
        return os.path.join(self.base_dir, self.case_dir)
    
    @property
    def shell_script_path(self) -> str:
        return os.path.join(self.full_case_dir, "run_world_model_interaction.sh")
    
    @property
    def gt_video_path(self) -> str:
        return os.path.join(self.full_case_dir, 
                          f"{self.task_name}_{self.case_name}.mp4")
    
    def get_args(self) -> Dict[str, Any]:
        """从对应的 shell 脚本提取参数"""
        return extract_args_from_shell_script(self.shell_script_path)


# ===================== 批量运行器 =====================
class BatchInferenceExecutor:
    """批量推理执行器 - 复用模型实例"""
    
    def __init__(self, base_dir: str, gpu_id: int = 0):
        self.base_dir = base_dir
        self.gpu_id = gpu_id
        
        # 模型缓存 (config_path -> model)
        self._cached_model = None
        self._cached_config_path = None
        
        # 数据集缓存 (config_path -> data)
        self._cached_data = None
        self._cached_data_config = None
        
        # 结果记录
        self.execution_results: List[Dict] = []
        
    def _load_or_reuse_model(self, config_path: str, ckpt_path: str):
        """加载或复用模型实例"""
        from omegaconf import OmegaConf
        from unifolm_wma.utils.utils import instantiate_from_config
        
        # 检查是否可以复用
        if (self._cached_model is not None and 
            self._cached_config_path == config_path):
            print(f"[BatchRunner] 复用已加载的模型")
            return self._cached_model
        
        print(f"[BatchRunner] 加载模型: {config_path}")
        config = OmegaConf.load(config_path)
        config['model']['params']['wma_config']['params']['use_checkpoint'] = False
        
        # 创建模型
        model = instantiate_from_config(config.model)
        
        # 加载权重
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        
        model.eval()
        model = model.cuda(self.gpu_id)
        
        # 缓存
        self._cached_model = model
        self._cached_config_path = config_path
        
        return model
    
    def _load_or_reuse_data(self, config_path: str):
        """加载或复用数据集实例"""
        from omegaconf import OmegaConf
        from unifolm_wma.utils.utils import instantiate_from_config
        
        if (self._cached_data is not None and 
            self._cached_data_config == config_path):
            print(f"[BatchRunner] 复用已加载的数据集")
            return self._cached_data
        
        print(f"[BatchRunner] 加载数据集配置: {config_path}")
        config = OmegaConf.load(config_path)
        data = instantiate_from_config(config.data)
        data.setup()
        
        self._cached_data = data
        self._cached_data_config = config_path
        
        return data
    
    def run_single_case(self, case_def: CaseDefinition, 
                       seed_value: int) -> Tuple[bool, float, Optional[str]]:
        """
        运行单个 case
        
        Args:
            case_def: Case 定义
            seed_value: 随机种子（每个 case 独立设置，防止随机数漂移）
            
        Returns:
            (成功标志, 耗时秒数, 错误信息)
        """
        # 关键: 每个 case 重新设置种子，防止跨 case 随机数流漂移
        seed_everything(seed_value)
        torch.cuda.empty_cache()
        
        start_time = time.time()
        error_msg = None
        
        try:
            # 获取 case 参数
            case_args = case_def.get_args()
            
            print(f"\n{'='*60}")
            print(f"[BatchRunner] 运行 Case: {case_def.case_dir}")
            print(f"[BatchRunner] Seed: {seed_value}")
            print(f"{'='*60}\n")
            
            # 加载/复用模型和数据
            model = self._load_or_reuse_model(
                case_args['config'], 
                case_args['ckpt_path']
            )
            data = self._load_or_reuse_data(case_args['config'])
            
            # 运行推理
            self._run_inference_internal(model, data, case_args, case_def)
            
            elapsed = time.time() - start_time
            return True, elapsed, None
            
        except Exception as e:
            import traceback
            elapsed = time.time() - start_time
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[BatchRunner] Case {case_def.case_dir} 失败: {error_msg}")
            return False, elapsed, error_msg
    
    def _run_inference_internal(self, model, data, args: Dict, case_def: CaseDefinition):
        """
        内部推理实现 - 直接调用推理函数以复用模型
        """
        import sys
        
        # 确保可以导入推理模块
        eval_scripts_dir = os.path.join(self.base_dir, 'scripts', 'evaluation')
        if eval_scripts_dir not in sys.path:
            sys.path.insert(0, eval_scripts_dir)
        
        # 导入推理函数
        from world_model_interaction import run_inference
        
        # 构造 argparse.Namespace 对象
        import argparse
        
        # 确定 savedir 和 prompt_dir
        savedir = args.get('savedir')
        if savedir and '${' in savedir:
            # 处理变量替换
            savedir = case_def.full_case_dir + '/output'
        elif not savedir or not os.path.isabs(savedir):
            savedir = os.path.join(self.base_dir, savedir) if savedir else case_def.full_case_dir + '/output'
        
        prompt_dir = args.get('prompt_dir')
        if prompt_dir and '${' in prompt_dir:
            prompt_dir = case_def.full_case_dir + '/world_model_interaction_prompts'
        elif not prompt_dir or not os.path.isabs(prompt_dir):
            prompt_dir = os.path.join(self.base_dir, prompt_dir) if prompt_dir else case_def.full_case_dir + '/world_model_interaction_prompts'
        
        # config 和 ckpt 路径处理
        config_path = args.get('config')
        if config_path and not os.path.isabs(config_path):
            config_path = os.path.join(self.base_dir, config_path)
        
        ckpt_path = args.get('ckpt_path')
        if ckpt_path and not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(self.base_dir, ckpt_path)
        
        inference_args = argparse.Namespace(
            savedir=savedir,
            ckpt_path=ckpt_path,
            config=config_path,
            prompt_dir=prompt_dir,
            dataset=args.get('dataset', case_def.task_name),
            ddim_steps=args.get('ddim_steps', 50),
            ddim_eta=args.get('ddim_eta', 1.0),
            bs=1,
            height=args.get('height', 320),
            width=args.get('width', 512),
            frame_stride=[args.get('frame_stride', 4)],
            unconditional_guidance_scale=args.get('unconditional_guidance_scale', 1.0),
            seed=args.get('seed', 123),
            video_length=args.get('video_length', 16),
            num_generation=1,
            timestep_spacing=args.get('timestep_spacing', 'uniform'),
            guidance_rescale=args.get('guidance_rescale', 0.0),
            perframe_ae=args.get('perframe_ae', False),
            n_action_steps=16,
            exe_steps=args.get('exe_steps', 16),
            n_iter=args.get('n_iter', 40),
            zero_pred_state=args.get('zero_pred_state', False),
            save_fps=args.get('save_fps', 8),
            use_deepcache=args.get('use_deepcache', False),
            deepcache_interval=args.get('deepcache_interval', 3),
            deepcache_branch_id=args.get('deepcache_branch_id', 0),
        )
        
        print(f"[BatchRunner] 直接调用推理函数")
        print(f"[BatchRunner] savedir: {inference_args.savedir}")
        print(f"[BatchRunner] prompt_dir: {inference_args.prompt_dir}")
        print(f"[BatchRunner] dataset: {inference_args.dataset}")
        print(f"[BatchRunner] n_iter: {inference_args.n_iter}")
        print(f"[BatchRunner] timestep_spacing: {inference_args.timestep_spacing}")
        
        # 调用推理函数，传入缓存的模型和数据
        run_inference(inference_args, gpu_num=1, gpu_no=self.gpu_id,
                     cached_model=model, cached_data=data)
    
    def run_all_cases(self, case_specs: List[str], 
                     base_seed: int = 123) -> Dict[str, Any]:
        """
        运行所有 case
        
        Args:
            case_specs: Case 规格列表
            base_seed: 基础种子值
            
        Returns:
            汇总结果
        """
        total_start = time.time()
        success_count = 0
        fail_count = 0
        
        for idx, spec in enumerate(case_specs):
            case_def = CaseDefinition(spec, self.base_dir)
            
            # 每个 case 使用独立种子
            case_seed = base_seed + idx * 1000
            
            success, elapsed, error = self.run_single_case(case_def, case_seed)
            
            result = {
                'case': spec,
                'success': success,
                'elapsed_sec': round(elapsed, 2),
                'error': error
            }
            self.execution_results.append(result)
            
            if success:
                success_count += 1
                print(f"[BatchRunner] ✓ {spec} 完成 ({elapsed:.1f}s)")
            else:
                fail_count += 1
                print(f"[BatchRunner] ✗ {spec} 失败 ({elapsed:.1f}s)")
        
        total_elapsed = time.time() - total_start
        
        summary = {
            'total_cases': len(case_specs),
            'success': success_count,
            'failed': fail_count,
            'total_time_sec': round(total_elapsed, 2),
            'results': self.execution_results
        }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='批量推理运行器')
    parser.add_argument('--base_dir', type=str, default='.',
                       help='工作区根目录')
    parser.add_argument('--gpu', type=int, default=0,
                       help='使用的 GPU ID')
    parser.add_argument('--seed', type=int, default=123,
                       help='基础随机种子')
    parser.add_argument('--cases', type=str, nargs='*', default=None,
                       help='要运行的 case 列表，默认运行所有')
    parser.add_argument('--output', type=str, default='batch_results.json',
                       help='结果输出文件')
    
    args = parser.parse_args()
    
    # 确定要运行的 cases
    case_list = args.cases if args.cases else DEFAULT_CASE_LIST
    
    print(f"[BatchRunner] 准备运行 {len(case_list)} 个 case")
    print(f"[BatchRunner] GPU: {args.gpu}, Seed: {args.seed}")
    
    # 创建执行器并运行
    executor = BatchInferenceExecutor(args.base_dir, args.gpu)
    summary = executor.run_all_cases(case_list, args.seed)
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"[BatchRunner] 执行完成!")
    print(f"[BatchRunner] 成功: {summary['success']}/{summary['total_cases']}")
    print(f"[BatchRunner] 总耗时: {summary['total_time_sec']}s")
    print(f"[BatchRunner] 结果保存到: {args.output}")
    print(f"{'='*60}")
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
