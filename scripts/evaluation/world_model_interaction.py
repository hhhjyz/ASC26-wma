import argparse, os, glob, sys
import contextlib
import shutil
import pandas as pd
import random
import torch
import torchvision
import h5py
import numpy as np
import logging
import einops
import warnings
import imageio
import time
import json

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
from torch import nn
from eval_utils import populate_queues, log_to_tensorboard
from collections import deque
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from typing import Any, Optional, Dict

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.models.samplers.ddim_deepcache import DDIMSamplerDeepCache
from unifolm_wma.utils.utils import instantiate_from_config


# ===================== 优化: 环境变量控制 =====================
# TensorBoard 开关 (0=禁用, 1=启用)
_TENSORBOARD_ENABLED = os.environ.get("WMA_ENABLE_TB", "1") == "1"
# 中间结果保存开关
_SAVE_INTERMEDIATE = os.environ.get("WMA_SAVE_INTERMEDIATE", "1") == "1"
# 性能分析开关
_PROFILING_ENABLED = os.environ.get("WMA_PROFILING", "0") == "1"
# 迭代次数调整 (delta 模式)
_ITER_ADJUSTMENT = int(os.environ.get("WMA_ITER_DELTA", "0"))
# 迭代次数覆盖 (绝对值模式，优先级高于 delta)
_ITER_OVERRIDE = os.environ.get("WMA_ITER_OVERRIDE", "")
# torch.compile 开关
_USE_TORCH_COMPILE = os.environ.get("WMA_TORCH_COMPILE", "0") == "1"
# Action 步骤单独的 ddim_steps
_ACTION_DDIM_STEPS = os.environ.get("WMA_ACTION_DDIM_STEPS", "")
# ── Cast 模式选择 ──
# WMA_CAST_MODE=v1  → 旧版: model.cuda() 在 data.setup() 之前, 仅 autocast, 无 model.half()
# WMA_CAST_MODE=v2  → 新版: model.cuda() 在 data.setup() 之后, 支持 model.half() + autocast
_CAST_MODE = os.environ.get("WMA_CAST_MODE", "v2")
# FP16 autocast 推理开关
_USE_FP16 = os.environ.get("WMA_FP16", os.environ.get("WMA_USE_FP16", "0")) == "1"
# FP16 模型权重 casting 策略 (仅 v2 模式 + _USE_FP16=True 时有效)
_FORCE_FULL_FP16 = os.environ.get("WMA_FORCE_FULL_FP16", "0") == "1"
_FORCE_FP16_DIFFUSION_ONLY = os.environ.get("WMA_FORCE_FP16_DIFFUSION_ONLY", "0") == "1"


# ===================== 优化: StageTimer =====================
class StageTimer:
    """Lightweight stage timer for startup/preparation profiling."""

    def __init__(self, name: str, prefix: str = "prep-timer"):
        self.name = name
        self.prefix = prefix
        self._start = time.perf_counter()
        self._last = self._start

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        step = now - self._last
        total = now - self._start
        print(f">>> [{self.prefix}] {self.name}.{stage}: {step:.3f}s (total {total:.3f}s)")
        self._last = now


# ===================== 优化: SHM 预加载 =====================
def ensure_ckpt_preloaded_to_shm(src_path: str,
                                 shm_dir: str = "/dev/shm/unifolm_wma_ckpts",
                                 force_copy: bool = False) -> str:
    """将 checkpoint 预复制到 /dev/shm 高速内存，加速 torch.load。
    
    为避免多用户权限冲突，自动在 shm_dir 下按用户名隔离子目录。
    """
    import getpass
    src_abs = os.path.abspath(os.path.expanduser(src_path))
    # 每个用户使用独立子目录，避免 Permission denied
    user = getpass.getuser()
    shm_dir_abs = os.path.abspath(os.path.join(os.path.expanduser(shm_dir), user))
    shm_real = os.path.realpath(shm_dir_abs)
    src_real = os.path.realpath(src_abs)
    if src_real == shm_real or src_real.startswith(shm_real + os.sep):
        print(f">>> [shm] checkpoint already in shm: {src_abs}")
        return src_abs

    os.makedirs(shm_dir_abs, exist_ok=True)
    dst_abs = os.path.join(shm_dir_abs, os.path.basename(src_abs))
    if os.path.exists(dst_abs) and not force_copy:
        # 额外检查可读性，防止残留的无权限文件
        if os.access(dst_abs, os.R_OK):
            print(f">>> [shm] reuse preloaded checkpoint: {dst_abs}")
            return dst_abs
        else:
            print(f">>> [shm] cached file not readable, re-copying: {dst_abs}")
            try:
                os.remove(dst_abs)
            except OSError:
                pass

    src_size = os.path.getsize(src_abs)
    free_bytes = shutil.disk_usage(shm_dir_abs).free
    if free_bytes < src_size:
        print(f">>> [shm] insufficient shm space (need {src_size}, free {free_bytes}), fallback to source ckpt.")
        return src_abs

    tmp_abs = dst_abs + ".tmp"
    copy_start = time.perf_counter()
    with open(src_abs, "rb", buffering=0) as src_f, \
         open(tmp_abs, "wb", buffering=0) as dst_f:
        shutil.copyfileobj(src_f, dst_f, length=64 * 1024 * 1024)
        dst_f.flush()
        os.fsync(dst_f.fileno())
    os.replace(tmp_abs, dst_abs)
    elapsed = max(time.perf_counter() - copy_start, 1e-6)
    speed_mb_s = (src_size / (1024 * 1024)) / elapsed
    print(f">>> [shm] preloaded checkpoint to shm: {dst_abs} ({elapsed:.2f}s, {speed_mb_s:.1f} MB/s)")
    return dst_abs


# ===================== 优化: 性能计时 =====================
class InferenceTimer:
    """推理性能计时器"""
    
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.records: Dict[str, list] = {}
        
    def start(self, tag: str):
        if not self.enabled:
            return
        if tag not in self.records:
            self.records[tag] = []
        self.records[tag].append({'start': time.perf_counter()})
        
    def stop(self, tag: str):
        if not self.enabled or tag not in self.records:
            return
        if self.records[tag] and 'end' not in self.records[tag][-1]:
            self.records[tag][-1]['end'] = time.perf_counter()
            self.records[tag][-1]['elapsed'] = (
                self.records[tag][-1]['end'] - self.records[tag][-1]['start']
            )
    
    def get_stats(self) -> Dict[str, Dict]:
        stats = {}
        for tag, entries in self.records.items():
            times = [e.get('elapsed', 0) for e in entries if 'elapsed' in e]
            if times:
                stats[tag] = {
                    'count': len(times),
                    'total': sum(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        return stats
    
    def report(self):
        if not self.enabled:
            return
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("性能统计:")
        print("=" * 60)
        for tag, s in stats.items():
            print(f"  {tag}: avg={s['avg']:.3f}s, total={s['total']:.2f}s, count={s['count']}")
        print("=" * 60 + "\n")


# 全局计时器
_PERF_TIMER = InferenceTimer(_PROFILING_ENABLED)


def _build_autocast_ctx():
    """
    构建 FP16 autocast 上下文管理器 (与 asc26-wma-opt 参考代码对齐)
    通过 WMA_FP16 / WMA_USE_FP16 环境变量控制
    """
    if _USE_FP16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        return contextlib.nullcontext()


@contextlib.contextmanager
def skip_module_init_context(enabled: bool):
    """
    上下文管理器: 跳过模块参数初始化以加速推理模式启动
    在推理时，参数会被 checkpoint 覆盖，初始化是多余的
    
    优化内容：
    1. 跳过 nn.Module 子类的 reset_parameters 方法
    2. 跳过 nn.init.* 的所有初始化函数
    3. 使用 meta device 创建参数（PyTorch 2.0+）
    """
    if not enabled:
        yield
        return

    def _noop(*_args, **_kwargs):
        return None
    
    def _noop_return_tensor(tensor, *_args, **_kwargs):
        """对于需要返回 tensor 的 init 函数"""
        return tensor

    # 需要跳过 reset_parameters 的模块类型
    target_classes = [
        nn.Linear, nn.Embedding, nn.LayerNorm, nn.GroupNorm,
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.MultiheadAttention,
    ]
    
    # nn.init 中需要跳过的函数
    init_funcs = [
        'uniform_', 'normal_', 'constant_', 'ones_', 'zeros_',
        'eye_', 'dirac_', 'xavier_uniform_', 'xavier_normal_',
        'kaiming_uniform_', 'kaiming_normal_', 'orthogonal_', 'sparse_',
        'trunc_normal_',
    ]
    
    saved_methods = []
    saved_init_funcs = []
    
    # 跳过 reset_parameters
    for cls in target_classes:
        if hasattr(cls, "reset_parameters"):
            saved_methods.append((cls, "reset_parameters", cls.reset_parameters))
            cls.reset_parameters = _noop
    
    # 跳过 nn.init.* 函数
    for func_name in init_funcs:
        if hasattr(nn.init, func_name):
            saved_init_funcs.append((func_name, getattr(nn.init, func_name)))
            setattr(nn.init, func_name, _noop_return_tensor)

    try:
        yield
    finally:
        # 恢复 reset_parameters
        for cls, method_name, original_fn in saved_methods:
            setattr(cls, method_name, original_fn)
        # 恢复 nn.init.* 函数
        for func_name, original_fn in saved_init_funcs:
            setattr(nn.init, func_name, original_fn)


def load_checkpoint_optimized(ckpt_path: str, 
                              target_device: Optional[torch.device] = None,
                              use_mmap: bool = True) -> dict:
    """
    优化的 checkpoint 加载 - 支持直接加载到 GPU
    
    Args:
        ckpt_path: checkpoint 路径
        target_device: 目标设备，None 表示 CPU，可指定 'cuda:0' 等直接加载到 GPU
        use_mmap: 是否使用内存映射 (仅在 CPU 加载时有效)
    """
    # 确定加载目标设备
    if target_device is None:
        map_loc = 'cpu'
    elif isinstance(target_device, torch.device):
        map_loc = target_device
    else:
        map_loc = str(target_device)
    
    load_kwargs = {'map_location': map_loc}
    
    # PyTorch 2.0+ 支持 mmap 和 weights_only (仅 CPU 时使用 mmap)
    if use_mmap and map_loc == 'cpu':
        try:
            load_kwargs['mmap'] = True
            load_kwargs['weights_only'] = True
        except TypeError:
            pass
    elif map_loc != 'cpu':
        # GPU 直接加载时也尝试 weights_only 以提高安全性
        try:
            load_kwargs['weights_only'] = True
        except TypeError:
            pass
    
    state_dict = torch.load(ckpt_path, **load_kwargs)
    
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    return state_dict


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Args:
        module (nn.Module): The model whose device is to be inferred.

    Returns:
        torch.device: The device of the model's parameters.
    """
    return next(iter(module.parameters())).device


def write_video(video_path: str, stacked_frames: list, fps: int) -> None:
    """Save a list of frames to a video file.

    Args:
        video_path (str): Output path for the video.
        stacked_frames (list): List of image frames.
        fps (int): Frames per second for the video.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                "pkg_resources is deprecated as an API",
                                category=DeprecationWarning)
        imageio.mimsave(video_path, stacked_frames, fps=fps)


def get_filelist(data_dir: str, postfixes: list[str]) -> list[str]:
    """Return sorted list of files in a directory matching specified postfixes.

    Args:
        data_dir (str): Directory path to search in.
        postfixes (list[str]): List of file extensions to match.

    Returns:
        list[str]: Sorted list of file paths.
    """
    patterns = [
        os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes
    ]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def load_model_checkpoint(model: nn.Module, ckpt: str, 
                          map_location: str = "cpu") -> nn.Module:
    """Load model weights from checkpoint file.

    Args:
        model (nn.Module): Model instance.
        ckpt (str): Path to the checkpoint file.
        map_location: 'cpu' or 'cuda:X'. GPU 直接加载在本地 NVMe 上更快，
                      但在 CephFS 网络存储上 CPU 加载更快。

    Returns:
        nn.Module: Model with loaded weights.
    """
    t0 = time.perf_counter()
    state_dict = torch.load(ckpt, map_location=map_location, weights_only=False)
    t1 = time.perf_counter()
    print(f'>>> [ckpt] torch.load({map_location}): {t1-t0:.2f}s')

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Optional verification to detect missing keys / mismatches that can
    # cause quality regressions (enable with WMA_VERIFY_CKPT_MATCH=1).
    verify = os.environ.get("WMA_VERIFY_CKPT_MATCH", "0") == "1"

    if verify:
        # Compare keys and shapes before loading
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing_in_model = ckpt_keys - model_keys
        unexpected_in_model = model_keys - ckpt_keys
        if missing_in_model:
            print(f">>> [verify] Warning: {len(missing_in_model)} ckpt keys not found in model (examples: {list(missing_in_model)[:5]})")
        if unexpected_in_model:
            print(f">>> [verify] Note: {len(unexpected_in_model)} model keys missing in ckpt (examples: {list(unexpected_in_model)[:5]})")

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        new_pl_sd = OrderedDict()
        for k, v in state_dict.items():
            new_pl_sd[k] = v

        for k in list(new_pl_sd.keys()):
            if "framestride_embed" in k:
                new_key = k.replace("framestride_embed", "fps_embedding")
                new_pl_sd[new_key] = new_pl_sd[k]
                del new_pl_sd[k]
        model.load_state_dict(new_pl_sd, strict=True)

    if verify:
        # Compute a lightweight aggregate L2 difference between ckpt tensors and
        # model parameters to catch silent mismatches (run on CPU to avoid GPU
        # memory blowup). If anything non-zero appears it indicates keys were
        # not fully applied or shapes mismatched.
        total_sq = 0.0
        count = 0
        for k, ckpt_v in state_dict.items():
            try:
                model_v = model.state_dict()[k]
            except KeyError:
                continue
            # move to cpu and numeric check
            ck = ckpt_v.detach().cpu().to(torch.float32)
            mv = model_v.detach().cpu().to(torch.float32)
            if ck.shape != mv.shape:
                print(f">>> [verify] shape mismatch for {k}: ckpt={ck.shape}, model={mv.shape}")
                total_sq += 1.0
                count += 1
                continue
            diff = (ck - mv).float()
            total_sq += float(torch.sum(diff * diff).item())
            count += ck.numel()
        if count > 0:
            rmse = math.sqrt(total_sq / max(1, count))
            print(f">>> [verify] checkpoint vs model RMSE={rmse:.6e} (over {count} elements)")
            if rmse > 1e-6:
                print(">>> [verify] Warning: non-zero RMSE indicates parameters may not match checkpoint exactly.")

    print('>>> model checkpoint loaded.')
    return model


def is_inferenced(save_dir: str, filename: str) -> bool:
    """Check if a given filename has already been processed and saved.

    Args:
        save_dir (str): Directory where results are saved.
        filename (str): Name of the file to check.

    Returns:
        bool: True if processed file exists, False otherwise.
    """
    video_file = os.path.join(save_dir, "samples_separate",
                              f"{filename[:-4]}_sample0.mp4")
    return os.path.exists(video_file)


def save_results(video: Tensor, filename: str, fps: int = 8) -> None:
    """Save video tensor to file using torchvision.

    Args:
        video (Tensor): Tensor of shape (B, C, T, H, W).
        filename (str): Output file path.
        fps (int, optional): Frames per second. Defaults to 8.
    """
    video = video.detach()
    video = torch.clamp(video.float(), -1., 1.)
    n = video.shape[0]
    video = video.permute(2, 0, 1, 3, 4)

    frame_grids = [
        torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0)
        for framesheet in video
    ]
    grid = torch.stack(frame_grids, dim=0)
    grid = (grid + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
    if grid.device.type != 'cpu':
        grid = grid.cpu()
    torchvision.io.write_video(filename,
                               grid,
                               fps=fps,
                               video_codec='h264',
                               options={'crf': '10'})


def get_init_frame_path(data_dir: str, sample: dict) -> str:
    """Construct the init_frame path from directory and sample metadata.

    Args:
        data_dir (str): Base directory containing videos.
        sample (dict): Dictionary containing 'data_dir' and 'videoid'.

    Returns:
        str: Full path to the video file.
    """
    rel_video_fp = os.path.join(sample['data_dir'],
                                str(sample['videoid']) + '.png')
    full_image_fp = os.path.join(data_dir, 'images', rel_video_fp)
    return full_image_fp


def get_transition_path(data_dir: str, sample: dict) -> str:
    """Construct the full transition file path from directory and sample metadata.

    Args:
        data_dir (str): Base directory containing transition files.
        sample (dict): Dictionary containing 'data_dir' and 'videoid'.

    Returns:
        str: Full path to the HDF5 transition file.
    """
    rel_transition_fp = os.path.join(sample['data_dir'],
                                     str(sample['videoid']) + '.h5')
    full_transition_fp = os.path.join(data_dir, 'transitions',
                                      rel_transition_fp)
    return full_transition_fp


def prepare_init_input(start_idx: int,
                       init_frame_path: str,
                       transition_dict: dict[str, torch.Tensor],
                       frame_stride: int,
                       wma_data,
                       video_length: int = 16,
                       n_obs_steps: int = 2) -> dict[str, Tensor]:
    """
    Extracts a structured sample from a video sequence including frames, states, and actions,
    along with properly padded observations and pre-processed tensors for model input.

    Args:
        start_idx (int): Starting frame index for the current clip.
        video: decord video instance.
        transition_dict (Dict[str, Tensor]): Dictionary containing tensors for 'action', 
                                             'observation.state', 'action_type', 'state_type'.
        frame_stride (int): Temporal stride between sampled frames.
        wma_data: Object that holds configuration and utility functions like normalization, 
                transformation, and resolution info.
        video_length (int, optional): Number of frames to sample from the video. Default is 16.
        n_obs_steps (int, optional): Number of historical steps for observations. Default is 2.
    """

    indices = [start_idx + frame_stride * i for i in range(video_length)]
    init_frame = Image.open(init_frame_path).convert('RGB')
    init_frame = torch.tensor(np.array(init_frame)).unsqueeze(0).permute(
        3, 0, 1, 2).float()

    if start_idx < n_obs_steps - 1:
        state_indices = list(range(0, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]
        num_padding = n_obs_steps - 1 - start_idx
        first_slice = states[0:1, :]  # (t, d)
        padding = first_slice.repeat(num_padding, 1)
        states = torch.cat((padding, states), dim=0)
    else:
        state_indices = list(range(start_idx - n_obs_steps + 1, start_idx + 1))
        states = transition_dict['observation.state'][state_indices, :]

    actions = transition_dict['action'][indices, :]

    ori_state_dim = states.shape[-1]
    ori_action_dim = actions.shape[-1]

    frames_action_state_dict = {
        'action': actions,
        'observation.state': states,
    }
    frames_action_state_dict = wma_data.normalizer(frames_action_state_dict)
    frames_action_state_dict = wma_data.get_uni_vec(
        frames_action_state_dict,
        transition_dict['action_type'],
        transition_dict['state_type'],
    )

    if wma_data.spatial_transform is not None:
        init_frame = wma_data.spatial_transform(init_frame)
    init_frame = (init_frame / 255 - 0.5) * 2

    data = {
        'observation.image': init_frame,
    }
    data.update(frames_action_state_dict)
    return data, ori_state_dim, ori_action_dim


def get_latent_z(model, videos: Tensor) -> Tensor:
    """
    Extracts latent features from a video batch using the model's first-stage encoder.

    Args:
        model: the world model.
        videos (Tensor): Input videos of shape [B, C, T, H, W].

    Returns:
        Tensor: Latent video tensor of shape [B, C, T, H, W].
    """
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def preprocess_observation(
        model, observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # Map to expected inputs for the policy
    return_observations = {}

    if isinstance(observations["pixels"], dict):
        imgs = {
            f"observation.images.{key}": img
            for key, img in observations["pixels"].items()
        }
    else:
        imgs = {"observation.images.top": observations["pixels"]}

    for imgkey, img in imgs.items():
        img = torch.from_numpy(img)

        # Sanity check that images are channel last
        _, h, w, c = img.shape
        assert c < h and c < w, f"expect channel first images, but instead {img.shape}"

        # Sanity check that images are uint8
        assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

        # Convert to channel first of type float32 in range [0,1]
        img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
        img = img.type(torch.float32)

        return_observations[imgkey] = img

    return_observations["observation.state"] = torch.from_numpy(
        observations["agent_pos"]).float()
    return_observations['observation.state'] = model.normalize_inputs({
        'observation.state':
        return_observations['observation.state'].to(model.device)
    })['observation.state']

    return return_observations


def image_guided_synthesis_sim_mode(
        model: torch.nn.Module,
        prompts: list[str],
        observation: dict,
        noise_shape: tuple[int, int, int, int, int],
        action_cond_step: int = 16,
        n_samples: int = 1,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        unconditional_guidance_scale: float = 1.0,
        fs: int | None = None,
        text_input: bool = True,
        timestep_spacing: str = 'uniform',
        guidance_rescale: float = 0.0,
        sim_mode: bool = True,
        ddim_sampler: DDIMSampler | None = None,
        decode_video: bool = True,
        use_deepcache: bool = False,
        deepcache_interval: int = 3,
        deepcache_branch_id: int = 0,
        **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs image-guided video generation in a simulation-style mode with optional multimodal guidance (image, state, action, text).

    Args:
        model (torch.nn.Module): The diffusion-based generative model with multimodal conditioning.
        prompts (list[str]): A list of textual prompts to guide the synthesis process.
        observation (dict): A dictionary containing observed inputs including:
            - 'observation.images.top': Tensor of shape [B, O, C, H, W] (top-down images)
            - 'observation.state': Tensor of shape [B, O, D] (state vector)
            - 'action': Tensor of shape [B, T, D] (action sequence)
        noise_shape (tuple[int, int, int, int, int]): Shape of the latent variable to generate, 
            typically (B, C, T, H, W).
        action_cond_step (int): Number of time steps where action conditioning is applied. Default is 16.
        n_samples (int): Number of samples to generate (unused here, always generates 1). Default is 1.
        ddim_steps (int): Number of DDIM sampling steps. Default is 50.
        ddim_eta (float): DDIM eta parameter controlling the stochasticity. Default is 1.0.
        unconditional_guidance_scale (float): Scale for classifier-free guidance. If 1.0, guidance is off.
        fs (int | None): Frame index to condition on, broadcasted across the batch if specified. Default is None.
        text_input (bool): Whether to use text prompt as conditioning. If False, uses empty strings. Default is True.
        timestep_spacing (str): Timestep sampling method in DDIM sampler. Typically "uniform" or "linspace".
        guidance_rescale (float): Guidance rescaling factor to mitigate overexposure from classifier-free guidance.
        sim_mode (bool): Whether to perform world-model interaction or decision-making using the world-model.
        ddim_sampler (DDIMSampler | None): Pre-created DDIM sampler for reuse. If None, creates new one.
        decode_video (bool): Whether to decode latents to video. Set False for action generation.
        **kwargs: Additional arguments passed to the DDIM sampler.

    Returns:
        batch_variants (torch.Tensor): Predicted pixel-space video frames [B, C, T, H, W].
        actions (torch.Tensor): Predicted action sequences [B, T, D] from diffusion decoding.
        states (torch.Tensor): Predicted state sequences [B, T, D] from diffusion decoding.
    """
    b, _, t, _, _ = noise_shape
    
    # Reuse sampler if provided, otherwise create new one
    if ddim_sampler is None:
        if use_deepcache:
            ddim_sampler = DDIMSamplerDeepCache(
                model,
                use_deepcache=True,
                cache_interval=deepcache_interval,
                cache_branch_id=deepcache_branch_id
            )
        else:
            ddim_sampler = DDIMSampler(model)
    
    batch_size = noise_shape[0]

    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    img = observation['observation.images.top'].permute(0, 2, 1, 3, 4)
    cond_img = rearrange(img, 'b o c h w -> (b o) c h w')[-1:]
    cond_img_emb = model.embedder(cond_img)
    cond_img_emb = model.image_proj_model(cond_img_emb)

    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, img.permute(0, 2, 1, 3, 4))
        img_cat_cond = z[:, :, -1:, :, :]
        img_cat_cond = repeat(img_cat_cond,
                              'b c t h w -> b c (repeat t) h w',
                              repeat=noise_shape[2])
        cond = {"c_concat": [img_cat_cond]}

    if not text_input:
        prompts = [""] * batch_size
    cond_ins_emb = model.get_learned_conditioning(prompts)

    cond_state_emb = model.state_projector(observation['observation.state'])
    cond_state_emb = cond_state_emb + model.agent_state_pos_emb

    cond_action_emb = model.action_projector(observation['action'])
    cond_action_emb = cond_action_emb + model.agent_action_pos_emb

    if not sim_mode:
        cond_action_emb = torch.zeros_like(cond_action_emb)

    cond["c_crossattn"] = [
        torch.cat(
            [cond_state_emb, cond_action_emb, cond_ins_emb, cond_img_emb],
            dim=1)
    ]
    cond["c_crossattn_action"] = [
        observation['observation.images.top'][:, :,
                                              -model.n_obs_steps_acting:],
        observation['observation.state'][:, -model.n_obs_steps_acting:],
        sim_mode,
        False,
    ]

    uc = None
    kwargs.update({"unconditional_conditioning_img_nonetext": None})
    cond_mask = None
    cond_z0 = None
    if ddim_sampler is not None:
        samples, actions, states, intermedia = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=batch_size,
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            cfg_img=None,
            mask=cond_mask,
            x0=cond_z0,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs)

        # Reconstruct from latent to pixel space only if needed
        if decode_video:
            batch_images = model.decode_first_stage(samples)
            batch_variants = batch_images
        else:
            # Skip expensive VAE decode for action generation
            batch_variants = samples

    return batch_variants, actions, states


def run_inference(args: argparse.Namespace, gpu_num: int, gpu_no: int,
                  cached_model: Optional[nn.Module] = None,
                  cached_data: Optional[Any] = None) -> None:
    """
    Run inference pipeline on prompts and image inputs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        gpu_num (int): Number of GPUs.
        gpu_no (int): Index of the current GPU.
        cached_model: 可选的预加载模型实例（用于批量复用）
        cached_data: 可选的预加载数据集实例（用于批量复用）

    Returns:
        None
    """
    global _PERF_TIMER
    _PERF_TIMER = InferenceTimer(_PROFILING_ENABLED)
    
    print(f">>> Cast mode: {_CAST_MODE}")
    
    # ===================== 优化: CUDA 后端设置 =====================
    if _CAST_MODE == "v1":
        # v1: 旧版默认值
        enable_tf32 = os.environ.get("WMA_ENABLE_TF32", "0") == "1"
        cudnn_benchmark = os.environ.get("WMA_CUDNN_BENCHMARK", "1") == "1"
    else:
        # v2: 与 asc26-wma-opt 参考代码对齐
        enable_tf32 = os.environ.get("WMA_ENABLE_TF32", "1") == "1"
        cudnn_benchmark = os.environ.get("WMA_CUDNN_BENCHMARK", "0") == "1"

    # FP16 参数验证
    if _FORCE_FULL_FP16 and _FORCE_FP16_DIFFUSION_ONLY:
        raise ValueError("WMA_FORCE_FULL_FP16 and WMA_FORCE_FP16_DIFFUSION_ONLY are mutually exclusive.")
    if (_FORCE_FULL_FP16 or _FORCE_FP16_DIFFUSION_ONLY) and not _USE_FP16:
        logging.warning("WMA_FORCE_FULL_FP16 / WMA_FORCE_FP16_DIFFUSION_ONLY is set but WMA_FP16 is not enabled; ignoring.")
    print(f">>> FP16 config: autocast={_USE_FP16}, force_full={_FORCE_FULL_FP16}, diffusion_only={_FORCE_FP16_DIFFUSION_ONLY}")
    
    if torch.cuda.is_available():
        # TF32 加速 (A100/RTX30xx+ 支持, 默认关闭以保证精度)
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        # cuDNN benchmark (自动选择最快的卷积算法)
        torch.backends.cudnn.benchmark = cudnn_benchmark
        # 确定性算法 (关闭以提高性能)
        torch.backends.cudnn.deterministic = False
        # PyTorch 2.0+ 矩阵乘法精度
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")
        print(f">>> CUDA backend: TF32={enable_tf32}, cuDNN_benchmark={cudnn_benchmark}")
    
    # Create inference and tensorboard dirs
    os.makedirs(args.savedir + '/inference', exist_ok=True)
    
    # TensorBoard 可选
    writer = None
    if _TENSORBOARD_ENABLED:
        log_dir = args.savedir + f"/tensorboard"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
    
    # Load prompt
    csv_path = os.path.join(args.prompt_dir, f"{args.dataset}.csv")
    df = pd.read_csv(csv_path)

    # 计算实际迭代次数
    actual_n_iter = args.n_iter
    if _ITER_OVERRIDE:
        actual_n_iter = int(_ITER_OVERRIDE)
        print(f">>> 迭代次数覆盖为: {actual_n_iter}")
    elif _ITER_ADJUSTMENT != 0:
        actual_n_iter = max(1, args.n_iter + _ITER_ADJUSTMENT)
        print(f">>> 迭代次数调整: {args.n_iter} + {_ITER_ADJUSTMENT} = {actual_n_iter}")
    
    # Action 分支的 ddim_steps
    action_ddim_steps = args.ddim_steps
    if _ACTION_DDIM_STEPS:
        action_ddim_steps = int(_ACTION_DDIM_STEPS)
        print(f">>> Action 分支 ddim_steps: {action_ddim_steps}")

    # 模型加载 (支持缓存复用)
    preloaded_to_gpu = False
    if cached_model is not None:
        model = cached_model
        preloaded_to_gpu = next(model.parameters()).device.type == 'cuda'
        print(f'>>> 复用已加载的模型实例')
    else:
        startup_timer = StageTimer("init")
        
        # Load config
        config = OmegaConf.load(args.config)
        config['model']['params']['wma_config']['params']['use_checkpoint'] = False
        startup_timer.mark("load_config")
        
        # ── 解析初始化策略参数 ──
        skip_init = os.environ.get("WMA_SKIP_INIT", "1") == "1"
        init_model_device = getattr(args, 'init_model_device', 'cpu')
        checkpoint_load_device = getattr(args, 'checkpoint_load_device', 'cpu')
        preload_ckpt_to_shm = getattr(args, 'preload_ckpt_to_shm', False)
        preload_shm_dir = getattr(args, 'preload_shm_dir', '/dev/shm/unifolm_wma_ckpts')
        
        print(f'>>> [init] skip_param_init={skip_init}, init_model_device={init_model_device}, '
              f'checkpoint_load_device={checkpoint_load_device}, preload_shm={preload_ckpt_to_shm}')
        
        # ── Step 1: 模型实例化 (跳过权重初始化 + 可选 GPU 直接构建) ──
        print(f'>>> 模型实例化 (skip_init={skip_init}, device={init_model_device})...')
        if init_model_device == "gpu" and torch.cuda.is_available():
            torch.cuda.set_device(gpu_no)
            with torch.device(f"cuda:{gpu_no}"):
                with skip_module_init_context(skip_init):
                    model = instantiate_from_config(config.model)
        else:
            with skip_module_init_context(skip_init):
                model = instantiate_from_config(config.model)
        startup_timer.mark("instantiate_model")
        
        model.perframe_ae = args.perframe_ae
        assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
        
        # ── Step 2: 可选 SHM 预加载 ──
        ckpt_path = args.ckpt_path
        if preload_ckpt_to_shm:
            ckpt_path = ensure_ckpt_preloaded_to_shm(args.ckpt_path, preload_shm_dir)
            startup_timer.mark("preload_ckpt_to_shm")
        
        # ── Step 3: 加载 checkpoint ──
        preloaded_to_gpu = (init_model_device == "gpu") and torch.cuda.is_available()
        ckpt_map_location = "cpu"
        
        if checkpoint_load_device == "gpu" and torch.cuda.is_available():
            torch.cuda.set_device(gpu_no)
            ckpt_map_location = f"cuda:{gpu_no}"
            # 如果模型还在 CPU，先搬到 GPU
            if not preloaded_to_gpu:
                print(f'>>> 模型移动到 GPU {gpu_no} (for ckpt direct load)...')
                model = model.cuda(gpu_no)
                startup_timer.mark("move_model_to_gpu_for_ckpt")
                preloaded_to_gpu = True
        
        print(f'>>> 加载 checkpoint (map_location={ckpt_map_location})...')
        model = load_model_checkpoint(model, ckpt_path, map_location=ckpt_map_location)
        startup_timer.mark("load_checkpoint")
        
        if _CAST_MODE == "v1":
            # v1: 旧版流程 — model.cuda() 在 checkpoint 加载后立即执行
            if not preloaded_to_gpu:
                print(f'>>> 模型移动到 GPU {gpu_no}...')
                model = model.cuda(gpu_no)
                startup_timer.mark("move_model_to_gpu")
            else:
                startup_timer.mark("model_already_on_gpu")
        
        model.eval()
        
        # 可选 torch.compile
        if _USE_TORCH_COMPILE and hasattr(torch, 'compile'):
            print(f'>>> 应用 torch.compile...')
            model = torch.compile(model, mode='reduce-overhead')
        
        print(f'>>> 模型加载完成')
    
    print(f'>>> Load pre-trained model ...')

    # Build unnomalizer (支持缓存复用)
    if cached_data is not None:
        data = cached_data
        print(f'>>> 复用已加载的数据集实例')
    else:
        logging.info("***** Configing Data *****")
        config = OmegaConf.load(args.config)
        data = instantiate_from_config(config.data)
        data.setup()
        print(">>> Dataset is successfully loaded ...")

    if _CAST_MODE == "v1":
        # v1: 旧版 — 模型已经在 GPU 上，仅检查兜底
        if next(model.parameters()).device.type == 'cpu':
            model = model.cuda(gpu_no)
    else:
        # v2: 新版 — model.cuda() 在 data.setup() 之后
        # data.setup() 内部调用 seed_everything(123) 重置 RNG，
        # 之后 model.cuda() 消耗 CUDA RNG 的序列与参考代码完全对齐。
        if not preloaded_to_gpu:
            print(f'>>> 模型移动到 GPU {gpu_no}...')
            model = model.cuda(gpu_no)
        else:
            # 即使已在 GPU 上，也调用 .cuda() 保持与参考代码一致的 RNG 消耗
            model = model.cuda(gpu_no)

    # ── FP16 权重 casting (v2 模式下与 asc26-wma-opt 参考代码对齐) ──
    if _FORCE_FULL_FP16 and _USE_FP16:
        model = model.half()
        print('>>> WMA_FORCE_FULL_FP16: 整个模型权重已转为 fp16')
    elif _FORCE_FP16_DIFFUSION_ONLY and _USE_FP16:
        model.model = model.model.half()
        print('>>> WMA_FORCE_FP16_DIFFUSION_ONLY: diffusion wrapper 权重已转为 fp16')

    device = get_device_from_parameters(model)

    # 创建 autocast 上下文
    autocast_ctx = _build_autocast_ctx()
    
    # Create shared DDIM sampler ONCE (baseline optimization)
    if args.use_deepcache:
        shared_ddim_sampler = DDIMSamplerDeepCache(
            model,
            use_deepcache=True,
            cache_interval=args.deepcache_interval,
            cache_branch_id=args.deepcache_branch_id
        )
    else:
        shared_ddim_sampler = DDIMSampler(model)
    print(f'>>> Created shared DDIM sampler')

    # NOTE: 不要在这里再次 seed_everything！
    # data.setup() 内部已经调用了 seed_everything(123) 重置 RNG，
    # 到这里 RNG 状态已经与参考代码一致。
    # 如果多 seed 一次会改变采样噪声，导致 PSNR 下降 ~2dB。

    # Run over data
    assert (args.height % 16 == 0) and (
        args.width % 16
        == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"

    # Get latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'>>> Generate {n_frames} frames under each generation ...')
    noise_shape = [args.bs, channels, n_frames, h, w]

    # DeepCache status
    if args.use_deepcache:
        print(f'>>> DeepCache ENABLED: interval={args.deepcache_interval}, branch_id={args.deepcache_branch_id}')
    else:
        print(f'>>> DeepCache DISABLED')

    # Start inference
    for idx in range(0, len(df)):
        sample = df.iloc[idx]

        # Got initial frame path
        init_frame_path = get_init_frame_path(args.prompt_dir, sample)
        ori_fps = float(sample['fps'])

        video_save_dir = args.savedir + f"/inference/sample_{sample['videoid']}"
        os.makedirs(video_save_dir, exist_ok=True)
        os.makedirs(video_save_dir + '/dm', exist_ok=True)
        os.makedirs(video_save_dir + '/wm', exist_ok=True)

        # Load transitions to get the initial state later
        transition_path = get_transition_path(args.prompt_dir, sample)
        with h5py.File(transition_path, 'r') as h5f:
            transition_dict = {}
            for key in h5f.keys():
                transition_dict[key] = torch.tensor(h5f[key][()])
            for key in h5f.attrs.keys():
                transition_dict[key] = h5f.attrs[key]

        # If many, test various frequence control and world-model generation
        for fs in args.frame_stride:

            # For saving imagens in policy
            sample_save_dir = f'{video_save_dir}/dm/{fs}'
            os.makedirs(sample_save_dir, exist_ok=True)
            # For saving environmental changes in world-model
            sample_save_dir = f'{video_save_dir}/wm/{fs}'
            os.makedirs(sample_save_dir, exist_ok=True)
            # For collecting interaction videos
            wm_video = []
            # Initialize observation queues
            cond_obs_queues = {
                "observation.images.top":
                deque(maxlen=model.n_obs_steps_imagen),
                "observation.state": deque(maxlen=model.n_obs_steps_imagen),
                "action": deque(maxlen=args.video_length),
            }
            # Obtain initial frame and state
            start_idx = 0
            model_input_fs = ori_fps // fs
            batch, ori_state_dim, ori_action_dim = prepare_init_input(
                start_idx,
                init_frame_path,
                transition_dict,
                fs,
                data.test_datasets[args.dataset],
                n_obs_steps=model.n_obs_steps_imagen)
            observation = {
                'observation.images.top':
                batch['observation.image'].permute(1, 0, 2,
                                                   3)[-1].unsqueeze(0),
                'observation.state':
                batch['observation.state'][-1].unsqueeze(0),
                'action':
                torch.zeros_like(batch['action'][-1]).unsqueeze(0)
            }
            observation = {
                key: observation[key].to(device, non_blocking=True)
                for key in observation
            }
            # Update observation queues
            cond_obs_queues = populate_queues(cond_obs_queues, observation)

            # Multi-round interaction with the world-model
            # 使用实际迭代次数
            for itr in tqdm(range(actual_n_iter)):
                _PERF_TIMER.start('iteration_total')

                # Get observation
                observation = {
                    'observation.images.top':
                    torch.stack(list(
                        cond_obs_queues['observation.images.top']),
                                dim=1).permute(0, 2, 1, 3, 4),
                    'observation.state':
                    torch.stack(list(cond_obs_queues['observation.state']),
                                dim=1),
                    'action':
                    torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }

                # Use world-model in policy to generate action
                print(f'>>> Step {itr}: generating actions ...')
                _PERF_TIMER.start('action_generation')
                with torch.no_grad(), autocast_ctx:
                    pred_videos_0, pred_actions, _ = image_guided_synthesis_sim_mode(
                        model,
                        sample['instruction'],
                        observation,
                        noise_shape,
                        action_cond_step=args.exe_steps,
                        ddim_steps=action_ddim_steps,  # 使用 action 专用步数
                        ddim_eta=args.ddim_eta,
                        unconditional_guidance_scale=args.
                        unconditional_guidance_scale,
                        fs=model_input_fs,
                        timestep_spacing=args.timestep_spacing,
                        guidance_rescale=args.guidance_rescale,
                        sim_mode=False,
                        ddim_sampler=shared_ddim_sampler,
                        decode_video=False,
                        use_deepcache=args.use_deepcache,
                        deepcache_interval=args.deepcache_interval,
                        deepcache_branch_id=args.deepcache_branch_id)
                _PERF_TIMER.stop('action_generation')

                # Update future actions in the observation queues
                for idx in range(len(pred_actions[0])):
                    observation = {'action': pred_actions[0][idx:idx + 1]}
                    observation['action'][:, ori_action_dim:] = 0.0
                    cond_obs_queues = populate_queues(cond_obs_queues,
                                                      observation)

                # Collect data for interacting the world-model using the predicted actions
                observation = {
                    'observation.images.top':
                    torch.stack(list(
                        cond_obs_queues['observation.images.top']),
                                dim=1).permute(0, 2, 1, 3, 4),
                    'observation.state':
                    torch.stack(list(cond_obs_queues['observation.state']),
                                dim=1),
                    'action':
                    torch.stack(list(cond_obs_queues['action']), dim=1),
                }
                observation = {
                    key: observation[key].to(device, non_blocking=True)
                    for key in observation
                }

                # Interaction with the world-model
                print(f'>>> Step {itr}: interacting with world model ...')
                _PERF_TIMER.start('world_model_interaction')
                with torch.no_grad(), autocast_ctx:
                    pred_videos_1, _, pred_states = image_guided_synthesis_sim_mode(
                        model,
                        "",
                        observation,
                        noise_shape,
                        action_cond_step=args.exe_steps,
                        ddim_steps=args.ddim_steps,
                        ddim_eta=args.ddim_eta,
                        unconditional_guidance_scale=args.
                        unconditional_guidance_scale,
                        fs=model_input_fs,
                        text_input=False,
                        timestep_spacing=args.timestep_spacing,
                        guidance_rescale=args.guidance_rescale,
                        ddim_sampler=shared_ddim_sampler,
                        decode_video=True,
                        use_deepcache=args.use_deepcache,
                        deepcache_interval=args.deepcache_interval,
                        deepcache_branch_id=args.deepcache_branch_id)
                _PERF_TIMER.stop('world_model_interaction')

                for idx in range(args.exe_steps):
                    observation = {
                        'observation.images.top':
                        pred_videos_1[0][:, idx:idx + 1].permute(1, 0, 2, 3),
                        'observation.state':
                        torch.zeros_like(pred_states[0][idx:idx + 1]) if
                        args.zero_pred_state else pred_states[0][idx:idx + 1],
                        'action':
                        torch.zeros_like(pred_actions[0][-1:])
                    }
                    observation['observation.state'][:, ori_state_dim:] = 0.0
                    cond_obs_queues = populate_queues(cond_obs_queues,
                                                      observation)

                # 可选: 保存中间结果到 TensorBoard
                if _TENSORBOARD_ENABLED and writer is not None:
                    sample_tag = f"{args.dataset}-vid{sample['videoid']}-dm-fs-{fs}/itr-{itr}"
                    log_to_tensorboard(writer,
                                       pred_videos_0,
                                       sample_tag,
                                       fps=args.save_fps)
                    sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/itr-{itr}"
                    log_to_tensorboard(writer,
                                       pred_videos_1,
                                       sample_tag,
                                       fps=args.save_fps)

                # 可选: 保存中间视频文件
                if _SAVE_INTERMEDIATE:
                    sample_video_file = f'{video_save_dir}/dm/{fs}/itr-{itr}.mp4'
                    save_results(pred_videos_0.cpu(),
                                 sample_video_file,
                                 fps=args.save_fps)
                    sample_video_file = f'{video_save_dir}/wm/{fs}/itr-{itr}.mp4'
                    save_results(pred_videos_1.cpu(),
                                 sample_video_file,
                                 fps=args.save_fps)

                _PERF_TIMER.stop('iteration_total')
                print('>' * 24)
                # Collect the result of world-model interactions
                wm_video.append(pred_videos_1[:, :, :args.exe_steps].cpu())

            full_video = torch.cat(wm_video, dim=2)
            if _TENSORBOARD_ENABLED and writer is not None:
                sample_tag = f"{args.dataset}-vid{sample['videoid']}-wd-fs-{fs}/full"
                log_to_tensorboard(writer,
                                   full_video,
                                   sample_tag,
                                   fps=args.save_fps)
            sample_full_video_file = f"{video_save_dir}/../{sample['videoid']}_full_fs{fs}.mp4"
            save_results(full_video, sample_full_video_file, fps=args.save_fps)
    
    # 输出性能统计
    _PERF_TIMER.report()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
                        type=str,
                        default=None,
                        help="Path to save the results.")
    parser.add_argument("--ckpt_path",
                        type=str,
                        default=None,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config",
                        type=str,
                        help="Path to the model checkpoint.")
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default=None,
        help="Directory containing videos and corresponding prompts.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="the name of dataset to test")
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps. If non-positive, DDPM is used instead.")
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="Eta for DDIM sampling. Set to 0.0 for deterministic results.")
    parser.add_argument("--bs",
                        type=int,
                        default=1,
                        help="Batch size for inference. Must be 1.")
    parser.add_argument("--height",
                        type=int,
                        default=320,
                        help="Height of the generated images in pixels.")
    parser.add_argument("--width",
                        type=int,
                        default=512,
                        help="Width of the generated images in pixels.")
    parser.add_argument(
        "--frame_stride",
        type=int,
        nargs='+',
        required=True,
        help=
        "frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)"
    )
    parser.add_argument(
        "--unconditional_guidance_scale",
        type=float,
        default=1.0,
        help="Scale for classifier-free guidance during sampling.")
    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Random seed for reproducibility.")
    parser.add_argument("--video_length",
                        type=int,
                        default=16,
                        help="Number of frames in the generated video.")
    parser.add_argument("--num_generation",
                        type=int,
                        default=1,
                        help="seed for seed_everything")
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default="uniform",
        help=
        "Strategy for timestep scaling. See Table 2 in the paper: 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--guidance_rescale",
        type=float,
        default=0.0,
        help=
        "Rescale factor for guidance as discussed in 'Common Diffusion Noise Schedules and Sample Steps are Flawed' (https://huggingface.co/papers/2305.08891)."
    )
    parser.add_argument(
        "--perframe_ae",
        action='store_true',
        default=False,
        help=
        "Use per-frame autoencoder decoding to reduce GPU memory usage. Recommended for models with resolutions like 576x1024."
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="num of samples per prompt",
    )
    parser.add_argument(
        "--exe_steps",
        type=int,
        default=16,
        help="num of samples to execute",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=40,
        help="num of iteration to interact with the world model",
    )
    parser.add_argument("--zero_pred_state",
                        action='store_true',
                        default=False,
                        help="not using the predicted states as comparison")
    parser.add_argument("--save_fps",
                        type=int,
                        default=8,
                        help="fps for the saving video")
    # DeepCache acceleration parameters
    parser.add_argument(
        "--use_deepcache",
        action='store_true',
        default=False,
        help="Enable DeepCache acceleration for faster inference (training-free, ~1.5x speedup)."
    )
    parser.add_argument(
        "--deepcache_interval",
        type=int,
        default=3,
        help="DeepCache: Steps between cache updates. Higher = faster but lower quality. Recommended: 2-5."
    )
    parser.add_argument(
        "--deepcache_branch_id",
        type=int,
        default=0,
        help="DeepCache: Block level to start caching. 0 = most aggressive, higher = more conservative."
    )
    # ===================== 初始化优化参数 =====================
    parser.add_argument(
        "--init_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Model init device. 'gpu' constructs model parameters directly on CUDA."
    )
    parser.add_argument(
        "--checkpoint_load_device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Checkpoint load device. 'gpu' loads checkpoint tensors directly on CUDA."
    )
    parser.add_argument(
        "--preload_ckpt_to_shm",
        action='store_true',
        default=False,
        help="Preload checkpoint to /dev/shm before torch.load for faster IO."
    )
    parser.add_argument(
        "--preload_shm_dir",
        type=str,
        default="/dev/shm/unifolm_wma_ckpts",
        help="Target shm directory for checkpoint preload."
    )
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2**31)
    args.seed = seed
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)
