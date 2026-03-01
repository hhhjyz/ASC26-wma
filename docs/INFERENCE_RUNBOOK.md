# WMA 推理优化运行手册

本文档说明如何使用优化后的 WMA 推理代码。

---

## 快速开始

### 单个 Case 运行

```bash
cd /path/to/unifolm-world-model-action

# 运行单个 case
bash unitree_g1_pack_camera/case1/run_world_model_interaction.sh
```

### 批量运行（推荐，复用模型）

```bash
# 使用批量运行器，避免每个 case 重新加载模型
python scripts/evaluation/batch_inference_runner.py \
    --base_dir . \
    --gpu 0 \
    --seed 123 \
    --output batch_results.json
```

---

## 环境变量配置

### 性能开关

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `WMA_ENABLE_TB` | `1` | TensorBoard 日志开关 |
| `WMA_SAVE_INTERMEDIATE` | `1` | 中间结果视频保存开关 |
| `WMA_PROFILING` | `0` | 性能分析开关 |
| `WMA_SKIP_INIT` | `1` | 跳过模块参数初始化 |
| `WMA_TORCH_COMPILE` | `0` | 使用 torch.compile |

### 迭代控制

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `WMA_ITER_DELTA` | `0` | 迭代次数调整量 (+/-) |
| `WMA_ITER_OVERRIDE` | `` | 迭代次数绝对覆盖 |
| `WMA_ACTION_DDIM_STEPS` | `` | Action 分支独立 ddim_steps |

### 模型内部优化

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ATTN_IMPL_TYPE` | `auto` | 注意力后端 (auto/xformers/native) |
| `USE_KV_STORE` | `1` | KV 投影缓存 |
| `REUSE_CLIP_WEIGHTS` | `1` | OpenCLIP 权重复用 |
| `USE_CTX_STORE` | `1` | Context 处理缓存 |
| `DDIM_TIMING` | `0` | DDIM 采样计时 |

---

## 运行示例

### 1. 最快速度运行（禁用非必要 I/O）

```bash
export WMA_ENABLE_TB=0
export WMA_SAVE_INTERMEDIATE=0

bash iter_case.sh
```

### 2. 性能分析模式

```bash
export WMA_PROFILING=1
export DDIM_TIMING=1

python scripts/evaluation/world_model_interaction.py \
    --config configs/inference/wma_video.yaml \
    --ckpt_path ckpts/wma_model.pt \
    ...
```

### 3. 调整迭代次数

```bash
# 减少 5 次迭代
export WMA_ITER_DELTA=-5

# 或者直接设置为固定值
export WMA_ITER_OVERRIDE=35

bash unitree_g1_pack_camera/case1/run_world_model_interaction.sh
```

### 4. 使用原生 PyTorch 注意力

```bash
export ATTN_IMPL_TYPE=native

bash iter_case.sh
```

---

## 批量执行脚本说明

`batch_inference_runner.py` 的主要优势：

1. **模型复用**: 多个 case 共享同一个模型实例
2. **数据集复用**: 避免重复加载数据集配置
3. **独立种子**: 每个 case 使用独立的随机种子，防止随机数流漂移

```python
# 核心机制
for idx, spec in enumerate(case_specs):
    case_seed = base_seed + idx * 1000  # 独立种子
    seed_everything(case_seed)
    # 复用 cached_model 和 cached_data
```

---

## 性能优化清单

1. ✅ DDIM 调度参数缓存
2. ✅ 注意力后端选择 (xformers/native SDPA)
3. ✅ KV 投影缓存
4. ✅ 时间步嵌入频率缓存
5. ✅ OpenCLIP 权重复用
6. ✅ Context 处理缓存
7. ✅ 数值稳定性优化 (FP32 for v-prediction)
8. ✅ 模块初始化跳过
9. ✅ 批量执行模型复用
10. ✅ TensorBoard / 中间结果开关

---

## 常见问题

### Q: PSNR 结果不一致？

确保每个 case 使用独立的随机种子：

```bash
# 使用 batch_inference_runner.py 会自动处理
# 或者手动在 shell 脚本中设置 --seed
```

### Q: 内存不足？

尝试禁用缓存：

```bash
export USE_KV_STORE=0
export REUSE_CLIP_WEIGHTS=0
export USE_CTX_STORE=0
```

### Q: 如何查看性能瓶颈？

启用性能分析：

```bash
export WMA_PROFILING=1
export DDIM_TIMING=1
```

运行后会输出各阶段耗时统计。

---

## 结果目录结构

```
results/
├── psnr_results/
│   ├── unitree_g1_pack_camera_case1_psnr.json
│   ├── ...
│   └── psnr_summary_YYYYMMDD-HHMMSS.json
├── inference/
│   ├── sample_0/
│   │   ├── dm/  # 决策模型视频
│   │   └── wm/  # 世界模型视频
│   └── ...
└── tensorboard/
```
