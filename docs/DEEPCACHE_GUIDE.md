# DeepCache for WMA (World-Model-Action) 使用指南

## 概述

DeepCache 是一种无需训练的扩散模型加速技术，通过缓存和复用 UNet 深层特征来减少计算量。本指南介绍如何在 WMA 模型上应用 DeepCache。

## 原理

WMA 模型的 UNet 架构包含：
- `input_blocks`: 下采样路径（编码器）
- `middle_block`: 瓶颈层
- `output_blocks`: 上采样路径（解码器）

DeepCache 的核心思想：
1. **观察**：UNet 深层特征在相邻时间步之间变化较小
2. **优化**：在部分时间步跳过深层块的计算，使用缓存的特征
3. **效果**：可实现 1.5x-2.5x 加速，质量损失很小

```
时间步:  t=1000  t=997   t=994   t=991   t=988   ...
缓存:    [计算]  [复用]  [复用]  [计算]  [复用]  ...
                 ↑       ↑               ↑
              使用 t=1000 的缓存      使用 t=991 的缓存
```

## 安装

DeepCache 实现已集成到项目中，无需额外安装：

```python
from unifolm_wma.utils.deepcache import DeepCacheWMAHelper, apply_deepcache_to_wma
```

## 基本使用

### 方法 1: 使用便捷函数

```python
from unifolm_wma.utils.deepcache import apply_deepcache_to_wma

# 获取 WMA 模型
model = ...  # 你的 WMA 模型实例

# 应用 DeepCache
helper = apply_deepcache_to_wma(
    model,
    cache_interval=3,      # 每3步更新一次缓存
    cache_branch_id=0      # 从第0层开始缓存
)

# 启用
helper.enable()

# 运行推理
output = model(x, x_action, x_state, timesteps, context, ...)

# 禁用（推理完成后）
helper.disable()
```

### 方法 2: 手动配置

```python
from unifolm_wma.utils.deepcache import DeepCacheWMAHelper

helper = DeepCacheWMAHelper(model)

# 设置参数
helper.set_params(
    cache_interval=3,      # 缓存间隔
    cache_branch_id=0,     # 缓存分支 ID
    skip_mode='uniform'    # 跳过模式
)

# 启用
helper.enable()

# ... 推理 ...

# 禁用
helper.disable()
```

## 在 DDIM 采样器中使用

修改采样代码以集成 DeepCache：

```python
from unifolm_wma.utils.deepcache import DeepCacheWMAHelper

class DDIMSamplerWithDeepCache(DDIMSampler):
    def __init__(self, model, schedule="linear", 
                 use_deepcache=False, cache_interval=3, cache_branch_id=0, **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.use_deepcache = use_deepcache
        
        if use_deepcache:
            self.deepcache_helper = DeepCacheWMAHelper(model.model)  # model.model 是 WMAModel
            self.deepcache_helper.set_params(
                cache_interval=cache_interval,
                cache_branch_id=cache_branch_id
            )
    
    @torch.no_grad()
    def sample(self, *args, **kwargs):
        if self.use_deepcache:
            self.deepcache_helper.enable()
        
        try:
            result = super().sample(*args, **kwargs)
        finally:
            if self.use_deepcache:
                self.deepcache_helper.disable()
        
        return result
```

## 在推理脚本中使用

修改 `world_model_interaction.py` 或推理脚本：

```python
# 在模型加载后添加
from unifolm_wma.utils.deepcache import DeepCacheWMAHelper

# 假设 model 是 DDPM 模型
wma_model = model.model  # 获取内部的 WMAModel

# 创建 DeepCache helper
deepcache_helper = DeepCacheWMAHelper(wma_model)
deepcache_helper.set_params(
    cache_interval=args.deepcache_interval,  # 例如 3
    cache_branch_id=args.deepcache_branch_id  # 例如 0
)

# 在采样前启用
deepcache_helper.enable()

# 运行采样
samples, actions, states, _ = ddim_sampler.sample(...)

# 采样后禁用
deepcache_helper.disable()
```

## 参数说明

### `cache_interval`
- **含义**：缓存更新间隔，即每隔多少步重新计算深层特征
- **推荐值**：2-5
- **权衡**：
  - 值越大 → 加速越明显，但质量可能下降
  - 值越小 → 质量越好，但加速效果减弱
- **建议**：从 3 开始尝试

### `cache_branch_id`
- **含义**：从哪个层级开始缓存
- **推荐值**：0-2
- **权衡**：
  - 0 = 最激进，缓存最多层，加速最大
  - 越大 = 越保守，只缓存更深的层
- **建议**：从 0 开始，如质量不佳则增大

### `skip_mode`
- **含义**：跳过步骤的策略
- **选项**：
  - `'uniform'`：均匀跳过
  - `'adaptive'`：自适应（未来扩展）

## 性能预期

| cache_interval | 预期加速比 | 质量影响 |
|----------------|-----------|---------|
| 2              | ~1.3x     | 很小    |
| 3              | ~1.5x     | 小      |
| 4              | ~1.8x     | 中等    |
| 5              | ~2.0x     | 较大    |

**注意**：实际加速比取决于：
- 模型大小
- 硬件配置
- 其他优化（如 FP16）

## 与其他优化配合使用

DeepCache 可与其他优化技术叠加：

```python
# 1. FP16 精度
model = model.half()

# 2. DeepCache
helper = apply_deepcache_to_wma(model.model, cache_interval=3)
helper.enable()

# 3. xformers 加速（如果可用）
# 在 attention 模块中自动使用

# 4. 减少采样步数
samples = ddim_sampler.sample(S=25, ...)  # 原来可能是 50 步
```

## 注意事项

1. **每次推理重置**：每次完整推理后应调用 `helper.disable()` 或 `helper.reset_states()`

2. **内存开销**：缓存会占用额外显存，大约增加 10-20%

3. **不适用于训练**：DeepCache 仅用于推理加速

4. **Action/State Head**：当前实现不缓存 action_unet 和 state_unet 的计算，因为它们依赖完整特征

## 故障排除

### 问题：输出质量明显下降
- 尝试增大 `cache_interval` 的值
- 尝试增大 `cache_branch_id` 的值

### 问题：加速效果不明显
- 确认 DeepCache 已正确启用
- 检查是否有其他瓶颈（如数据加载）

### 问题：内存不足
- 减小 batch size
- 使用 FP16 精度

## 参考

- [DeepCache Paper](https://arxiv.org/abs/2312.00858)
- [DeepCache GitHub](https://github.com/horseee/DeepCache)
