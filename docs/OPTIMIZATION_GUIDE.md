# WMA 推理优化指南

本文档介绍了 WMA (World-Model-Action) 模型推理过程中的各项性能优化措施。

---

## 目录

1. [环境变量配置](#1-环境变量配置)
2. [DDIM 采样器优化](#2-ddim-采样器优化)
3. [注意力机制优化](#3-注意力机制优化)
4. [时间步嵌入优化](#4-时间步嵌入优化)
5. [OpenCLIP 权重复用](#5-openclip-权重复用)
6. [数值稳定性优化](#6-数值稳定性优化)
7. [Context 处理优化](#7-context-处理优化)

---

## 1. 环境变量配置

以下环境变量可用于控制各项优化功能：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ATTN_IMPL_TYPE` | `auto` | 注意力实现类型 (`auto`/`xformers`/`native`) |
| `REUSE_CLIP_WEIGHTS` | `1` | 启用 OpenCLIP 权重复用 |
| `USE_CTX_STORE` | `1` | 启用 Context 处理缓存 |
| `DDIM_TIMING` | `0` | 启用 DDIM 计时统计 |

### 使用示例

```bash
# 使用 PyTorch 原生 SDPA 注意力
export ATTN_IMPL_TYPE=native

# 禁用 CLIP 权重复用（调试用）
export REUSE_CLIP_WEIGHTS=0

# 禁用 Context 缓存
export USE_CTX_STORE=0

# 运行推理
python scripts/evaluation/world_model_interaction.py ...
```

---

## 2. DDIM 采样器优化

**文件**: `src/unifolm_wma/models/samplers/ddim.py`

### 2.1 调度参数缓存

采用 `_DDIMScheduleCache` 类管理调度参数的缓存：

- 通过配置签名（哈希）判断是否需要重新计算
- 签名包含：步数、离散化方法、eta 值、设备信息
- 配置不变时跳过 `make_schedule()` 的重复计算

```python
class _DDIMScheduleCache:
    """DDIM调度参数缓存管理器"""
    __slots__ = ('_cfg_sig', '_dev_str')
    
    def check_valid(self, num_steps, disc_method, eta_val, device):
        """验证缓存是否可用"""
        sig = self.make_signature(num_steps, disc_method, eta_val, device)
        return (self._cfg_sig == sig and self._dev_str == str(device))
```

### 2.2 张量转换工具

`_tensor_convert()` 方法提供统一的转换接口：

- 自动检测模型的数据类型
- 支持混合精度推理场景
- 简化 buffer 注册代码

---

## 3. 注意力机制优化

**文件**: `src/unifolm_wma/modules/attention.py`

### 3.1 多实现后端支持

通过 `ATTN_IMPL_TYPE` 环境变量选择后端：

- `xformers`: 使用 xformers 库的高效实现（默认优先）
- `native`: 使用 PyTorch 原生 `scaled_dot_product_attention`
- `auto`: 自动选择（优先 xformers，不可用时回退 native）

```python
def _select_attn_impl():
    """选择注意力实现后端"""
    impl = os.environ.get("ATTN_IMPL_TYPE", "auto").lower()
    if impl == "auto":
        return "xformers" if XFORMERS_IS_AVAILBLE else "native"
    return impl
```

### 3.2 KV 投影缓存

`_CrossAttnKVStore` 类实现 Cross-Attention 中的 K/V 投影缓存：

- 通过张量 ID 和版本号验证 context 是否变化
- 相同 context 时直接复用缓存的 K/V 投影结果
- 显著减少重复的线性变换计算

---

## 4. 时间步嵌入优化

**文件**: `src/unifolm_wma/utils/diffusion.py`

### 4.1 频率张量缓存

`_SinusoidalFreqStore` 类缓存正弦位置编码的频率张量：

- 缓存键：维度、周期值、设备类型和索引
- 限制最大缓存条目数（16）防止内存泄漏
- 返回副本避免原地修改问题

```python
class _SinusoidalFreqStore:
    """缓存时间步嵌入的频率张量"""
    _data = {}
    _limit = 16
    
    @classmethod
    def fetch(cls, dim_half, period_val, dev):
        """获取频率张量（有缓存时复用）"""
        key = (dim_half, float(period_val), dev.type, 
               dev.index if dev.index else 0)
        ...
```

---

## 5. OpenCLIP 权重复用

**文件**: `src/unifolm_wma/modules/encoders/condition.py`

### 5.1 权重管理器

`_CLIPWeightManager` 实现进程内模型权重复用：

- 首次加载时缓存完整的基础模型
- 后续请求返回深拷贝确保独立性
- 支持按类型裁剪（仅文本/仅视觉）

### 5.2 主要优势

- 减少重复的磁盘 I/O 操作
- 降低内存占用（约 2GB for ViT-H-14）
- 加速模型初始化过程

---

## 6. 数值稳定性优化

### 6.1 V-Prediction 计算优化

**文件**: `src/unifolm_wma/models/ddpms.py`

- `predict_start_from_z_and_v()` 强制使用 FP32 计算
- `predict_eps_from_z_and_v()` 强制使用 FP32 计算
- 避免混合精度下的 NaN/Inf 问题

### 6.2 Spatial Softmax 优化

**文件**: `src/unifolm_wma/models/diffusion_head/base_nets.py`

- softmax 和期望值计算转换为 FP32
- 坐标计算使用 FP32 精度
- 防止注意力权重数值溢出

---

## 7. Context 处理优化

**文件**: `src/unifolm_wma/modules/networks/wma_model.py`

### 7.1 张量签名机制

`_CrossAttnCtxStore` 使用多特征签名判断张量是否变化：

- 张量 ID
- 数据指针
- 版本号
- 形状和设备信息

```python
@staticmethod
def _compute_sig(t):
    """计算张量签名"""
    if t is None:
        return None
    return (
        id(t), t.data_ptr(), t._version,
        tuple(t.shape), t.device.type, ...
    )
```

### 7.2 缓存策略

- 相同 context 和 timestep 时复用格式化结果
- 减少重复的 repeat_interleave、rearrange 操作
- 提升推理吞吐量

---

## 优化效果预期

| 优化项 | 预期收益 |
|--------|----------|
| DDIM 调度缓存 | 减少初始化时间 |
| KV 投影缓存 | 减少 Cross-Attention 计算量 |
| 时间步频率缓存 | 减少重复张量创建 |
| OpenCLIP 权重复用 | 减少内存占用，加速初始化 |
| FP32 数值计算 | 提高数值稳定性 |
| Context 缓存 | 减少张量操作开销 |

---

## 注意事项

1. **缓存失效**: 训练模式下所有缓存自动禁用
2. **内存管理**: 缓存池有大小限制，自动清理
3. **兼容性**: 所有优化可通过环境变量独立开关
4. **调试**: 可通过设置 `DDIM_TIMING=1` 开启性能监控

