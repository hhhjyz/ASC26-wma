# UniFoLM World Model Action - 项目架构深度分析

## 📊 项目整体架构概览

```
unifolm-world-model-action/
├── 🎯 核心模型层 (src/unifolm_wma/)
│   ├── models/          # 主模型实现
│   ├── modules/         # 可复用模块（Attention等）
│   ├── data/            # 数据加载和预处理
│   └── utils/           # 工具函数
│
├── 📝 配置文件 (configs/)
│   ├── train/           # 训练配置
│   └── inference/       # 推理配置
│
├── 🔧 脚本 (scripts/)
│   ├── trainer.py       # 训练脚本
│   └── evaluation/      # 评估脚本
│
├── 💾 权重文件 (ckpts/)
│   └── unifolm_wma_dual.ckpt
│
└── 📂 数据和结果 (unitree_*/)
    └── case*/           # 各个测试用例
```

---

## 🏗️ 核心架构：五大组件

### 1️⃣ **LatentVisualDiffusion (主模型)**
📍 位置: `src/unifolm_wma/models/ddpms.py`

这是整个系统的顶层模型，继承自 PyTorch Lightning Module。

```python
class LatentVisualDiffusion(DDPM):
    """
    潜在空间视觉扩散模型
    
    核心职责：
    1. 管理扩散过程（加噪、去噪）
    2. 协调各个子模块
    3. 处理训练和推理逻辑
    """
    
    def __init__(self, ...):
        # 子模块初始化
        self.model = WMAModel(...)           # 核心扩散 UNet
        self.first_stage_model = VAE(...)    # 视频编解码器
        self.cond_stage_model = CLIP(...)    # 文本编码器
        self.img_cond_stage = CLIPVision(...)  # 图像编码器
        self.diffusion_policy_head = ...     # 动作预测头
```

**数据流向：**
```
输入视频 
  ↓
[VAE Encoder] → 潜在表示 z (压缩到 1/8 尺寸)
  ↓
[加噪声] → z_t (添加时间步 t 的噪声)
  ↓
[WMAModel] → 预测噪声 ε 或 z_0
  ↓
[去噪] → 恢复的潜在表示
  ↓
[VAE Decoder] → 生成的视频
```

---

### 2️⃣ **WMAModel (World Model Action)**
📍 位置: `src/unifolm_wma/modules/networks/wma_model.py`

这是核心的 3D UNet 架构，负责在潜在空间进行扩散去噪。

#### 架构组成：

```python
class WMAModel(nn.Module):
    """
    时空扩散 UNet
    
    结构：
    - input_blocks:  编码路径 (下采样)
    - middle_block:  瓶颈层
    - output_blocks: 解码路径 (上采样)
    """
    
    def __init__(self):
        # === 1. 时间嵌入层 ===
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        # === 2. 输入卷积 ===
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])
        
        # === 3. 下采样路径 ===
        # 包含：
        # - ResBlock (残差块)
        # - SpatialTransformer (空间注意力)
        # - TemporalTransformer (时间注意力)
        # - Downsample (下采样)
        
        # === 4. 中间层 ===
        self.middle_block = TimestepEmbedSequential(
            ResBlock(...),
            SpatialTransformer(...),   # 空间自注意力
            TemporalTransformer(...),  # 时间自注意力
            ResBlock(...),
        )
        
        # === 5. 上采样路径 ===
        # 包含：
        # - ResBlock + Skip Connection
        # - SpatialTransformer
        # - TemporalTransformer
        # - Upsample (上采样)
        
        # === 6. 输出层 ===
        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        
        # === 7. 动作预测头（可选）===
        if not base_model_gen_only:
            self.unet_head = ConditionalUnet1D(...)  # 1D UNet for action
```

#### 前向传播流程：

```python
def forward(self, x, timesteps, context=None, ...):
    """
    x: [B, C=4, T=16, H=40, W=64] - 潜在表示
    timesteps: [B] - 扩散时间步
    context: [B, 77, 1024] - 文本特征
    """
    
    # 1. 时间嵌入
    t_emb = self.time_embed(timestep_embedding(timesteps, ...))
    # t_emb: [B, 1280]
    
    # 2. 编码路径
    hs = []  # 存储中间特征用于跳跃连接
    h = x
    for module in self.input_blocks:
        h = module(h, t_emb, context)
        hs.append(h)
    # h: [B, C, T, H↓, W↓] - 逐渐下采样
    
    # 3. 瓶颈层
    h = self.middle_block(h, t_emb, context)
    # h: [B, C_max, T, H_min, W_min]
    
    # 4. 解码路径（带跳跃连接）
    for module in self.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)  # Skip connection
        h = module(h, t_emb, context)
    # h: [B, C, T, H, W] - 恢复到原始尺寸
    
    # 5. 输出预测
    h = self.out(h)
    # h: [B, 4, 16, 40, 64]
    
    # 6. 动作预测（如果需要）
    if self.unet_head is not None:
        action = self.unet_head(obs, ...)
        return h, action
    
    return h
```

---

### 3️⃣ **Attention 模块（核心计算）**
📍 位置: `src/unifolm_wma/modules/attention.py`

这是整个模型的计算核心！包含多种注意力机制。

#### A. **CrossAttention (交叉注意力)**

```python
class CrossAttention(nn.Module):
    """
    多头交叉注意力机制
    
    用途：
    1. 文本-图像交叉注意力（Text → Video）
    2. 图像-图像交叉注意力（Image → Video）
    3. 状态-视频交叉注意力（State → Video）
    """
    
    def __init__(self, 
                 query_dim,           # 查询维度 (来自视频特征)
                 context_dim,         # 键值维度 (来自文本/图像)
                 heads=8,             # 注意力头数
                 dim_head=64,         # 每个头的维度
                 image_cross_attention=True,  # 是否启用图像交叉注意力
                 ...):
        
        self.to_q = nn.Linear(query_dim, heads * dim_head)   # Q 投影
        self.to_k = nn.Linear(context_dim, heads * dim_head) # K 投影
        self.to_v = nn.Linear(context_dim, heads * dim_head) # V 投影
        self.to_out = nn.Linear(heads * dim_head, query_dim)
        
    def forward(self, x, context=None, mask=None):
        """
        x: [B*T, H*W, C] - 查询 (视频特征)
        context: [B, L, C_ctx] - 上下文 (文本/图像特征)
        
        计算流程：
        1. Q = x @ W_q
        2. K = context @ W_k  
        3. V = context @ W_v
        4. Attention = Softmax(Q @ K^T / sqrt(d)) @ V
        """
        
        b, n, c = x.shape
        
        # 1. 线性投影
        q = self.to_q(x)  # [B*T, H*W, heads * dim_head]
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 2. 重排为多头格式
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        # q,k,v: [B, heads, N, dim_head]
        
        # 3. 计算注意力分数
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # sim: [B, heads, H*W, L] - 每个像素对每个token的注意力
        
        # 4. Softmax 归一化
        attn = sim.softmax(dim=-1)
        
        # 5. 加权求和
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # out: [B, heads, H*W, dim_head]
        
        # 6. 合并多头并输出
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```

**注意力计算位置：**
1. **SpatialTransformer** 中的 `CrossAttention`
   - 文本引导：每个视频patch关注文本token
   - 图像引导：每个patch关注图像特征
   
2. **TemporalTransformer** 中的 `CrossAttention`
   - 时间维度的自注意力
   - 跨帧信息传递

#### B. **SpatialTransformer (空间注意力块)**

```python
class SpatialTransformer(nn.Module):
    """
    2D 空间注意力变换器
    
    处理空间维度 (H, W) 的注意力
    """
    
    def __init__(self, in_channels, n_heads, d_head, context_dim):
        self.norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, 1)
        
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head, 
                context_dim=context_dim
            )
            for _ in range(depth)
        ])
        
        self.proj_out = nn.Conv2d(inner_dim, in_channels, 1)
        
    def forward(self, x, context=None):
        """
        x: [B*T, C, H, W] - 视频帧特征
        context: [B, L, C_ctx] - 文本/图像条件
        
        流程：
        1. 投影到 transformer 维度
        2. 展平空间维度 [B*T, C, H, W] → [B*T, H*W, C]
        3. 应用多层 Transformer Block
        4. 恢复空间维度并投影回原始通道数
        """
        b, c, h, w = x.shape
        x_in = x
        
        # Normalize and project
        x = self.norm(x)
        x = self.proj_in(x)  # [B*T, inner_dim, H, W]
        
        # Reshape: [B*T, C, H, W] → [B*T, H*W, C]
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, context=context)
        
        # Reshape back and project out
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        
        return x + x_in  # Residual connection
```

#### C. **TemporalTransformer (时间注意力块)**

```python
class TemporalTransformer(nn.Module):
    """
    1D 时间注意力变换器
    
    处理时间维度 (T) 的注意力
    """
    
    def forward(self, x, context=None):
        """
        x: [B, C, T, H, W] - 视频特征
        
        流程：
        1. 重排维度：将时间维度移到序列位置
        2. 在时间维度上应用自注意力
        3. 允许不同帧之间交互
        """
        b, c, t, h, w = x.shape
        x_in = x
        
        # Reshape: [B, C, T, H, W] → [B*H*W, T, C]
        x = rearrange(x, 'b c t h w -> (b h w) t c')
        
        # Temporal self-attention
        for block in self.transformer_blocks:
            x = block(x, context=None)  # 纯自注意力
        
        # Reshape back
        x = rearrange(x, '(b h w) t c -> b c t h w', b=b, h=h, w=w)
        
        return x + x_in
```

---

### 4️⃣ **VAE (视频自编码器)**
📍 位置: `src/unifolm_wma/models/autoencoder.py`

```python
class AutoencoderKL(nn.Module):
    """
    变分自编码器 - 压缩视频到潜在空间
    
    压缩比: 8x (空间) × 1x (时间)
    输入: [B, 3, T, 320, 512]
    输出: [B, 4, T, 40, 64]
    """
    
    def __init__(self):
        # 编码器
        self.encoder = Encoder(
            in_channels=3,
            out_channels=4,
            ch_mult=[1, 2, 4, 4],  # 逐层通道倍增
            num_res_blocks=2
        )
        
        # 解码器
        self.decoder = Decoder(
            in_channels=4,
            out_channels=3,
            ch_mult=[1, 2, 4, 4]
        )
        
        # 潜在空间映射
        self.quant_conv = nn.Conv2d(8, 8, 1)  # μ 和 σ
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        
    def encode(self, x):
        """
        x: [B, 3, T, 320, 512] - 原始视频
        returns: z ~ N(μ, σ) - 潜在表示
        """
        # 逐帧编码（perframe_ae=True）
        b, c, t, h, w = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        h = self.encoder(x)  # [B*T, 8, 40, 64]
        moments = self.quant_conv(h)  # [B*T, 8, 40, 64]
        
        # 分离均值和方差
        mu, logvar = torch.chunk(moments, 2, dim=1)
        # mu, logvar: [B*T, 4, 40, 64]
        
        # 重参数化采样
        posterior = DiagonalGaussianDistribution(mu, logvar)
        z = posterior.sample()  # [B*T, 4, 40, 64]
        
        z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
        return z  # [B, 4, T, 40, 64]
        
    def decode(self, z):
        """
        z: [B, 4, T, 40, 64] - 潜在表示
        returns: [B, 3, T, 320, 512] - 重建视频
        """
        b, c, t, h, w = z.shape
        z = rearrange(z, 'b c t h w -> (b t) c h w')
        
        z = self.post_quant_conv(z)
        dec = self.decoder(z)  # [B*T, 3, 320, 512]
        
        dec = rearrange(dec, '(b t) c h w -> b c t h w', b=b, t=t)
        return dec
```

---

### 5️⃣ **条件编码器组**
📍 位置: `src/unifolm_wma/modules/encoders/`

#### A. **文本编码器**

```python
class FrozenOpenCLIPEmbedder(nn.Module):
    """
    冻结的 OpenCLIP 文本编码器
    模型: ViT-H-14 (laion2b_s32b_b79k)
    """
    
    def __init__(self):
        from transformers import CLIPTokenizer, CLIPTextModel
        self.tokenizer = CLIPTokenizer.from_pretrained("...")
        self.transformer = CLIPTextModel.from_pretrained("...")
        self.freeze()  # 冻结参数
        
    def encode(self, text):
        """
        text: List[str] - 文本提示词
        returns: [B, 77, 1024] - 文本特征
        """
        tokens = self.tokenizer(
            text, 
            padding="max_length", 
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        outputs = self.transformer(**tokens)
        z = outputs.last_hidden_state  # [B, 77, 1024]
        return z
```

#### B. **图像编码器**

```python
class FrozenOpenCLIPImageEmbedderV2(nn.Module):
    """
    冻结的 OpenCLIP 图像编码器
    用于 image-to-video 条件
    """
    
    def forward(self, image):
        """
        image: [B, 3, 224, 224] - 输入图像
        returns: [B, 257, 1280] - 图像特征
        """
        # 使用 CLIP ViT 编码
        z = self.model.encode_image(image)
        # z: [B, 257, 1280] - 包含 CLS token + patch tokens
        return z
```

#### C. **图像投影器 (Resampler)**

```python
class Resampler(nn.Module):
    """
    Perceiver Resampler - 将图像特征投影到文本空间
    
    作用：将 CLIP 图像特征 [B, 257, 1280] 
         投影为固定数量的 query tokens [B, 16, 1024]
    """
    
    def __init__(self):
        self.latents = nn.Parameter(torch.randn(num_queries, dim))
        # num_queries=16 个可学习的 query
        
        self.layers = nn.ModuleList([
            PerceiverAttention(dim, heads, dim_head)
            for _ in range(depth)  # depth=4
        ])
        
    def forward(self, x):
        """
        x: [B, 257, 1280] - CLIP 图像特征
        returns: [B, 16, 1024] - 投影后的特征
        """
        b = x.shape[0]
        
        # 扩展 learnable queries
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        
        # 多层 cross-attention
        for attn in self.layers:
            latents = attn(latents, context=x)
            # latents 从 x 中提取信息
        
        return latents  # [B, 16, 1024]
```

---

### 6️⃣ **动作预测头 (Diffusion Policy)**
📍 位置: `src/unifolm_wma/models/diffusion_head/conditional_unet1d.py`

```python
class ConditionalUnet1D(nn.Module):
    """
    1D 条件 UNet - 用于预测机器人动作
    
    输入：
    - noisy_action: [B, T=16, A=16] - 带噪声的动作序列
    - timestep: [B] - 扩散时间步
    - obs_features: [B, obs_dim] - 观察特征（来自 ResNet18）
    - imagen_features: 来自 WMAModel 的视频特征
    
    输出：
    - pred_action: [B, T=16, A=16] - 预测的动作
    """
    
    def __init__(self):
        # 观察编码器
        self.obs_encoder = MultiImageObsEncoder(
            rgb_model=ResNet18,
            use_spatial_softmax=True
        )
        
        # 1D UNet 结构
        self.down_modules = nn.ModuleList([...])  # 下采样
        self.mid_modules = nn.ModuleList([...])   # 中间层
        self.up_modules = nn.ModuleList([...])    # 上采样
        
        # Cross-attention with imagen features
        self.cross_attn = ActionLatentImageCrossAttention(...)
        
    def forward(self, 
                sample,          # [B, T, A] - noisy action
                timestep,        # [B] - diffusion step
                obs,             # Dict with 'agentview_image', 'state'
                imagen_features  # From WMAModel
                ):
        """
        扩散式动作预测
        
        流程：
        1. 编码观察（图像 + 状态）
        2. 时间步嵌入
        3. 1D UNet 去噪
        4. 与 imagen 特征做交叉注意力
        5. 输出预测的噪声
        """
        
        # 1. 观察编码
        obs_features = self.obs_encoder(
            obs['agentview_image']  # [B, n_obs, 3, H, W]
        )
        # obs_features: [B, obs_dim]
        
        # 2. 时间步嵌入
        timesteps_emb = self.diffusion_step_encoder(timestep)
        
        # 3. 条件拼接
        global_cond = torch.cat([obs_features, timesteps_emb], dim=-1)
        
        # 4. 1D UNet forward
        x = sample.permute(0, 2, 1)  # [B, A, T]
        
        # 下采样路径
        h = []
        for module in self.down_modules:
            x = module(x, global_cond)
            h.append(x)
        
        # 中间层 + 交叉注意力
        for module in self.mid_modules:
            x = module(x, global_cond)
        
        # 与 imagen 特征交叉注意力
        if imagen_features is not None:
            x = self.cross_attn(x, imagen_features)
        
        # 上采样路径
        for module in self.up_modules:
            x = torch.cat([x, h.pop()], dim=1)
            x = module(x, global_cond)
        
        x = x.permute(0, 2, 1)  # [B, T, A]
        return x
```

---

## 🔄 完整数据流（Training）

```
┌─────────────────────────────────────────────────────────────┐
│                    训练数据流                                 │
└─────────────────────────────────────────────────────────────┘

1. 输入批次
   ├─ video: [B, 3, 16, 320, 512]
   ├─ text: ["a robot arm stacking boxes", ...]
   ├─ actions: [B, 16, 16]  # 机器人动作序列
   └─ states: [B, 16, 16]   # 机器人状态

2. VAE 编码
   video → Encoder → z ~ N(μ, σ)
   z: [B, 4, 16, 40, 64]  # 潜在表示

3. 添加噪声（扩散前向过程）
   t ~ Uniform(0, 1000)
   z_t = √(α_t) * z_0 + √(1-α_t) * ε
   ε ~ N(0, I)

4. 条件编码
   ├─ text → CLIPTextEncoder → text_features: [B, 77, 1024]
   ├─ first_frame → CLIPVisionEncoder → img_features: [B, 257, 1280]
   └─ img_features → Resampler → img_tokens: [B, 16, 1024]

5. 去噪网络
   WMAModel(
       z_t,              # 噪声潜在表示
       t,                # 时间步
       text_features,    # 文本条件
       img_tokens        # 图像条件
   ) → ε_pred, action_pred

6. 损失计算
   ├─ imagen_loss = MSE(ε, ε_pred)           # 图像生成损失
   └─ action_loss = MSE(action, action_pred) # 动作预测损失
   
   total_loss = imagen_loss + action_loss

7. 反向传播
   optimizer.step()
```

---

## 🎮 完整数据流（Inference - World Model Interaction）

```
┌─────────────────────────────────────────────────────────────┐
│                交互式推理数据流                               │
└─────────────────────────────────────────────────────────────┘

初始化:
├─ obs_queue: 保存最近 n_obs_steps 帧观察
├─ action_queue: 保存动作历史
└─ current_frame: 当前帧索引

循环 (每一步):

1. 准备观察
   obs_imgs = obs_queue.get_last_n(n_obs_steps)  # 最近2帧
   # obs_imgs: [2, 3, 320, 512]

2. 文本条件 (固定)
   text_prompt = "stack the green box on top of the red box"
   text_features = CLIP.encode(text_prompt)

3. 图像条件
   img_features = CLIPVision.encode(obs_imgs[0])
   img_tokens = Resampler(img_features)

4. 生成潜在表示
   a. 采样随机噪声
      z_T ~ N(0, I): [1, 4, 16, 40, 64]
   
   b. DDIM 采样（50步）
      for t in reversed(range(0, 1000, 20)):  # 50 steps
          # 预测噪声
          ε_pred = WMAModel(z_t, t, text_features, img_tokens)
          
          # 更新 z
          z_{t-1} = DDIM_step(z_t, ε_pred, t)
      
      z_0 = z_{t-1}  # 最终潜在表示

5. 解码生成视频
   generated_video = VAE.decode(z_0)
   # generated_video: [1, 3, 16, 320, 512]

6. 预测动作
   noisy_action ~ N(0, I): [1, 16, 16]
   
   for t in diffusion_steps:
       action_noise = DiffusionPolicyHead(
           noisy_action,
           t,
           obs={'image': obs_imgs, 'state': current_state},
           imagen_features=wma_mid_features
       )
       noisy_action = DDPM_step(noisy_action, action_noise, t)
   
   pred_actions = noisy_action  # [1, 16, 16]

7. 执行动作
   action_to_execute = pred_actions[0]  # 只执行第一个动作
   
   # 在模拟器/真实机器人上执行
   new_obs = env.step(action_to_execute)

8. 更新队列
   obs_queue.append(new_obs)
   action_queue.append(action_to_execute)
   current_frame += 1

9. 记录结果
   save_video(generated_video, f"pred_{current_frame}.mp4")
   log_action(action_to_execute)

重复步骤 1-9，直到任务完成
```

---

## 📊 Attention 计算全景图

```
┌──────────────────────────────────────────────────────────┐
│              Attention 计算位置总览                        │
└──────────────────────────────────────────────────────────┘

WMAModel 结构:
├─ Input Blocks (4个)
│  ├─ ResBlock
│  ├─ SpatialTransformer  ←── ⚡ 空间注意力
│  │   └─ CrossAttention (Text → Video)
│  ├─ TemporalTransformer ←── ⚡ 时间注意力
│  │   └─ SelfAttention (Frame ↔ Frame)
│  └─ Downsample
│
├─ Middle Block
│  ├─ ResBlock
│  ├─ SpatialTransformer  ←── ⚡ 空间注意力
│  │   ├─ CrossAttention (Text → Video)
│  │   └─ CrossAttention (Image → Video)  # 如果有图像条件
│  ├─ TemporalTransformer ←── ⚡ 时间注意力
│  └─ ResBlock
│
├─ Output Blocks (4个)
│  ├─ ResBlock + Skip Connection
│  ├─ SpatialTransformer  ←── ⚡ 空间注意力
│  ├─ TemporalTransformer ←── ⚡ 时间注意力
│  └─ Upsample
│
└─ UNet Head (动作预测)
   └─ ActionLatentImageCrossAttention ←── ⚡ 动作-图像交叉注意力

总计：
- 空间交叉注意力: ~12 次 (每个 block)
- 时间自注意力: ~12 次
- 动作交叉注意力: 1 次
```

### Attention 详细计算

#### 1. **Spatial CrossAttention (空间交叉注意力)**

```python
# 在 SpatialTransformer 中
# 位置: input_blocks[i], middle_block, output_blocks[i]

输入:
- x: [B*T, H*W=2560, C=320]  # 视频特征（展平）
- context: [B, 77, 1024]      # 文本特征

计算:
Q = x @ W_q           # [B*T, 2560, 320]
K = context @ W_k     # [B, 77, 320]
V = context @ W_v     # [B, 77, 320]

# 需要广播 K, V 到所有帧
K = K.unsqueeze(1).repeat(1, T, 1, 1).flatten(0, 1)
V = V.unsqueeze(1).repeat(1, T, 1, 1).flatten(0, 1)
# K, V: [B*T, 77, 320]

# 多头注意力
Q = Q.view(B*T, 2560, heads=8, dim_head=40)
K = K.view(B*T, 77, 8, 40)
V = V.view(B*T, 77, 8, 40)

sim = einsum('b h i d, b h j d -> b h i j', Q, K) / sqrt(40)
# sim: [B*T, 8, 2560, 77]
# 每个像素对每个文本 token 的注意力分数

attn = softmax(sim, dim=-1)
out = einsum('b h i j, b h j d -> b h i d', attn, V)
# out: [B*T, 8, 2560, 40]

out = out.flatten(1, 2)  # [B*T, 2560, 320]

意义:
- 每个视频 patch 关注文本描述的相关部分
- 例如：物体区域关注 "green box"，背景关注 "table"
```

#### 2. **Temporal SelfAttention (时间自注意力)**

```python
# 在 TemporalTransformer 中

输入:
- x: [B*H*W=2560, T=16, C=320]  # 重排后的特征

计算:
Q = x @ W_q    # [2560, 16, 320]
K = x @ W_k
V = x @ W_v

sim = einsum('n i d, n j d -> n i j', Q, K) / sqrt(d)
# sim: [2560, 16, 16]
# 每个空间位置，帧与帧之间的注意力

attn = softmax(sim, dim=-1)
out = einsum('n i j, n j d -> n i d', attn, V)

意义:
- 同一空间位置的不同帧之间交互
- 捕捉运动信息、时间连续性
- 例如：物体在 t=0 和 t=5 的位置关联
```

#### 3. **Image CrossAttention (图像交叉注意力)**

```python
# 只在 middle_block 中，当有图像条件时

输入:
- x: [B*T, H*W, C]
- img_tokens: [B, 16, 1024]  # 来自 Resampler

计算:
类似文本交叉注意力，但：
- context 是图像 tokens 而非文本
- 通常权重更大 (image_cross_attention_scale=1.0)

意义:
- 生成的视频与输入图像保持一致
- 例如：保持物体外观、场景布局
```

#### 4. **Action-Image CrossAttention (动作-图像交叉注意力)**

```python
# 在 ConditionalUnet1D 的 mid_modules 中

输入:
- action_features: [B, A=16, T=16]  # 动作特征（1D）
- imagen_features: [B, C=320, T=16, H=40, W=64]  # 来自 WMAModel

计算:
# 先展平 imagen_features
img_flat = imagen_features.flatten(2, 4)
# img_flat: [B, 320, 40960]

Q = action_features @ W_q  # [B, 16, 320]
K = img_flat.transpose(1, 2) @ W_k  # [B, 40960, 320]
V = img_flat.transpose(1, 2) @ W_v

sim = Q @ K.transpose(-2, -1) / sqrt(d)
# sim: [B, 16, 40960]
# 每个动作维度关注整个视频

attn = softmax(sim, dim=-1)
out = attn @ V  # [B, 16, 320]

意义:
- 动作预测基于生成的视频想象
- 闭环：视频引导动作，动作影响视频
```

---

## 🔧 关键技术细节

### 1. **时间步嵌入 (Timestep Embedding)**

```python
# src/unifolm_wma/utils/diffusion.py

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    正弦位置编码
    
    timesteps: [B] - 时间步 0~1000
    dim: 嵌入维度 (通常 320)
    
    returns: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * 
        torch.arange(half) / half
    )
    # freqs: [160]
    
    args = timesteps[:, None].float() * freqs[None]
    # args: [B, 160]
    
    embedding = torch.cat([
        torch.cos(args), 
        torch.sin(args)
    ], dim=-1)
    # embedding: [B, 320]
    
    return embedding
```

### 2. **扩散调度 (Diffusion Schedule)**

```python
# src/unifolm_wma/utils/diffusion.py

def make_beta_schedule(
    schedule="linear",
    n_timestep=1000,
    linear_start=1e-4,
    linear_end=2e-2,
):
    """
    创建噪声调度
    
    linear: β_t = linear_start + t/T * (linear_end - linear_start)
    cosine: β_t = ... (更平滑)
    """
    if schedule == "linear":
        betas = np.linspace(
            linear_start**0.5, 
            linear_end**0.5, 
            n_timestep
        )**2
    
    # 计算 α_t = 1 - β_t
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    
    return betas, alphas_cumprod
```

### 3. **DDIM 采样 (加速推理)**

```python
# src/unifolm_wma/models/samplers/ddim.py

def ddim_sampling_step(
    z_t,          # 当前噪声状态
    t,            # 当前时间步
    t_next,       # 下一个时间步
    eps_pred,     # 预测的噪声
    alphas_cumprod,
):
    """
    DDIM 确定性采样步骤
    
    公式:
    z_{t-1} = √(α_{t-1}) * x_0_pred + 
              √(1-α_{t-1}) * eps_pred
    """
    alpha_t = alphas_cumprod[t]
    alpha_t_next = alphas_cumprod[t_next]
    
    # 预测 x_0
    x_0_pred = (z_t - sqrt(1-alpha_t) * eps_pred) / sqrt(alpha_t)
    
    # 确定性更新
    z_t_next = (
        sqrt(alpha_t_next) * x_0_pred + 
        sqrt(1 - alpha_t_next) * eps_pred
    )
    
    return z_t_next
```

---

## 📈 性能优化技术

### 1. **XFormers 高效注意力**

```python
# src/unifolm_wma/modules/attention.py

if XFORMERS_IS_AVAILBLE:
    def efficient_forward(self, x, context):
        """
        使用 xformers.ops.memory_efficient_attention
        
        优势:
        - 减少显存占用
        - 加速计算（Flash Attention）
        - 支持长序列
        """
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        out = xformers.ops.memory_efficient_attention(
            q, k, v, 
            attn_bias=None,
            scale=self.scale
        )
        
        return self.to_out(out)
```

### 2. **梯度检查点 (Gradient Checkpointing)**

```python
# 在 WMAModel 中
use_checkpoint = True  # 配置项

if use_checkpoint:
    h = checkpoint(module, h, emb, context)
else:
    h = module(h, emb, context)
    
# 作用：
# - 节省显存（不保存中间激活）
# - 代价：略微增加计算时间（重新计算）
```

### 3. **混合精度训练**

```python
# configs/train/config.yaml
lightning:
  precision: 16  # 使用 FP16

# 效果:
# - 减少显存占用 ~50%
# - 加速训练 ~2x
# - 需要注意数值稳定性
```

---

## 🎯 总结：关键组件职责

| 组件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **LatentVisualDiffusion** | 顶层协调器 | 视频+文本+动作 | 损失值 |
| **WMAModel** | 核心扩散 UNet | 噪声潜在表示 + 条件 | 预测噪声 |
| **SpatialTransformer** | 空间注意力 | 视频特征 | 文本引导的特征 |
| **TemporalTransformer** | 时间注意力 | 视频帧序列 | 时序关联特征 |
| **CrossAttention** | 注意力计算 | Query + Context | 加权特征 |
| **AutoencoderKL** | 视频压缩 | 原始视频 | 潜在表示 |
| **CLIPTextEncoder** | 文本编码 | 文本提示 | 文本特征 |
| **CLIPVisionEncoder** | 图像编码 | 输入图像 | 图像特征 |
| **Resampler** | 特征投影 | CLIP特征 | Token序列 |
| **ConditionalUnet1D** | 动作预测 | 观察+噪声 | 预测动作 |

---

## 💡 理解项目的关键点

1. **潜在空间扩散**
   - 不直接在像素空间操作（太大）
   - 先用 VAE 压缩 8 倍
   - 在压缩空间做扩散（更高效）

2. **多尺度处理**
   - UNet 下采样：捕捉粗粒度结构
   - UNet 上采样：恢复细节
   - Skip connections：保留高频信息

3. **多模态条件**
   - 文本：语义级指导
   - 图像：外观级约束
   - 状态：精确位姿信息

4. **统一框架**
   - 同时做视频生成和动作预测
   - 共享视频表示（WMAModel 特征）
   - 端到端训练

5. **注意力是核心**
   - Spatial：文本→视频的语义对齐
   - Temporal：帧间的运动建模
   - Cross：动作←→视频的闭环

这就是整个 UniFoLM World Model Action 项目的完整架构！🎉
