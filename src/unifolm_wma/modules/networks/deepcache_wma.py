"""
DeepCache for WMAModel - 针对 WMA 视频扩散模型的 DeepCache 实现

核心思想：
- 在相邻的扩散步骤中，UNet 深层特征具有时间一致性
- 缓存深层特征，仅更新浅层特征，避免重复计算
- 使用 1:N 非均匀缓存策略

参考：DeepCache: Accelerating Diffusion Models for Free
https://github.com/horseee/DeepCache
"""

import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple
from einops import rearrange


class DeepCacheWMA:
    """
    DeepCache 适配器，用于加速 WMAModel 推理
    
    用法：
        model = WMAModel(...)
        deep_cache = DeepCacheWMA(model)
        deep_cache.enable(cache_interval=3, cache_branch_id=0)
        
        # 在采样循环中
        for i, t in enumerate(timesteps):
            deep_cache.set_timestep(i)
            output = model(x, x_action, x_state, t, ...)
        
        deep_cache.disable()
    """
    
    def __init__(self, model):
        """
        Args:
            model: WMAModel 实例
        """
        self.model = model
        self.enabled = False
        self.reset_cache()
        
    def reset_cache(self):
        """重置所有缓存状态"""
        self.cur_timestep = 0
        self.start_timestep = None
        self.cached_features = {}  # 缓存的中间特征
        self.cached_hs = None      # 缓存的 skip connection 特征
        self.cached_hs_a = None    # 缓存的 action/state head 特征
        self.cached_h_mid = None   # 缓存的 middle block 输出
        self.params = {
            'cache_interval': 3,
            'cache_block_id': 0,
            'cache_layer_id': 0,
            'skip_mode': 'uniform'
        }
        
    def set_params(self, cache_interval: int = 3, cache_branch_id: int = 0, 
                   skip_mode: str = 'uniform'):
        """
        设置缓存参数
        
        Args:
            cache_interval: 缓存间隔，每 N 步重新计算一次
            cache_branch_id: 缓存分支点 ID
                - 0: 从最深层开始缓存 (最激进)
                - 更大的值: 从更浅的层开始缓存
            skip_mode: 跳过模式
                - 'uniform': 均匀间隔
                - 'early': 早期步骤跳过更多
                - 'late': 后期步骤跳过更多
        """
        # 计算 block 和 layer ID
        cache_layer_id = cache_branch_id % 3
        cache_block_id = cache_branch_id // 3
        
        self.params = {
            'cache_interval': cache_interval,
            'cache_block_id': cache_block_id,
            'cache_layer_id': cache_layer_id,
            'skip_mode': skip_mode
        }
        
    def enable(self, cache_interval: int = 3, cache_branch_id: int = 0,
               skip_mode: str = 'uniform'):
        """启用 DeepCache"""
        self.reset_cache()
        self.set_params(cache_interval, cache_branch_id, skip_mode)
        self.enabled = True
        self._wrap_forward()
        
    def disable(self):
        """禁用 DeepCache"""
        if self.enabled:
            self._unwrap_forward()
        self.reset_cache()
        self.enabled = False
        
    def set_timestep(self, timestep_idx: int):
        """设置当前时间步索引"""
        if self.start_timestep is None:
            self.start_timestep = timestep_idx
        self.cur_timestep = timestep_idx
        
    def should_compute_full(self) -> bool:
        """判断是否需要完整计算（不使用缓存）"""
        if self.start_timestep is None:
            return True
            
        cache_interval = self.params['cache_interval']
        skip_mode = self.params['skip_mode']
        
        relative_step = self.cur_timestep - self.start_timestep
        
        if skip_mode == 'uniform':
            return relative_step % cache_interval == 0
        elif skip_mode == 'early':
            # 早期步骤更频繁计算
            if relative_step < 10:
                return relative_step % 2 == 0
            return relative_step % cache_interval == 0
        elif skip_mode == 'late':
            # 后期步骤更频繁计算
            if relative_step > 40:
                return relative_step % 2 == 0
            return relative_step % cache_interval == 0
        else:
            return relative_step % cache_interval == 0
            
    def should_skip_block(self, block_idx: int, layer_idx: int, 
                          block_type: str = "down") -> bool:
        """
        判断是否应该跳过某个 block
        
        Args:
            block_idx: block 索引
            layer_idx: layer 索引  
            block_type: "down", "mid", "up"
        """
        if self.should_compute_full():
            return False
            
        cache_block_id = self.params['cache_block_id']
        cache_layer_id = self.params['cache_layer_id']
        
        if block_type == 'mid':
            return True
            
        if block_idx > cache_block_id:
            return True
        if block_idx < cache_block_id:
            return False
            
        # block_idx == cache_block_id
        if block_type == 'down':
            return layer_idx >= cache_layer_id
        else:  # up
            return layer_idx > cache_layer_id
            
    def _wrap_forward(self):
        """包装 model.forward 方法"""
        self._original_forward = self.model.forward
        self.model.forward = self._cached_forward
        
    def _unwrap_forward(self):
        """恢复原始 forward 方法"""
        if hasattr(self, '_original_forward'):
            self.model.forward = self._original_forward
            
    def _cached_forward(self,
                        x: Tensor,
                        x_action: Tensor,
                        x_state: Tensor,
                        timesteps: Tensor,
                        context: Tensor | None = None,
                        context_action: Tensor | None = None,
                        features_adapter: Any = None,
                        fs: Tensor | None = None,
                        **kwargs) -> Tuple[Tensor, ...]:
        """
        带缓存的 forward 方法
        """
        model = self.model
        
        b, _, t, _, _ = x.shape
        
        # Time embedding
        from unifolm_wma.utils.diffusion import timestep_embedding
        t_emb = timestep_embedding(timesteps,
                                   model.model_channels,
                                   repeat_only=False).type(x.dtype)
        emb = model.time_embed(t_emb)
        
        # Context processing (保持不变)
        bt, l_context, _ = context.shape
        if model.base_model_gen_only:
            assert l_context == 77 + model.n_obs_steps * 16
        else:
            if l_context == model.n_obs_steps + 77 + t * 16:
                context_agent_state = context[:, :model.n_obs_steps]
                context_text = context[:, model.n_obs_steps:model.n_obs_steps + 77, :]
                context_img = context[:, model.n_obs_steps + 77:, :]
                context_agent_state = context_agent_state.repeat_interleave(repeats=t, dim=0)
                context_text = context_text.repeat_interleave(repeats=t, dim=0)
                context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
                context = torch.cat([context_agent_state, context_text, context_img], dim=1)
            elif l_context == model.n_obs_steps + 16 + 77 + t * 16:
                context_agent_state = context[:, :model.n_obs_steps]
                context_agent_action = context[:, model.n_obs_steps:model.n_obs_steps + 16, :]
                context_agent_action = rearrange(context_agent_action.unsqueeze(2), 
                                                  'b t l d -> (b t) l d')
                context_agent_action = model.action_token_projector(context_agent_action)
                context_agent_action = rearrange(context_agent_action, '(b o) l d -> b o l d', o=t)
                context_agent_action = rearrange(context_agent_action, 'b o (t l) d -> b o t l d', t=t)
                context_agent_action = context_agent_action.permute(0, 2, 1, 3, 4)
                context_agent_action = rearrange(context_agent_action, 'b t o l d -> (b t) (o l) d')
                
                context_text = context[:, model.n_obs_steps + 16:model.n_obs_steps + 16 + 77, :]
                context_text = context_text.repeat_interleave(repeats=t, dim=0)
                
                context_img = context[:, model.n_obs_steps + 16 + 77:, :]
                context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
                context_agent_state = context_agent_state.repeat_interleave(repeats=t, dim=0)
                context = torch.cat([context_agent_state, context_agent_action, 
                                    context_text, context_img], dim=1)
        
        emb = emb.repeat_interleave(repeats=t, dim=0)
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        
        # FPS embedding
        if model.fs_condition:
            if fs is None:
                fs = torch.tensor([model.default_fs] * b, dtype=torch.long, device=x.device)
            fs_emb = timestep_embedding(fs, model.model_channels, repeat_only=False).type(x.dtype)
            fs_embed = model.fps_embedding(fs_emb)
            fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
            emb = emb + fs_embed
            
        h = x.type(model.dtype)
        
        # 判断是否需要完整计算
        compute_full = self.should_compute_full()
        
        if compute_full:
            # ========== 完整计算路径 ==========
            adapter_idx = 0
            hs = []
            hs_a = []
            
            # Input blocks (Encoder)
            for id, module in enumerate(model.input_blocks):
                h = module(h, emb, context=context, batch_size=b)
                if id == 0 and model.addition_attention:
                    h = model.init_attn(h, emb, context=context, batch_size=b)
                if ((id + 1) % 3 == 0) and features_adapter is not None:
                    h = h + features_adapter[adapter_idx]
                    adapter_idx += 1
                if id != 0:
                    if isinstance(module[0], Downsample):
                        hs_a.append(rearrange(hs[-1], '(b t) c h w -> b t c h w', t=t))
                hs.append(h)
            hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))
            
            # Middle block
            h = model.middle_block(h, emb, context=context, batch_size=b)
            hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))
            
            # 缓存 middle block 输出和 skip connections
            self.cached_h_mid = h.clone()
            self.cached_hs = [h_i.clone() for h_i in hs]
            
            # Output blocks (Decoder)
            hs_out = []
            for module in model.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context=context, batch_size=b)
                if isinstance(module[-1], Upsample):
                    hs_a.append(rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))
                hs_out.append(h)
            h = h.type(x.dtype)
            hs_a.append(rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))
            
            # 缓存 hs_a 用于 action/state unet
            self.cached_hs_a = [h_i.clone() for h_i in hs_a]
            
        else:
            # ========== 缓存路径 (跳过深层计算) ==========
            cache_block_id = self.params['cache_block_id']
            
            adapter_idx = 0
            hs = []
            hs_a = []
            
            # Input blocks - 只计算到 cache_block_id
            for id, module in enumerate(model.input_blocks):
                if id <= cache_block_id:
                    # 浅层：正常计算
                    h = module(h, emb, context=context, batch_size=b)
                    if id == 0 and model.addition_attention:
                        h = model.init_attn(h, emb, context=context, batch_size=b)
                    if ((id + 1) % 3 == 0) and features_adapter is not None:
                        h = h + features_adapter[adapter_idx]
                        adapter_idx += 1
                else:
                    # 深层：使用缓存
                    h = self.cached_hs[id]
                    
                if id != 0:
                    if isinstance(module[0], Downsample):
                        hs_a.append(rearrange(hs[-1] if id <= cache_block_id else self.cached_hs[id-1], 
                                              '(b t) c h w -> b t c h w', t=t))
                hs.append(h if id <= cache_block_id else self.cached_hs[id])
                
            hs_a.append(rearrange(hs[-1], '(b t) c h w -> b t c h w', t=t))
            
            # Middle block - 使用缓存
            h = self.cached_h_mid
            hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))
            
            # Output blocks - 从缓存恢复
            num_output_blocks = len(model.output_blocks)
            hs_out = []
            for idx, module in enumerate(model.output_blocks):
                # 对于 output blocks，从深到浅
                # 前面的 blocks 使用缓存，后面的重新计算
                skip_idx = num_output_blocks - 1 - idx
                if skip_idx >= cache_block_id:
                    # 使用缓存的 skip connection
                    h = torch.cat([h, self.cached_hs[-(idx+1)]], dim=1)
                else:
                    h = torch.cat([h, hs.pop()], dim=1)
                    
                if skip_idx > cache_block_id:
                    # 跳过深层计算，使用缓存
                    pass  # h 已经是缓存的了
                else:
                    # 重新计算浅层
                    h = module(h, emb, context=context, batch_size=b)
                    
                if isinstance(module[-1], Upsample) and len(hs_out) > 0:
                    hs_a.append(rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))
                hs_out.append(h)
                
            h = h.type(x.dtype)
            hs_a.append(rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))
            
            # 使用缓存的 hs_a
            hs_a = self.cached_hs_a
        
        # Output layer
        y = model.out(h)
        y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
        
        # Action and state prediction
        if not model.base_model_gen_only:
            ba, _, _ = x_action.shape
            a_y = model.action_unet(x_action, timesteps[:ba], hs_a,
                                    context_action[:2], **kwargs)
            if b > 1:
                s_y = model.state_unet(x_state, timesteps[:ba], hs_a,
                                       context_action[:2], **kwargs)
            else:
                s_y = model.state_unet(x_state, timesteps, hs_a,
                                       context_action[:2], **kwargs)
        else:
            a_y = torch.zeros_like(x_action)
            s_y = torch.zeros_like(x_state)
            
        return y, a_y, s_y


class SimpleDeepCacheWMA:
    """
    简化版 DeepCache - 直接缓存整个 UNet 的输出
    
    这是一个更简单但可能更实用的实现：
    - 每 N 步完整计算一次
    - 其他步骤复用上一次的 hs_a（传给 action/state unet 的特征）
    - 仍然计算完整的视频输出
    
    用法：
        cache = SimpleDeepCacheWMA(cache_interval=3)
        
        for i, t in enumerate(timesteps):
            cache.set_step(i)
            if cache.should_update():
                y, a_y, s_y = model(...)
                cache.update(hs_a_features)
            else:
                # 使用缓存的 hs_a
                hs_a = cache.get_cached_hs_a()
    """
    
    def __init__(self, cache_interval: int = 3):
        self.cache_interval = cache_interval
        self.current_step = 0
        self.cached_hs_a = None
        self.cached_output = None
        
    def set_step(self, step: int):
        self.current_step = step
        
    def should_update(self) -> bool:
        """是否需要重新计算"""
        return self.current_step % self.cache_interval == 0
    
    def update(self, hs_a: List[Tensor], output: Tuple[Tensor, ...] = None):
        """更新缓存"""
        self.cached_hs_a = [h.clone() for h in hs_a]
        if output is not None:
            self.cached_output = tuple(o.clone() for o in output)
            
    def get_cached_hs_a(self) -> List[Tensor]:
        """获取缓存的 hs_a"""
        return self.cached_hs_a
    
    def get_cached_output(self) -> Tuple[Tensor, ...]:
        """获取缓存的输出"""
        return self.cached_output
    
    def reset(self):
        """重置缓存"""
        self.current_step = 0
        self.cached_hs_a = None
        self.cached_output = None
