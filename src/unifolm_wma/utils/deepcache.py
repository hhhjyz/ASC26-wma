"""
DeepCache for WMA (World-Model-Action) Model

This module implements DeepCache acceleration technique for the WMA diffusion model.
DeepCache works by caching high-level features from deeper UNet blocks and reusing them
across multiple timesteps, reducing computational cost.

Reference: "DeepCache: Accelerating Diffusion Models for Free" (Ma et al., 2023)
https://arxiv.org/abs/2312.00858

The WMA model has a UNet structure with:
- input_blocks: Downsampling path (encoder)
- middle_block: Bottleneck
- output_blocks: Upsampling path (decoder)

DeepCache skips computation of deeper blocks during "retrieval" steps and uses
cached outputs from the previous "caching" step.
"""

import torch
from typing import Optional, Dict, Any, Tuple, List
from functools import partial


class DeepCacheWMAHelper:
    """
    DeepCache helper for WMA (World-Model-Action) model.
    
    This class wraps the WMA model's forward pass to implement DeepCache,
    which caches and reuses features from deeper UNet blocks across timesteps.
    
    Args:
        model: The WMAModel (nn.Module) - the actual UNet backbone, 
               NOT the DiffusionWrapper or DDPM model.
        
    Usage:
        # Get the actual WMAModel from the DDPM model hierarchy
        # model.model = DiffusionWrapper
        # model.model.diffusion_model = WMAModel
        wma_model = ddpm_model.model.diffusion_model
        
        helper = DeepCacheWMAHelper(wma_model)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
        
        # Run inference
        output = sampler.sample(...)
        
        helper.disable()
    """
    
    def __init__(self, model=None):
        """
        Initialize DeepCache helper.
        
        Args:
            model: The WMAModel instance (not DiffusionWrapper or DDPM).
                   Should have input_blocks, middle_block, output_blocks attributes.
        """
        self.model = model
        self.enabled = False
        self.reset_states()
        
    def set_params(
        self,
        cache_interval: int = 3,
        cache_branch_id: int = 0,
        skip_mode: str = 'uniform'
    ):
        """
        Set DeepCache parameters.
        
        Args:
            cache_interval: Number of steps between cache updates. 
                           Higher = faster but potentially lower quality.
                           Recommended: 2-5 for good speed/quality trade-off.
            cache_branch_id: Which branch/block level to start caching from.
                            0 = cache from shallowest (most aggressive)
                            Higher = cache from deeper blocks (more conservative)
            skip_mode: How to determine skip steps ('uniform' or 'adaptive')
        """
        self.params = {
            'cache_interval': cache_interval,
            'cache_branch_id': cache_branch_id,
            'skip_mode': skip_mode
        }
        
    def enable(self):
        """Enable DeepCache by wrapping the model's forward method."""
        if self.model is None:
            raise ValueError("Model not set. Initialize with model or call set_model first.")
        
        # Verify model has expected attributes
        if not hasattr(self.model, 'input_blocks'):
            raise ValueError(
                "Model doesn't have 'input_blocks' attribute. "
                "Make sure you're passing WMAModel, not DiffusionWrapper or DDPM. "
                "Use: ddpm_model.model.diffusion_model"
            )
        
        self.reset_states()
        self._wrap_forward()
        self.enabled = True
        print(f"[DeepCache] Enabled with interval={self.params['cache_interval']}, "
              f"branch_id={self.params['cache_branch_id']}")
        
    def disable(self):
        """Disable DeepCache and restore original forward method."""
        if hasattr(self, '_original_forward') and self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        self.reset_states()
        self.enabled = False
        
    def set_model(self, model):
        """Set the model to apply DeepCache to."""
        self.model = model
        
    def reset_states(self):
        """Reset all cached states."""
        self.cur_timestep_idx = 0
        self.start_timestep_idx = None
        self.cached_features = {}
        self._original_forward = None
        
    def _is_cache_step(self) -> bool:
        """Determine if current step should update cache (vs retrieve from cache)."""
        if self.start_timestep_idx is None:
            self.start_timestep_idx = self.cur_timestep_idx
            return True  # First step always caches
            
        cache_interval = self.params['cache_interval']
        if self.params['skip_mode'] == 'uniform':
            # Cache on every cache_interval-th step
            return (self.cur_timestep_idx - self.start_timestep_idx) % cache_interval == 0
        return True
        
    def _should_skip_block(self, block_idx: int, block_type: str = 'input') -> bool:
        """
        Determine if a block should be skipped (use cached output).
        
        Args:
            block_idx: Index of the block
            block_type: 'input', 'middle', or 'output'
            
        Returns:
            True if block should use cached output, False if should compute
        """
        if self._is_cache_step():
            return False  # Don't skip on cache steps
            
        cache_branch_id = self.params['cache_branch_id']
        
        if block_type == 'middle':
            return True  # Always skip middle block on retrieval steps
        elif block_type == 'input':
            # Skip deeper input blocks (higher indices)
            return block_idx > cache_branch_id
        elif block_type == 'output':
            # For output blocks, the indexing is reversed
            # Skip earlier output blocks (which correspond to deeper features)
            num_output_blocks = len(self.model.output_blocks)
            return block_idx < (num_output_blocks - cache_branch_id - 1)
            
        return False
        
    def _wrap_forward(self):
        """Wrap the model's forward method to implement DeepCache."""
        self._original_forward = self.model.forward
        model = self.model
        helper = self  # Reference for closure
        
        def cached_forward(
            x: torch.Tensor,
            x_action: torch.Tensor,
            x_state: torch.Tensor,
            timesteps: torch.Tensor,
            context: Optional[torch.Tensor] = None,
            context_action: Optional[torch.Tensor] = None,
            features_adapter: Any = None,
            fs: Optional[torch.Tensor] = None,
            **kwargs
        ) -> Tuple[torch.Tensor, ...]:
            """
            Forward pass with DeepCache optimization.
            """
            from einops import rearrange
            from unifolm_wma.utils.diffusion import timestep_embedding
            
            # Increment timestep counter
            helper.cur_timestep_idx += 1
            
            b, _, t, _, _ = x.shape
            
            # Time embedding
            t_emb = timestep_embedding(
                timesteps, model.model_channels, repeat_only=False
            ).type(x.dtype)
            emb = model.time_embed(t_emb)
            
            # Process context (same as original)
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
                    context_agent_action = rearrange(
                        context_agent_action.unsqueeze(2), 'b t l d -> (b t) l d'
                    )
                    context_agent_action = model.action_token_projector(context_agent_action)
                    context_agent_action = rearrange(
                        context_agent_action, '(b o) l d -> b o l d', o=t
                    )
                    context_agent_action = rearrange(
                        context_agent_action, 'b o (t l) d -> b o t l d', t=t
                    )
                    context_agent_action = context_agent_action.permute(0, 2, 1, 3, 4)
                    context_agent_action = rearrange(
                        context_agent_action, 'b t o l d -> (b t) (o l) d'
                    )
                    context_text = context[:, model.n_obs_steps + 16:model.n_obs_steps + 16 + 77, :]
                    context_text = context_text.repeat_interleave(repeats=t, dim=0)
                    context_img = context[:, model.n_obs_steps + 16 + 77:, :]
                    context_img = rearrange(context_img, 'b (t l) c -> (b t) l c', t=t)
                    context_agent_state = context_agent_state.repeat_interleave(repeats=t, dim=0)
                    context = torch.cat([
                        context_agent_state, context_agent_action, context_text, context_img
                    ], dim=1)
            
            emb = emb.repeat_interleave(repeats=t, dim=0)
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            
            # FPS embedding
            if model.fs_condition:
                if fs is None:
                    fs = torch.tensor(
                        [model.default_fs] * b, dtype=torch.long, device=x.device
                    )
                fs_emb = timestep_embedding(
                    fs, model.model_channels, repeat_only=False
                ).type(x.dtype)
                fs_embed = model.fps_embedding(fs_emb)
                fs_embed = fs_embed.repeat_interleave(repeats=t, dim=0)
                emb = emb + fs_embed
            
            h = x.type(model.dtype)
            adapter_idx = 0
            hs = []
            hs_a = []
            
            is_cache_step = helper._is_cache_step()
            cache_branch_id = helper.params['cache_branch_id']
            
            # Input blocks with caching
            for id, module in enumerate(model.input_blocks):
                should_skip = helper._should_skip_block(id, 'input')
                
                if should_skip and ('input', id) in helper.cached_features:
                    # Use cached output
                    h = helper.cached_features[('input', id)]
                else:
                    # Compute and optionally cache
                    h = module(h, emb, context=context, batch_size=b)
                    if id == 0 and model.addition_attention:
                        h = model.init_attn(h, emb, context=context, batch_size=b)
                    
                    # Cache if this is a cache step and block is in caching range
                    if is_cache_step and id > cache_branch_id:
                        helper.cached_features[('input', id)] = h.clone()
                
                # Adapter features
                if ((id + 1) % 3 == 0) and features_adapter is not None:
                    h = h + features_adapter[adapter_idx]
                    adapter_idx += 1
                    
                if id != 0:
                    if hasattr(module[0], '__class__') and 'Downsample' in module[0].__class__.__name__:
                        hs_a.append(rearrange(hs[-1], '(b t) c h w -> b t c h w', t=t))
                hs.append(h)
            
            hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))
            
            if features_adapter is not None:
                assert len(features_adapter) == adapter_idx
            
            # Middle block with caching
            if helper._should_skip_block(0, 'middle') and ('middle', 0) in helper.cached_features:
                h = helper.cached_features[('middle', 0)]
            else:
                h = model.middle_block(h, emb, context=context, batch_size=b)
                if is_cache_step:
                    helper.cached_features[('middle', 0)] = h.clone()
            
            hs_a.append(rearrange(h, '(b t) c h w -> b t c h w', t=t))
            
            # Output blocks with caching
            hs_out = []
            num_output_blocks = len(model.output_blocks)
            
            for id, module in enumerate(model.output_blocks):
                h = torch.cat([h, hs.pop()], dim=1)
                
                should_skip = helper._should_skip_block(id, 'output')
                
                if should_skip and ('output', id) in helper.cached_features:
                    # For output blocks, we need special handling due to skip connections
                    # We can't simply reuse cached output because input h changes
                    # So we just compute normally (output block caching is trickier)
                    h = module(h, emb, context=context, batch_size=b)
                else:
                    h = module(h, emb, context=context, batch_size=b)
                
                if hasattr(module[-1], '__class__') and 'Upsample' in module[-1].__class__.__name__:
                    hs_a.append(rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))
                hs_out.append(h)
            
            h = h.type(x.dtype)
            hs_a.append(rearrange(hs_out[-1], '(b t) c h w -> b t c h w', t=t))
            
            # Output projection
            y = model.out(h)
            y = rearrange(y, '(b t) c h w -> b c t h w', b=b)
            
            # Action and state heads (not cached - they depend on full features)
            if not model.base_model_gen_only:
                ba, _, _ = x_action.shape
                a_y = model.action_unet(
                    x_action, timesteps[:ba], hs_a, context_action[:2], **kwargs
                )
                if b > 1:
                    s_y = model.state_unet(
                        x_state, timesteps[:ba], hs_a, context_action[:2], **kwargs
                    )
                else:
                    s_y = model.state_unet(
                        x_state, timesteps, hs_a, context_action[:2], **kwargs
                    )
            else:
                a_y = torch.zeros_like(x_action)
                s_y = torch.zeros_like(x_state)
            
            return y, a_y, s_y
        
        self.model.forward = cached_forward


def apply_deepcache_to_wma(
    ddpm_model,
    cache_interval: int = 3,
    cache_branch_id: int = 0
) -> DeepCacheWMAHelper:
    """
    Convenience function to apply DeepCache to a WMA model.
    
    Args:
        ddpm_model: DDPM model instance (the helper will extract WMAModel from it)
        cache_interval: Steps between cache updates (higher = faster, lower quality)
        cache_branch_id: Which block level to start caching (0 = most aggressive)
        
    Returns:
        DeepCacheWMAHelper instance (call .enable() to activate)
        
    Example:
        helper = apply_deepcache_to_wma(model, cache_interval=3, cache_branch_id=0)
        helper.enable()
        
        # Run inference
        output = sampler.sample(...)
        
        helper.disable()
    """
    # Extract the actual WMAModel from the model hierarchy
    # model.model = DiffusionWrapper
    # model.model.diffusion_model = WMAModel
    if hasattr(ddpm_model, 'model') and hasattr(ddpm_model.model, 'diffusion_model'):
        wma_model = ddpm_model.model.diffusion_model
    elif hasattr(ddpm_model, 'diffusion_model'):
        wma_model = ddpm_model.diffusion_model
    else:
        wma_model = ddpm_model
    
    helper = DeepCacheWMAHelper(wma_model)
    helper.set_params(
        cache_interval=cache_interval,
        cache_branch_id=cache_branch_id
    )
    return helper
