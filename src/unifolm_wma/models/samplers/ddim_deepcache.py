"""
DDIM Sampler with DeepCache Integration

This module provides a DDIM sampler with built-in DeepCache support
for accelerated inference of WMA models.
"""

import torch
import numpy as np
from typing import Optional, Any, Dict, Tuple

from unifolm_wma.models.samplers.ddim import DDIMSampler
from unifolm_wma.utils.deepcache import DeepCacheWMAHelper


class DDIMSamplerDeepCache(DDIMSampler):
    """
    DDIM Sampler with integrated DeepCache acceleration.
    
    This sampler automatically manages DeepCache lifecycle during sampling,
    enabling transparent acceleration without manual enable/disable calls.
    
    Args:
        model: The diffusion model (DDPM)
        schedule: Beta schedule type
        use_deepcache: Whether to enable DeepCache acceleration
        cache_interval: Steps between cache updates (higher = faster)
        cache_branch_id: Block level to start caching (0 = most aggressive)
        **kwargs: Additional arguments for DDIMSampler
        
    Example:
        sampler = DDIMSamplerDeepCache(
            model,
            use_deepcache=True,
            cache_interval=3,
            cache_branch_id=0
        )
        samples, actions, states, _ = sampler.sample(S=50, ...)
    """
    
    def __init__(
        self,
        model,
        schedule: str = "linear",
        use_deepcache: bool = False,
        cache_interval: int = 3,
        cache_branch_id: int = 0,
        **kwargs
    ):
        super().__init__(model, schedule, **kwargs)
        
        self.use_deepcache = use_deepcache
        self.cache_interval = cache_interval
        self.cache_branch_id = cache_branch_id
        self.deepcache_helper = None
        
        if use_deepcache:
            self._init_deepcache()
    
    def _init_deepcache(self):
        """Initialize the DeepCache helper."""
        # Model hierarchy:
        # self.model = DDPM (LatentDiffusion)
        # self.model.model = DiffusionWrapper
        # self.model.model.diffusion_model = WMAModel (the actual UNet)
        wma_model = self.model.model.diffusion_model
        
        self.deepcache_helper = DeepCacheWMAHelper(wma_model)
        self.deepcache_helper.set_params(
            cache_interval=self.cache_interval,
            cache_branch_id=self.cache_branch_id,
            skip_mode='uniform'
        )
    
    def enable_deepcache(
        self,
        cache_interval: Optional[int] = None,
        cache_branch_id: Optional[int] = None
    ):
        """
        Enable or reconfigure DeepCache.
        
        Args:
            cache_interval: Optional new cache interval
            cache_branch_id: Optional new cache branch ID
        """
        if cache_interval is not None:
            self.cache_interval = cache_interval
        if cache_branch_id is not None:
            self.cache_branch_id = cache_branch_id
            
        self.use_deepcache = True
        self._init_deepcache()
    
    def disable_deepcache(self):
        """Disable DeepCache."""
        if self.deepcache_helper is not None:
            self.deepcache_helper.disable()
        self.use_deepcache = False
        self.deepcache_helper = None
    
    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.,
        mask=None,
        x0=None,
        temperature=1.,
        noise_dropout=0.,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        schedule_verbose=False,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
        precision=None,
        fs=None,
        timestep_spacing='uniform',
        guidance_rescale=0.0,
        **kwargs
    ):
        """
        Sample with optional DeepCache acceleration.
        
        DeepCache is automatically enabled/disabled around the sampling loop.
        """
        # Enable DeepCache before sampling
        if self.use_deepcache and self.deepcache_helper is not None:
            self.deepcache_helper.enable()
        
        try:
            # Call parent's sample method
            samples, actions, states, intermediates = super().sample(
                S=S,
                batch_size=batch_size,
                shape=shape,
                conditioning=conditioning,
                callback=callback,
                normals_sequence=normals_sequence,
                img_callback=img_callback,
                quantize_x0=quantize_x0,
                eta=eta,
                mask=mask,
                x0=x0,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                verbose=verbose,
                schedule_verbose=schedule_verbose,
                x_T=x_T,
                log_every_t=log_every_t,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                precision=precision,
                fs=fs,
                timestep_spacing=timestep_spacing,
                guidance_rescale=guidance_rescale,
                **kwargs
            )
        finally:
            # Always disable/reset DeepCache after sampling
            if self.use_deepcache and self.deepcache_helper is not None:
                self.deepcache_helper.disable()
        
        return samples, actions, states, intermediates


def create_deepcache_sampler(
    model,
    cache_interval: int = 3,
    cache_branch_id: int = 0,
    schedule: str = "linear"
) -> DDIMSamplerDeepCache:
    """
    Factory function to create a DeepCache-enabled DDIM sampler.
    
    Args:
        model: The DDPM model
        cache_interval: Steps between cache updates
        cache_branch_id: Block level to start caching
        schedule: Beta schedule type
        
    Returns:
        DDIMSamplerDeepCache instance with DeepCache enabled
        
    Example:
        sampler = create_deepcache_sampler(model, cache_interval=3)
        samples, _, _, _ = sampler.sample(S=50, ...)
    """
    return DDIMSamplerDeepCache(
        model=model,
        schedule=schedule,
        use_deepcache=True,
        cache_interval=cache_interval,
        cache_branch_id=cache_branch_id
    )
