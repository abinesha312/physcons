"""
Patch-based router inspired by Shodh-MoE.

Routes patches of the input field to different "experts" (simple MLPs).
The routing is deterministic and local, based on patch statistics.
"""

import numpy as np
from typing import List, Tuple


class SimpleExpert:
    """
    A tiny expert network (just a linear layer for this toy example).
    In a real system, this would be a more complex model.
    """
    
    def __init__(self, input_dim: int, seed: int):
        rng = np.random.RandomState(seed)
        # Simple linear transformation + bias
        self.weight = rng.randn(input_dim, input_dim) * 0.01
        self.bias = rng.randn(input_dim) * 0.01
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the expert transformation."""
        return x @ self.weight + self.bias


class PatchRouter:
    """
    Patch-based router that splits the field into patches and routes
    each patch to one of several experts.
    
    Routing is deterministic based on patch mean (local statistics).
    """
    
    def __init__(
        self,
        patch_size: int = 10,
        n_experts: int = 3,
        seed: int = 42
    ):
        """
        Args:
            patch_size: Number of points per patch
            n_experts: Number of expert networks
            seed: Random seed for expert initialization
        """
        self.patch_size = patch_size
        self.n_experts = n_experts
        
        # Initialize experts
        self.experts = [
            SimpleExpert(patch_size, seed + i)
            for i in range(n_experts)
        ]
        
    def _split_into_patches(self, field: np.ndarray) -> List[np.ndarray]:
        """Split field into patches."""
        n_patches = len(field) // self.patch_size
        patches = []
        
        for i in range(n_patches):
            start = i * self.patch_size
            end = start + self.patch_size
            patches.append(field[start:end])
            
        # Handle remainder if field length not divisible by patch_size
        remainder = len(field) % self.patch_size
        if remainder > 0:
            patches.append(field[-remainder:])
            
        return patches
    
    def _route_patch(self, patch: np.ndarray) -> int:
        """
        Deterministically route a patch to an expert based on its mean.
        This is a simple routing strategy; real systems would be more sophisticated.
        """
        patch_mean = patch.mean()
        # Use mean to select expert (simple modulo-based routing)
        # This ensures deterministic routing based on local statistics
        expert_idx = int(abs(patch_mean) * 1000) % self.n_experts
        return expert_idx
    
    def forward(self, field: np.ndarray) -> np.ndarray:
        """
        Route patches through experts.
        
        Args:
            field: 1D input field
            
        Returns:
            output: Processed field (same shape as input)
        """
        patches = self._split_into_patches(field)
        output_patches = []
        
        for patch in patches:
            # Route to expert
            expert_idx = self._route_patch(patch)
            
            # Process through expert (only if full patch)
            if len(patch) == self.patch_size:
                processed = self.experts[expert_idx].forward(patch)
            else:
                # For remainder patches, just pass through
                processed = patch
                
            output_patches.append(processed)
        
        # Concatenate patches back together
        output = np.concatenate(output_patches)
        return output
