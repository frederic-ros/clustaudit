"""
Hypercube Sampler: Sample cluster centers on hypercube vertices
================================================================

This module provides functionality to sample cluster centers on the vertices
of a {-1, +1}^d hypercube with controlled separation.

Part of clustaudit.
Date: January 2026
"""

import numpy as np
import warnings
from typing import Optional


def sample_hypercube_centers(
    dimension: int,
    n_clusters: int,
    p_connect: float,
    min_hamming_ratio: float = 0.3,
    random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Sample K cluster centers on {-1, +1}^d hypercube vertices using hybrid strategy.
    
    For low dimensions (d <= 15), uses exact enumeration.
    For high dimensions (d > 15), uses hybrid construction + sampling.
    
    Parameters
    ----------
    dimension : int
        Ambient dimension d
    n_clusters : int
        Number of clusters K
    p_connect : float
        Connectivity probability (0-1, higher = more separated)
    min_hamming_ratio : float, default=0.3
        Minimum Hamming distance ratio between centers (0-1)
    random_state : np.random.RandomState, optional
        Random state for reproducibility
        
    Returns
    -------
    centers : np.ndarray, shape (K, d)
        Cluster centers on hypercube vertices
        
    Raises
    ------
    ValueError
        If n_clusters > 2^dimension
    """
    if random_state is None:
        random_state = np.random.RandomState()
    
    n_vertices = 2 ** dimension
    if n_clusters > n_vertices:
        raise ValueError(f"Cannot sample {n_clusters} from {n_vertices} vertices")
    
    min_hamming = int(min_hamming_ratio * dimension)
    
    # ========================================================================
    # LOW DIMENSION (d <= 15): Exact enumeration
    # ========================================================================
    if dimension <= 15:
        # Generate all vertices
        vertices = np.array([[(i >> j) & 1 for j in range(dimension)] 
                             for i in range(n_vertices)], dtype=float)
        vertices = 2 * vertices - 1
        
        centers = [vertices[random_state.choice(n_vertices)]]
        available = list(range(n_vertices))
        available.remove(np.where((vertices == centers[0]).all(axis=1))[0][0])
        
        for _ in range(n_clusters - 1):
            candidates = [v_idx for v_idx in available
                          if all(np.sum(vertices[v_idx] != c) >= min_hamming for c in centers)]
            
            if not candidates:
                candidates = available
            
            if len(candidates) == 1:
                chosen_idx = candidates[0]
            else:
                distances = np.array([np.mean([np.sum(vertices[v_idx] != c) for c in centers])
                                      for v_idx in candidates])
                weights = distances ** p_connect
                weights /= weights.sum()
                chosen_idx = random_state.choice(candidates, p=weights)
            
            centers.append(vertices[chosen_idx])
            available.remove(chosen_idx)
        
        return np.array(centers)
    
    # ========================================================================
    # HIGH DIMENSION (d > 15): Hybrid construction + sampling
    # ========================================================================
    
    # Step 1: First center - random
    centers = [random_state.choice([-1, 1], size=dimension)]
    
    # Step 2: Subsequent centers - hybrid approach
    for i in range(n_clusters - 1):
        center_found = False
        
        # PHASE 1: CONSTRUCTION (50 attempts)
        for attempt in range(50):
            # Choose random base center
            base_center = centers[random_state.randint(len(centers))]
            
            # Calculate number of bits to flip
            min_flips = min_hamming
            max_flips = dimension
            target_flips = int(min_flips + p_connect * (max_flips - min_flips))
            
            # Add randomness
            n_flips = random_state.randint(
                max(min_flips, target_flips - max(1, dimension // 10)),
                min(max_flips, target_flips + max(1, dimension // 10)) + 1
            )
            
            # Build new center
            new_center = base_center.copy()
            indices_to_flip = random_state.choice(dimension, size=n_flips, replace=False)
            new_center[indices_to_flip] *= -1
            
            # Verify Hamming constraint
            if all(np.sum(new_center != c) >= min_hamming for c in centers):
                centers.append(new_center)
                center_found = True
                break
        
        # PHASE 2: SAMPLING if construction failed
        if not center_found:
            for attempt in range(1000):
                new_center = random_state.choice([-1, 1], size=dimension)
                
                if all(np.sum(new_center != c) >= min_hamming for c in centers):
                    centers.append(new_center)
                    center_found = True
                    break
        
        # PHASE 3: Last resort (relaxed)
        if not center_found:
            warnings.warn(
                f"Could not find center {i+2}/{n_clusters} with min_hamming_ratio={min_hamming_ratio}. "
                f"Adding random center."
            )
            new_center = random_state.choice([-1, 1], size=dimension)
            centers.append(new_center)
    
    return np.array(centers)
