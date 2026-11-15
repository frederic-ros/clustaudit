"""
ClusteringState: STATE Object for Synthetic Clustering Datasets
================================================================

This module defines the ClusteringState dataclass that encapsulates
all information about a generated synthetic clustering dataset.

Part of clustaudit.
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime


@dataclass
class ClusteringState:
    """
    STATE object S containing all dataset information.
    
    This object encapsulates the complete state of a synthetic clustering
    dataset, including points, labels, masks, and metadata.
    
    Attributes
    ----------
    Z : np.ndarray, shape (N, d)
        Data points in d-dimensional space
    y : np.ndarray, shape (N,)
        Cluster labels (-1 for noise)
    masks : Dict[str, np.ndarray]
        Point category masks (core, boundary, noise, etc.)
    entropy : np.ndarray, shape (N,)
        Pointwise entropy values
    confidence : np.ndarray, shape (N,)
        Classifier confidence values
    metadata : Dict[str, Any]
        Generation metadata and parameters
    timestamp : datetime
        Creation timestamp
    random_seed : Optional[int]
        Random seed used for generation
    """
    
    Z: np.ndarray
    y: np.ndarray
    masks: Dict[str, np.ndarray]
    entropy: np.ndarray
    confidence: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime
    random_seed: Optional[int]
    
    @classmethod
    def empty(cls, random_seed: Optional[int] = None) -> 'ClusteringState':
        """
        Initialize empty STATE.
        
        Parameters
        ----------
        random_seed : int, optional
            Random seed for generation
            
        Returns
        -------
        ClusteringState
            Empty state object
        """
        return cls(
            Z=np.empty((0, 0)),
            y=np.empty(0, dtype=int),
            masks={
                'core': np.empty(0, dtype=bool),
                'boundary': np.empty(0, dtype=bool),
                'noise': np.empty(0, dtype=bool),
                'internal_noise': np.empty(0, dtype=bool),
                'distant_noise': np.empty(0, dtype=bool)
            },
            entropy=np.empty(0),
            confidence=np.empty(0),
            metadata={},
            timestamp=datetime.now(),
            random_seed=random_seed
        )
    
    def n_samples(self) -> int:
        """Return number of samples."""
        return len(self.y)
    
    def n_clusters(self) -> int:
        """Return number of clusters (excluding noise)."""
        return len(np.unique(self.y[self.y >= 0]))
    
    def dimension(self) -> int:
        """Return dimensionality."""
        return self.Z.shape[1] if len(self.Z) > 0 else 0
    
    def core_points(self) -> np.ndarray:
        """Return indices of core points."""
        return np.where(self.masks['core'])[0]
    
    def boundary_points(self) -> np.ndarray:
        """Return indices of boundary points."""
        return np.where(self.masks['boundary'])[0]
    
    def noise_points(self) -> np.ndarray:
        """Return indices of noise points."""
        return np.where(self.masks['noise'])[0]
    
    def summary(self) -> Dict[str, Any]:
        """
        Return summary statistics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing dataset statistics
        """
        return {
            'n_samples': self.n_samples(),
            'n_clusters': self.n_clusters(),
            'dimension': self.dimension(),
            'n_core': np.sum(self.masks['core']),
            'n_boundary': np.sum(self.masks['boundary']),
            'n_noise': np.sum(self.masks['noise']),
            'n_internal_noise': np.sum(self.masks['internal_noise']),
            'n_distant_noise': np.sum(self.masks['distant_noise']),
            'mean_entropy': np.mean(self.entropy) if len(self.entropy) > 0 else 0.0,
            'mean_confidence': np.mean(self.confidence) if len(self.confidence) > 0 else 0.0
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ClusteringState(n_samples={self.n_samples()}, "
                f"n_clusters={self.n_clusters()}, "
                f"dimension={self.dimension()})")
