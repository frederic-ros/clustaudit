"""
Tracing Utilities for Pipeline Execution
=========================================

This module provides tracing functionality to track pipeline execution steps.

Part of clustaudit.
Date: January 2026
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import copy
import time


@dataclass
class TraceStep:
    """
    Single step in the generation trace.
    
    Attributes
    ----------
    name : str
        Name of the pipeline step
    state : object
        Snapshot of the ClusteringState at this step
    info : dict
        Additional information about this step
    timestamp : float
        Timestamp when this step was recorded
    """
    
    name: str
    state: object
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GenerationTrace:
    """
    Complete trace of the generation pipeline.
    
    Attributes
    ----------
    steps : list
        List of TraceStep objects recording each pipeline step
    """
    
    steps: list = field(default_factory=list)
    
    def log(self, name: str, state, info: Optional[Dict[str, Any]] = None):
        """
        Log a pipeline step.
        
        Parameters
        ----------
        name : str
            Name of the step
        state : ClusteringState
            Current state to snapshot
        info : dict, optional
            Additional information to record
        """
        self.steps.append(
            TraceStep(
                name=name,
                state=copy.deepcopy(state),  # IMPORTANT: deep copy
                info=info or {}
            )
        )
    
    def names(self):
        """Return list of step names."""
        return [s.name for s in self.steps]
    
    def get(self, name: str):
        """
        Get all steps with a given name.
        
        Parameters
        ----------
        name : str
            Step name to filter by
            
        Returns
        -------
        list
            List of TraceStep objects matching the name
        """
        return [s for s in self.steps if s.name == name]
