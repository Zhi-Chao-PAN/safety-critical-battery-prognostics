"""
Physics Constraints Abstraction Layer for PINN Battery Prognostics.

Design Principles:
1. Decoupled Architecture: Each constraint is a standalone plugin
2. GPU-Optimized: Batch-first tensor operations for RTX 4060
3. Mixed Precision Ready: Compatible with torch.cuda.amp
4. Extensible: Easy to add new physics constraints

Hardware Alignment: Optimized for Intel Core Ultra 9 + RTX 4060
- Maximize VRAM utilization through batch processing
- Minimize CPU-GPU synchronization overhead
- Leverage Tensor Cores with mixed precision
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Safety Constants ────────────────────────────────────────────
# When NaN/Inf is detected in predictions, return this penalty loss
# instead of 0.0 (which silently disables physics constraints).
# Value must be large enough to dominate the loss landscape and
# force optimizer recovery, but finite to avoid gradient explosion.
NAN_PENALTY_LOSS: float = 100.0

logger = logging.getLogger(__name__)


class PhysicsConstraint(ABC):
    """
    Abstract base class for all physics constraints in battery prognostics.
    
    This class defines the interface for implementing physics-based constraints
    that enforce battery operating principles and safety regulations. All 
    constraint implementations must inherit from this base class and implement
    the compute_loss method.
    
    Attributes:
        name: Unique identifier for the constraint
        base_weight: Static weighting factor for loss calculation
        adaptive: Flag indicating if weight should adjust based on cycle position
        device: Torch device (CPU/GPU) where constraint computations run
    """
    
    def __init__(self, name: str, weight: float = 1.0, adaptive: bool = False):
        self.name: str = name
        self.base_weight: float = weight
        self.adaptive: bool = adaptive
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._nan_count: int = 0  # Track NaN/Inf occurrences for diagnostics
        
    @abstractmethod
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    inputs: Dict[str, torch.Tensor],
                    **kwargs) -> torch.Tensor:
        """
        Compute constraint violation loss.
        
        Args:
            predictions: Model predictions tensor [batch_size, ...]
            inputs: Dictionary of input tensors (cycles, features, etc.)
            **kwargs: Additional context-specific parameters
            
        Returns:
            loss: Scalar loss tensor
        """
        pass
    
    def get_weight(self, 
                  cycles: Optional[torch.Tensor] = None,
                  max_cycle: Optional[float] = None) -> torch.Tensor:
        """
        Get constraint weight (adaptive or static).
        
        Args:
            cycles: Cycle numbers for adaptive weighting
            max_cycle: Maximum cycle for normalization
            
        Returns:
            weight: Scalar or per-sample weight tensor
        """
        if self.adaptive and cycles is not None and max_cycle is not None:
            return self._adaptive_weight(cycles, max_cycle)
        return torch.tensor(self.base_weight, device=self.device)
    
    def _adaptive_weight(self, cycles: torch.Tensor, max_cycle: float) -> torch.Tensor:
        """
        Default adaptive weighting: Sigmoid schedule based on cycle position.
        
        Late cycles (near end-of-life) get higher weight for safety-critical regions.
        """
        # Normalize cycles to [0, 1+] (extrapolation > 1.0)
        t = cycles / max(max_cycle, 1.0)
        
        # Sigmoid schedule: weight increases in late cycles
        k = 10.0  # Transition sharpness
        t_mid = 0.6  # Center of transition
        sigmoid = 1.0 / (1.0 + torch.exp(-k * (t - t_mid)))
        
        # Map from [0, 1] to [base_weight, 2*base_weight]
        weight_min = self.base_weight
        weight_max = 2.0 * self.base_weight
        return weight_min + (weight_max - weight_min) * sigmoid
    
    def validate(self, predictions: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> bool:
        """
        Validate input tensors for numerical stability.
        
        Returns:
            valid: True if inputs are numerically stable
        """
        # Check for NaN/Inf
        if torch.any(torch.isnan(predictions)) or torch.any(torch.isinf(predictions)):
            self._nan_count += 1
            logger.error(
                f"SAFETY: Constraint {self.name}: NaN/Inf detected in predictions "
                f"(occurrence #{self._nan_count}). Physics constraint will apply "
                f"HIGH PENALTY loss instead of being silently disabled."
            )
            return False
        
        # Check for extreme values
        if torch.max(torch.abs(predictions)) > 1e6:
            logger.error(
                f"SAFETY: Constraint {self.name}: Extreme values detected "
                f"(max abs = {torch.max(torch.abs(predictions)):.2e}). "
                f"Applying HIGH PENALTY loss."
            )
            return False
            
        return True
    
    def _nan_penalty_loss(self) -> torch.Tensor:
        """
        Return a high penalty loss when NaN/Inf is detected.
        
        Instead of returning 0.0 (which silently disables physics constraints),
        return a large but finite penalty that forces the optimizer to recover.
        """
        return torch.tensor(NAN_PENALTY_LOSS, device=self.device, requires_grad=False)
    
    def to(self, device: torch.device) -> "PhysicsConstraint":
        """Move constraint to specified device."""
        self.device = device
        return self


class MonotonicityConstraint(PhysicsConstraint):
    """
    Monotonicity Constraint: Capacity should generally decrease over cycles.
    
    Mathematical formulation:
        loss = mean( max(Δcapacity, 0)² )
    where Δcapacity = capacity[i] - capacity[i-1]
    
    Optimized for batch processing on RTX 4060.
    """
    
    def __init__(self, weight: float = 0.05, adaptive: bool = True):
        super().__init__("monotonicity", weight, adaptive)
        
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    inputs: Dict[str, torch.Tensor],
                    **kwargs) -> torch.Tensor:
        """
        Compute monotonicity violation loss.
        
        Args:
            predictions: Capacity predictions [batch_size, seq_len] or [batch_size, 1]
            inputs: Must contain 'cycles' tensor for sequence ordering
            
        Returns:
            loss: Monotonicity violation loss (scalar)
        """
        # Validate inputs
        if not self.validate(predictions, inputs):
            return self._nan_penalty_loss()
        
        # Ensure predictions are 2D [batch_size, seq_len]
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        
        # For sequential predictions, compute differences along sequence dimension
        if predictions.shape[1] > 1:
            # predictions shape: [batch_size, seq_len]
            diffs = predictions[:, 1:] - predictions[:, :-1]  # [batch_size, seq_len-1]
        else:
            # Single prediction per sample — must ensure batch is sorted by cycle.
            # DEFENSIVE: Sort predictions by cycle order from inputs to avoid
            # silent failure when mini-batching or random sampling is introduced.
            cycles = inputs.get("cycles")
            if cycles is not None and cycles.numel() >= predictions.shape[0]:
                cycle_vals = cycles[:predictions.shape[0]].squeeze()
                sort_idx = torch.argsort(cycle_vals)
                predictions_sorted = predictions[sort_idx]
            else:
                predictions_sorted = predictions
            diffs = predictions_sorted[1:] - predictions_sorted[:-1]  # [batch_size-1, 1]
        
        # Guard: no diffs to compute (single sample or single timestep)
        if diffs.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Penalize positive differences (capacity increases)
        violations = F.relu(diffs)  # [batch_size, seq_len-1] or [batch_size-1, 1]
        
        # Quadratic penalty for stronger enforcement
        loss = torch.mean(violations ** 2)
        
        return loss


class SPMResidualConstraint(PhysicsConstraint):
    """
    SPM Residual Constraint: Neural network should learn small residuals.
    
    Mathematical formulation:
        loss = mean( nn_residual² )
    
    This encourages the neural network to only correct the physics model
    where necessary, preventing overfitting to noise.
    """
    
    def __init__(self, weight: float = 0.1, adaptive: bool = True):
        super().__init__("spm_residual", weight, adaptive)
        
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    inputs: Dict[str, torch.Tensor],
                    **kwargs) -> torch.Tensor:
        """
        Compute SPM residual constraint loss.
        
        Args:
            predictions: Neural network residual predictions [batch_size, 1]
            inputs: Additional context (not used in basic residual constraint)
            
        Returns:
            loss: Residual magnitude loss (scalar)
        """
        # Validate inputs
        if not self.validate(predictions, inputs):
            return self._nan_penalty_loss()
        
        # Simple L2 penalty on residuals
        # predictions should be the NN residual (difference from physics baseline)
        loss = torch.mean(predictions ** 2)
        
        return loss


class VoltageConstraint(PhysicsConstraint):
    """
    Capacity Bound Safety Constraint (historically named VoltageConstraint).
    
    Despite its name, this constraint operates on CAPACITY predictions (Ah),
    not raw voltage values. It enforces that predicted capacity stays within
    a physically plausible operating range.
    
    In a capacity-prediction PINN, the model outputs capacity (e.g., 1.0–2.0 Ah).
    The defaults v_min=0.0, v_max=2.5 bound the predictions to this range.
    
    Mathematical formulation:
        loss = mean( max(pred - cap_max, 0)² + max(cap_min - pred, 0)² )
    
    NOTE: If your system predicts actual voltage, override v_min/v_max to
    voltage-appropriate ranges (e.g., 2.5V–4.2V for Li-ion).
    """
    
    def __init__(self, 
                 v_min: float = 0.0, 
                 v_max: float = 2.5,
                 weight: float = 0.02,
                 adaptive: bool = True):
        super().__init__("voltage_safety", weight, adaptive)
        self.v_min = v_min
        self.v_max = v_max
        
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    inputs: Dict[str, torch.Tensor],
                    **kwargs) -> torch.Tensor:
        """
        Compute capacity bound constraint loss.
        
        Args:
            predictions: Capacity predictions [batch_size, 1] (in Ah)
            inputs: Constraint context dict (cycles, features, etc.)
            
        Returns:
            loss: Capacity bound violation loss (scalar)
        """
        # Validate inputs
        if not self.validate(predictions, inputs):
            return self._nan_penalty_loss()
        
        # Penalize predictions outside safe capacity range
        over_bound = F.relu(predictions - self.v_max)   # [batch_size, 1]
        under_bound = F.relu(self.v_min - predictions)   # [batch_size, 1]
        
        # Quadratic penalty
        loss = torch.mean(over_bound ** 2 + under_bound ** 2)
        
        return loss


class TemperatureConstraint(PhysicsConstraint):
    """
    Capacity Upper-Bound Safety Constraint (historically named TemperatureConstraint).
    
    Despite its name, this constraint operates on CAPACITY predictions (Ah)
    in the current system (which predicts capacity, not temperature). It
    enforces that predicted capacity does not exceed the rated maximum.
    
    The default t_max=2.2 corresponds to a generous upper bound for
    a 2.0 Ah rated cell (110% headroom).
    
    Mathematical formulation:
        loss = mean( max(pred - cap_upper, 0)² )
    
    NOTE: If your system predicts actual temperature, override t_max to
    a temperature-appropriate value (e.g., 45°C for Li-ion).
    """
    
    def __init__(self, 
                 t_max: float = 2.2,  # Capacity upper bound (Ah), not Celsius
                 weight: float = 0.01,
                 adaptive: bool = True):
        super().__init__("temperature_safety", weight, adaptive)
        self.t_max = t_max
        
    def compute_loss(self, 
                    predictions: torch.Tensor, 
                    inputs: Dict[str, torch.Tensor],
                    **kwargs) -> torch.Tensor:
        """
        Compute capacity upper-bound constraint loss.
        
        Args:
            predictions: Capacity predictions [batch_size, 1] (in Ah)
            inputs: Constraint context dict (cycles, features, etc.)
            
        Returns:
            loss: Capacity upper-bound violation loss (scalar)
        """
        # Validate inputs
        if not self.validate(predictions, inputs):
            return self._nan_penalty_loss()
        
        # Penalize predictions exceeding upper capacity bound
        over_bound = F.relu(predictions - self.t_max)  # [batch_size, 1]
        
        # Quadratic penalty
        loss = torch.mean(over_bound ** 2)
        
        return loss


class ConstraintManager:
    """
    Manages multiple physics constraints with adaptive weighting.
    
    Features:
    1. Batch-optimized constraint evaluation
    2. Automatic device management
    3. Mixed precision compatibility
    4. Per-constraint validation and logging
    """
    
    def __init__(self, device: str = "cuda"):
        self.constraints: Dict[str, PhysicsConstraint] = {}
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
    def add_constraint(self, constraint: PhysicsConstraint) -> "ConstraintManager":
        """Add a constraint to the manager."""
        constraint = constraint.to(self.device)
        self.constraints[constraint.name] = constraint
        self.logger.info(f"Added constraint: {constraint.name} (weight={constraint.base_weight}, adaptive={constraint.adaptive})")
        return self
    
    def compute_total_loss(self,
                          predictions: torch.Tensor,
                          inputs: Dict[str, torch.Tensor],
                          cycles: Optional[torch.Tensor] = None,
                          max_cycle: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total constraint loss with adaptive weighting.
        
        Args:
            predictions: Model predictions
            inputs: Input tensors for constraints
            cycles: Cycle numbers for adaptive weighting
            max_cycle: Maximum cycle for normalization
            
        Returns:
            total_loss: Weighted sum of all constraint losses
            loss_breakdown: Dictionary of individual constraint losses
        """
        total_loss = torch.tensor(0.0, device=self.device)
        loss_breakdown = {}
        
        for name, constraint in self.constraints.items():
            # Get constraint-specific weight
            weight = constraint.get_weight(cycles, max_cycle)
            
            # Compute constraint loss
            constraint_loss = constraint.compute_loss(predictions, inputs)
            
            # Apply weight
            if torch.is_tensor(weight) and weight.numel() > 1:
                # If constraint_loss is already a scalar but weight is a batch vector,
                # we must average the weighted loss to prevent backward() crash.
                weighted_loss = torch.mean(weight) * constraint_loss
            else:
                weighted_loss = weight * constraint_loss
                
            # Accumulate
            total_loss = total_loss + weighted_loss
            
            # Store breakdown
            loss_breakdown[name] = {
                "loss": constraint_loss.item(),
                "weight": weight.mean().item() if torch.is_tensor(weight) else weight,
                "weighted_loss": weighted_loss.mean().item() if torch.is_tensor(weighted_loss) else weighted_loss
            }
            
            # Log if loss is significant
            if constraint_loss.item() > 1e-3:
                w_log = weight.mean().item() if torch.is_tensor(weight) else weight
                self.logger.debug(f"Constraint {name}: loss={constraint_loss.item():.4f}, "
                                f"weight={w_log:.4f}")
        
        return total_loss, loss_breakdown
    
    def validate_all(self, predictions: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> bool:
        """Validate all constraints for numerical stability."""
        all_valid = True
        for name, constraint in self.constraints.items():
            if not constraint.validate(predictions, inputs):
                self.logger.critical(
                    f"SAFETY-CRITICAL: Constraint {name} validation FAILED. "
                    f"NaN/Inf/extreme values detected. Model outputs may be unsafe."
                )
                all_valid = False
        return all_valid
    
    def to(self, device: torch.device) -> "ConstraintManager":
        """Move all constraints to specified device."""
        self.device = device
        for constraint in self.constraints.values():
            constraint.to(device)
        return self


# Factory function for common constraint configurations
def create_default_constraint_manager(device: str = "cuda") -> ConstraintManager:
    """
    Create a constraint manager with default battery physics constraints.
    
    Default configuration optimized for RTX 4060:
    1. Monotonicity constraint (most important)
    2. SPM residual constraint (physics consistency)
    3. Voltage safety constraint (operational safety)
    4. Temperature safety constraint (thermal safety)
    """
    manager = ConstraintManager(device)
    
    # Add default constraints with optimized weights for RTX 4060
    manager.add_constraint(MonotonicityConstraint(weight=0.05, adaptive=True))
    manager.add_constraint(SPMResidualConstraint(weight=0.1, adaptive=True))
    manager.add_constraint(VoltageConstraint(weight=0.02, adaptive=True))
    manager.add_constraint(TemperatureConstraint(weight=0.01, adaptive=True))
    
    return manager