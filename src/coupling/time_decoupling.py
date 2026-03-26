"""
Micro-Macro Time-Scale Decoupling Architecture

Resolves the "Time-Scale Black Hole" in Battery PINNs.
- Micro-scale (Seconds/Minutes): Runs SPM/ECM to resolve fast PDE/ODE dynamics inside a SINGLE cycle.
- Macro-scale (Days/Months): Pools intra-cycle states into macro \"Physics Features\" (e.g., max stress, 
  concentration gradients) to feed into the cycle-to-cycle RUL forecasting network (Chronos/TCN).

This cuts the Computational Graph (BPTT) between distinct cycles, strictly preventing 
memory explosion and gradient vanishing while preserving physics-informed bounds.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class PhysicsFeatureExtractor(nn.Module):
    def __init__(self, spm_model: nn.Module, ecm_model: nn.Module = None):
        super().__init__()
        self.spm = spm_model
        self.ecm = ecm_model

    def forward(self,
                i_app_cycle: torch.Tensor,
                dt_micro: float) -> torch.Tensor:
        """
        Executes the physics model over a SINGLE charge/discharge cycle to extract macro-features.
        
        Args:
            i_app_cycle: Current profile for ONE cycle. Shape: [Batch, Micro_Steps]
            dt_micro: Time step in seconds (must strictly obey SPM CFL condition, e.g., < 200s).
            
        Returns:
            physics_features: Shape [Batch, Feature_Dim] representing the aging drivers 
                              (max concentration gradient, accumulated stress, etc.) for this exact cycle.
        """
        batch_size, micro_steps = i_app_cycle.shape
        device = i_app_cycle.device

        # 1. Initialize micro-states (rest state)
        # Using SPM N shells to match the configured FDM dimension
        c_anode = torch.ones((batch_size, self.spm.N), device=device) * 25000.0  # mock initial conc
        c_cathode = torch.ones((batch_size, self.spm.N), device=device) * 10000.0

        max_grad_anode = torch.zeros(batch_size, device=device)
        accumulated_stress = torch.zeros(batch_size, device=device)

        # 2. Intra-cycle Forward Euler (Safely bounded by micro_steps, NOT lifetime)
        for t in range(micro_steps):
            j_n = i_app_cycle[:, t].unsqueeze(-1)  # Mock conversion of I to flux

            c_anode, c_cathode = self.spm(c_anode, c_cathode, j_n, -j_n, dt_micro)

            # 3. Feature Extraction: We care about DEGRADATION DRIVERS, not tracking every second.
            # Driver 1: Maximum Spatial Concentration Gradient (leads to particle cracking)
            grad_anode = torch.abs(c_anode[:, -1] - c_anode[:, 0])
            max_grad_anode = torch.max(max_grad_anode, grad_anode)

            # Driver 2: Accumulated strain/stress energy over the cycle
            # (Simplified proxy: proportional to flux magnitude and concentration shifts)
            stress_t = torch.abs(j_n.squeeze(-1)) * grad_anode
            accumulated_stress += stress_t * dt_micro

        # 4. Pack Macro Features for the RUL Network
        # These features summarize the entire cycle's physical abuse into a bounded vector.
        # Shape: [Batch, 2]
        physics_features = torch.stack([max_grad_anode, accumulated_stress], dim=-1)

        return physics_features
