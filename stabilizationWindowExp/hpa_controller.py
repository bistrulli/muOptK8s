#!/usr/bin/env python3
"""
HPAController module for HPA simulation.
Horizontal Pod Autoscaler controller with stabilization window.
"""

import simpy
from typing import TYPE_CHECKING, Optional
from math import ceil
from collections import deque

if TYPE_CHECKING:
    from application_server import ApplicationServer


class HPAController:
    """
    Horizontal Pod Autoscaler controller with stabilization window.
    Stabilization applies to DOWNSCALING ONLY: take the maximum desired replicas
    in the downscale window to avoid premature/oscillatory downscales.
    """
    
    def __init__(self, env: simpy.Environment, application_server: 'ApplicationServer',
                 target_utilization: float, evaluation_interval: float, 
                 upscale_stabilization_window: float,
                 downscale_stabilization_window: float,
                 min_replicas: int = 1,
                 max_replicas: Optional[int] = None,
                 tolerance: float = 0.1,
                 utilization_window: float = 60.0):
        self.env = env
        self.application_server = application_server
        self.target_utilization = target_utilization
        self.evaluation_interval = evaluation_interval
        self.upscale_stabilization_window = upscale_stabilization_window
        self.downscale_stabilization_window = downscale_stabilization_window
        self.min_replicas = max(1, int(min_replicas))
        self.max_replicas = int(max_replicas) if max_replicas is not None else None
        self.tolerance = max(0.0, float(tolerance))
        self.utilization_window = max(0.0, float(utilization_window))
        
        # Track scaling recommendations within the stabilization windows
        self.upscale_recommendations = deque()
        self.downscale_recommendations = deque()
        self.scaling_decisions = []  # Track all decisions for analysis
        self.last_scaling_time = -float('inf')
        self.pending_scaling = None  # Track if scaling is in progress

    def _compute_utilization_fraction(self) -> float:
        """Compute CPU utilization fraction using a recent window for responsiveness."""
        if self.utilization_window > 0:
            busy_cores = self.application_server.get_recent_cpu_utilization(self.utilization_window)
        else:
            busy_cores = self.application_server.get_avg_cpu_utilization()
        capacity = getattr(self.application_server.cpu_resource, 'capacity', 0) or 0
        if capacity <= 0:
            return 0.0
        return float(busy_cores) / float(capacity)
        
    def _get_stabilized_replicas(self, current_replicas: int, desired_replicas: int) -> int:
        """Apply stabilization window logic only for downscale (use max in window)."""
        current_time = self.env.now
        
        is_upscale = desired_replicas > current_replicas
        is_downscale = desired_replicas < current_replicas
        
        if is_upscale:
            # Upscaling: NO stabilization window, act immediately
            return desired_replicas
                
        elif is_downscale:
            # Downscaling: use downscale stabilization window
            self.downscale_recommendations.append({
                'time': current_time,
                'desired_replicas': desired_replicas
            })
            
            # Remove old recommendations outside downscale window
            while (self.downscale_recommendations and 
                   current_time - self.downscale_recommendations[0]['time'] > self.downscale_stabilization_window):
                self.downscale_recommendations.popleft()
            
            # Get maximum desired replicas in downscale window (conservative slow-down)
            if self.downscale_recommendations:
                max_desired = max(rec['desired_replicas'] for rec in self.downscale_recommendations)
                window_size = len(self.downscale_recommendations)
                print(f"[{current_time:.1f}s] DOWNSCALE STABILIZATION: {window_size} recommendations in {self.downscale_stabilization_window}s window, "
                      f"max desired: {max_desired}")
                return max_desired
        
        return desired_replicas
        
    def run(self) -> simpy.Process:
        """Main control loop - evaluates metrics and makes scaling decisions."""
        while True:
            yield self.env.timeout(self.evaluation_interval)
            
            # Observe current metrics
            # Utilization as fraction of capacity
            utilization_fraction = self._compute_utilization_fraction()
            current_replicas = self.application_server.current_replicas
            # Use EFFECTIVE replicas (active capacity / cpus_per_replica) for HPA formula to avoid
            # compounding on target replicas that are not active yet due to warm-up
            active_cores = getattr(self.application_server.cpu_resource, 'capacity', 0) or 0
            cpus_per_replica = getattr(self.application_server, 'cpus_per_replica', 1) or 1
            effective_replicas = max(1, int((active_cores + cpus_per_replica - 1) // cpus_per_replica))
            
            # Calculate desired replicas using HPA formula
            if utilization_fraction <= 0.0 or self.target_utilization <= 0.0:
                desired_replicas = current_replicas
            else:
                # Respect tolerance band: if within [1-tol, 1+tol], do not scale
                ratio = utilization_fraction / self.target_utilization
                within_tolerance = (1.0 - self.tolerance) <= ratio <= (1.0 + self.tolerance)
                if within_tolerance:
                    desired_replicas = current_replicas
                else:
                    desired_replicas = ceil(effective_replicas * ratio)
            
            # Enforce min/max bounds
            desired_replicas = max(self.min_replicas, desired_replicas)
            if self.max_replicas is not None:
                desired_replicas = min(self.max_replicas, desired_replicas)
            
            # Apply stabilization window after clamping
            stabilized_replicas = self._get_stabilized_replicas(current_replicas, desired_replicas)
            
            # Record scaling decision for analysis
            decision = {
                'time': self.env.now,
                'current_replicas': current_replicas,
                'effective_replicas': effective_replicas,
                'desired_replicas': desired_replicas,
                'stabilized_replicas': stabilized_replicas,
                'cpu_utilization_fraction': utilization_fraction,
                'target_utilization': self.target_utilization,
                'tolerance': self.tolerance,
                'ratio': (utilization_fraction / self.target_utilization) if self.target_utilization > 0 else 0.0,
                'scaling_needed': stabilized_replicas != current_replicas,
                'scaling_executed': False
            }
            
            # Check if scaling is needed after stabilization
            if stabilized_replicas != current_replicas and self.pending_scaling is None:
                print(f"[{self.env.now:.1f}s] SCALING DECISION: CPU={utilization_fraction:.2f} "
                      f"Target={self.target_utilization:.2f} Current={current_replicas} "
                      f"Desired={desired_replicas} Stabilized={stabilized_replicas}")
                
                # Start scaling process (now synchronous within the controller's turn)
                self.pending_scaling = stabilized_replicas
                self._execute_scaling(current_replicas, stabilized_replicas)
                decision['scaling_executed'] = True
            elif self.pending_scaling is not None:
                print(f"[{self.env.now:.1f}s] SCALING BLOCKED: Scaling to {self.pending_scaling} in progress")
            
            self.scaling_decisions.append(decision)
    
    def _execute_scaling(self, current_replicas: int, target_replicas: int) -> None:
        """Execute scaling action.
        Note: ApplicationServer applies a 30s warm-up for new replicas upon upscaling.
        """
        if target_replicas > current_replicas:
            # UPSCALE: act immediately; rely on 30s warm-up in ApplicationServer
            print(f"[{self.env.now:.1f}s] UPSCALE COMMAND: {current_replicas} -> {target_replicas} (pods warm up in 30s)")
        else:
            # DOWNSCALE: immediate action
            print(f"[{self.env.now:.1f}s] DOWNSCALE COMMAND: {current_replicas} -> {target_replicas}")
        
        # Apply the scaling action
        self.application_server.set_replicas(target_replicas)
        print(f"[{self.env.now:.1f}s] SCALING APPLIED: Target replicas = {target_replicas}")
        
        # Clear pending scaling
        self.pending_scaling = None
