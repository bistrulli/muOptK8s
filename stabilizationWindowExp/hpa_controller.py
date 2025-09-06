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


class AdaptiveHPAController(HPAController):
    """
    An HPAController that dynamically adjusts its downscale stabilization window
    based on a comparison between measured throughput and a theoretical model.
    """
    def __init__(self, env: simpy.Environment, application_server: 'ApplicationServer',
                 target_utilization: float, evaluation_interval: float,
                 upscale_stabilization_window: float,
                 # Simulation parameters needed for the model
                 service_time_mean: float,
                 think_time_mean: float,
                 log_data: list, # Shared log data for calculating averages
                 # Initial window, min/max bounds, and adaptation parameters
                 initial_downscale_window: float,
                 min_downscale_window: float = 0.0,
                 max_downscale_window: float = 600.0,
                 adaptation_interval: float = 180.0,
                 kp: float = 50.0, # Proportional gain
                 ki: float = 1.0,  # Integral gain
                 # Standard HPA params
                 min_replicas: int = 1,
                 max_replicas: Optional[int] = None,
                 tolerance: float = 0.1,
                 utilization_window: float = 60.0):
        
        # Initialize the base HPA controller
        super().__init__(
            env, application_server, target_utilization, evaluation_interval,
            upscale_stabilization_window, initial_downscale_window,
            min_replicas, max_replicas, tolerance, utilization_window
        )
        
        # Parameters for the adaptation logic
        self.service_time_mean = service_time_mean
        self.think_time_mean = think_time_mean
        self.log_data = log_data
        self.min_downscale_window = min_downscale_window
        self.max_downscale_window = max_downscale_window
        self.adaptation_interval = adaptation_interval
        self.kp = kp # Proportional gain
        self.ki = ki # Integral gain
        self.integral_error = 0.0 # Accumulator for the integral term
        self.last_adaptation_error = 0.0 # For logging
        
        # Store for historical data needed for averages
        self.history = deque()
        self.env.process(self.run_adapter())

    @property
    def adaptive_downscale_window(self) -> float:
        """Provide access to the current, dynamic window size."""
        return self.downscale_stabilization_window

    def run_adapter(self):
        """Periodically runs the adaptation logic for the stabilization window."""
        from simulation_utils import solve_with_line_mm_c
        
        # Initial delay to gather some data before the first adjustment
        yield self.env.timeout(self.adaptation_interval)
        
        while True:
            # 1. Collect data from the last `adaptation_interval` seconds
            now = self.env.now
            cutoff = now - self.adaptation_interval
            
            # Use the shared log_data to get a view of the last interval
            # This is more accurate than instantaneous measurements
            recent_history = [log for log in self.log_data if log['timestamp'] >= cutoff]
            if not recent_history:
                yield self.env.timeout(self.adaptation_interval)
                continue

            # 2. Calculate measured throughput and average users/replicas from historical data
            completions_in_interval = self.application_server.get_throughput(self.adaptation_interval) * self.adaptation_interval
            measured_throughput = completions_in_interval / self.adaptation_interval
            
            avg_users = sum(log['user_count'] for log in recent_history) / len(recent_history)
            avg_replicas = sum(log['current_replicas'] for log in recent_history) / len(recent_history)

            # 3. Calculate theoretical throughput using the LINE model
            try:
                theoretical_results = solve_with_line_mm_c(
                    num_users=int(round(avg_users)),
                    servers=int(round(avg_replicas * self.application_server.cpus_per_replica)),
                    service_time_mean=self.service_time_mean,
                    think_time_mean=self.think_time_mean
                )
                theoretical_throughput = theoretical_results['throughput']
            except Exception as e:
                print(f"[{now:.1f}s] ADAPTER: Could not calculate theoretical model: {e}")
                yield self.env.timeout(self.adaptation_interval)
                continue

            # 4. Calculate percentage error and adjust the window using PI control
            if theoretical_throughput > 0:
                error = (theoretical_throughput - measured_throughput) / theoretical_throughput
            else:
                error = 0.0 # Or handle as a special case where no adaptation is possible
            
            # Update integral term with anti-windup.
            # Only accumulate error if the window is not already at a boundary
            # that the controller is trying to push against.
            if (error > 0 and self.downscale_stabilization_window < self.max_downscale_window) or \
               (error < 0 and self.downscale_stabilization_window > self.min_downscale_window):
                self.integral_error += error * self.adaptation_interval

            self.last_adaptation_error = error # Store for logging
            
            proportional_term = self.kp * error
            integral_term = self.ki * self.integral_error
            adjustment = proportional_term + integral_term
            
            old_window = self.downscale_stabilization_window
            new_window = old_window + adjustment
            
            # Clamp the new window value to be within min/max bounds
            new_window_clamped = max(self.min_downscale_window, min(new_window, self.max_downscale_window))
            
            # Reset integral error if the window is clamped to prevent wind-up
            if new_window != new_window_clamped:
                self.integral_error = 0

            self.downscale_stabilization_window = new_window_clamped
            
            print(f"[{now:.1f}s] ADAPTER: T_th={theoretical_throughput:.2f}, T_m={measured_throughput:.2f}, "
                  f"Err={error:.2%}, P_term={proportional_term:.2f}, I_term={integral_term:.2f}, "
                  f"Old_Win={old_window:.1f}s, New_Win={self.downscale_stabilization_window:.1f}s")
            
            yield self.env.timeout(self.adaptation_interval)
