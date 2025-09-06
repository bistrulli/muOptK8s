#!/usr/bin/env python3
"""
ClosedWorkloadGenerator module for HPA simulation.
Simulates a configurable number of users in a think-request-response cycle.
"""

import simpy
import numpy as np
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from application_server import ApplicationServer
from application_server import user_behavior

class ClosedWorkloadGenerator:
    """
    Manages a population of users for a closed workload model.
    Can dynamically add and remove users during the simulation.
    """
    
    def __init__(self, env: simpy.Environment, app_server: 'ApplicationServer', 
                 think_time_mean: float, initial_users: int = 0):
        self.env = env
        self.app_server = app_server
        self.think_time_mean = think_time_mean
        self.user_processes: List[simpy.Process] = []
        self._next_user_id = 0

        if initial_users > 0:
            self.set_users(initial_users)

    @property
    def user_count(self) -> int:
        """Return the current number of active users."""
        return len(self.user_processes)

    def _create_user_process(self) -> simpy.Process:
        """Create and return a single user process."""
        user_id = self._next_user_id
        self._next_user_id += 1
        return self.env.process(user_behavior(
            env=self.env,
            app_server=self.app_server,
            user_id=user_id,
            session_duration=float('inf'),  # Run indefinitely
            think_time_mean=self.think_time_mean
        ))

    def set_users(self, num_target_users: int):
        """Set the number of users to a specific target, adding or removing as needed."""
        current_count = self.user_count
        delta = num_target_users - current_count
        
        if delta > 0:
            # Add new users
            print(f"[{self.env.now:.1f}s] WORKLOAD CHANGE: Adding {delta} users to reach target of {num_target_users}.")
            for _ in range(delta):
                self.user_processes.append(self._create_user_process())
        elif delta < 0:
            # Remove users
            num_to_remove = abs(delta)
            print(f"[{self.env.now:.1f}s] WORKLOAD CHANGE: Removing {num_to_remove} users to reach target of {num_target_users}.")
            
            # Stop and remove the last N user processes
            for _ in range(num_to_remove):
                if self.user_processes:
                    proc_to_stop = self.user_processes.pop()
                    # Interrupt the user process. This will stop it at the next yield.
                    # This simulates users leaving the session.
                    proc_to_stop.interrupt(f"User removed by workload generator at {self.env.now:.1f}s")

        print(f"[{self.env.now:.1f}s] WORKLOAD CHANGE: Total users now {self.user_count}.")
