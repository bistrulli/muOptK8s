#!/usr/bin/env python3
"""
HPA Autoscaling Simulation with Actuation Delay using SimPy
This script simulates the behavior of a web application managed by an autoscaling controller
similar to Kubernetes HPA, focusing on the impact of actuation delays.
"""

import simpy
import random
import numpy as np
import time
from typing import List, Dict, Any

# Import our modular components
from application_server import ApplicationServer
from closed_workload_generator import ClosedWorkloadGenerator
from hpa_controller import HPAController
from simulation_utils import (
    data_logger, 
    plot_results
)


def workload_from_schedule(env: simpy.Environment,
                           workload: ClosedWorkloadGenerator,
                           schedule: List[Dict[str, float]]) -> simpy.Process:
    """Replay a predetermined step schedule for fairness across experiments."""
    current_time = 0.0
    for step in schedule:
        target_time = float(step['time'])
        target_users = int(step['users'])
        
        if target_time > current_time:
            yield env.timeout(target_time - current_time)
            current_time = target_time
            
        workload.set_users(target_users)


def generate_periodic_user_schedule(total_duration: float,
                                    low_users: int,
                                    high_users: int,
                                    low_duration: float,
                                    high_duration: float) -> List[Dict[str, float]]:
    """Generate a simple periodic up/down user step schedule covering total_duration."""
    schedule: List[Dict[str, float]] = []
    remaining = float(total_duration)
    current_time = 0.0
    
    # Alternate: high then low repeatedly
    is_high_phase = True
    while remaining > 0:
        if is_high_phase:
            duration = min(remaining, float(high_duration))
            schedule.append({'time': current_time, 'users': float(high_users)})
        else:  # low phase
            duration = min(remaining, float(low_duration))
            schedule.append({'time': current_time, 'users': float(low_users)})
        
        current_time += duration
        remaining -= duration
        is_high_phase = not is_high_phase
        
    return schedule


def generate_spikes_and_dips_schedule(total_duration: float,
                                      low_users: int,
                                      high_users: int,
                                      dip_duration: float,
                                      spike_duration: float) -> List[Dict[str, float]]:
    """Generate a workload with long high-load spikes and short, sharp dips."""
    schedule: List[Dict[str, float]] = []
    remaining = float(total_duration)
    current_time = 0.0
    
    # Start with a spike
    is_spike_phase = True
    while remaining > 0:
        if is_spike_phase:
            duration = min(remaining, float(spike_duration))
            schedule.append({'time': current_time, 'users': float(high_users)})
        else:  # dip phase
            duration = min(remaining, float(dip_duration))
            schedule.append({'time': current_time, 'users': float(low_users)})
        
        current_time += duration
        remaining -= duration
        is_spike_phase = not is_spike_phase
        
    return schedule


def run_hpa_with_window_with_schedule(sim_duration: float, downscale_window: float,
                                      schedule: List[Dict[str, float]]) -> Dict[str, Any]:
    """Run HPA using a pre-generated workload schedule (same waveform across runs)."""
    INITIAL_REPLICAS = 1
    CPUS_PER_REPLICA = 1
    SERVICE_TIME_MEAN = 1.0
    TARGET_CPU_UTILIZATION = 0.20
    EVALUATION_INTERVAL = 15.0
    UPSCALE_WINDOW = 0.0
    THINK_TIME_MEAN = 1.0  # seconds
    INITIAL_USERS = 1

    random.seed(42)
    np.random.seed(42)

    env = simpy.Environment()
    service_time_func = lambda: random.expovariate(1.0 / SERVICE_TIME_MEAN)
    app_server = ApplicationServer(env, INITIAL_REPLICAS, CPUS_PER_REPLICA, service_time_func)
    workload_gen = ClosedWorkloadGenerator(env, app_server, THINK_TIME_MEAN, INITIAL_USERS)
    hpa_controller = HPAController(
        env, app_server, TARGET_CPU_UTILIZATION,
        EVALUATION_INTERVAL,
        UPSCALE_WINDOW, downscale_window,
        min_replicas=1, max_replicas=None, tolerance=0.1, utilization_window=10.0
    )

    log_data: List[Dict[str, Any]] = []

    env.process(hpa_controller.run())
    env.process(workload_from_schedule(env, workload_gen, schedule))
    env.process(data_logger(env, app_server, workload_gen, log_data, hpa_controller))

    env.run(until=sim_duration)

    total_completed = len(app_server.completed_requests)
    cumulative_throughput = total_completed / sim_duration if sim_duration > 0 else 0.0
    total_arrivals = len(app_server.request_arrivals)
    backlog = max(0, total_arrivals - total_completed)

    return {
        'downscale_window': downscale_window,
        'total_requests': total_completed,
        'total_arrivals': total_arrivals,
        'backlog': backlog,
        'cumulative_throughput': cumulative_throughput,
        'log_data': log_data,
    }


def main():
    """Main: sweep decreasing downscale windows on a high-variability workload."""
    SIM_DURATION = 2800.0
    windows = [300.0, 180.0, 60.0, 0.0]
    results: List[Dict[str, Any]] = []
    print("=== HPA Downscale Stabilization Window Sweep (Closed-Loop Workload) ===")
    
    # Equivalent closed-loop user counts for the previous open-loop rates
    # Approximation: N = lambda * (R + Z). Here R=1, Z=1.
    LOW_USERS = 2
    HIGH_USERS = 100

    schedule = generate_spikes_and_dips_schedule(
        total_duration=SIM_DURATION,
        low_users=LOW_USERS, high_users=HIGH_USERS,
        dip_duration=35.0,  # Short dip to bait the controller
        spike_duration=300.0 # Long spike to establish high replica count
    )

    for w in windows:
        print(f"Running with downscale_window={w}s ...")
        res = run_hpa_with_window_with_schedule(SIM_DURATION, w, schedule)
        print(f"  Cumulative throughput: {res['cumulative_throughput']:.3f} req/s (total {res['total_requests']})")
        results.append(res)
        
    # Plot per-window time series (user count, throughput MA, capacity cores)
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        for r in results:
            df = pd.DataFrame(r['log_data'])
            if 'throughput' in df.columns:
                df['throughput_ma_30s'] = df['throughput'].rolling(window=30, min_periods=1).mean()
            fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            fig.suptitle(f"Downscale window = {int(r['downscale_window'])}s")
            # Workload
            ax[0].plot(df['timestamp'], df['user_count'], color='tab:orange', linewidth=2)
            ax[0].set_ylabel('Active Users')
            ax[0].grid(True, alpha=0.3)
            # Throughput
            if 'throughput' in df.columns:
                ax[1].plot(df['timestamp'], df['throughput'], color='tab:blue', alpha=0.25, linewidth=1, label='Throughput (raw)')
            if 'throughput_ma_30s' in df.columns:
                ax[1].plot(df['timestamp'], df['throughput_ma_30s'], color='navy', linewidth=2.5, label='Throughput (30s MA)')
            ax[1].set_ylabel('Throughput (req/s)')
            ax[1].grid(True, alpha=0.3)
            ax[1].legend()
            # Cumulative requests (right axis)
            if 'cumulative_requests' in df.columns:
                ax2b = ax[1].twinx()
                ax2b.plot(df['timestamp'], df['cumulative_requests'], color='gray', linewidth=1.2, label='Cumulative requests')
                ax2b.set_ylabel('Cumulative requests')
            # Capacity cores
            if 'capacity_cores' in df.columns:
                ax[2].plot(df['timestamp'], df['capacity_cores'], color='tab:red', linewidth=2)
            ax[2].set_ylabel('Active cores')
            ax[2].set_xlabel('Time (s)')
            ax[2].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    except Exception:
        pass
    
    # Final summary table: cumulative throughput, arrivals, backlog and average replicas per window
    try:
        print("\n=== Summary Table ===")
        header = f"{'Window(s)':<12}{'Cumul. Throughput (req/s)':>28}{'TotalReq':>12}{'Arrivals':>12}{'Backlog':>10}{'Avg Replicas':>16}"
        print(header)
        print('-' * len(header))
        for r in results:
            logs = r.get('log_data', [])
            if logs:
                avg_repl = sum(e.get('current_replicas', 0) for e in logs) / float(len(logs))
            else:
                avg_repl = 0.0
            print(f"{int(r['downscale_window']):<12}{r['cumulative_throughput']:>28.3f}{r['total_requests']:>12}{r.get('total_arrivals', 0):>12}{r.get('backlog', 0):>10}{avg_repl:>16.2f}")
    except Exception:
        pass
    return results


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run the simulation
    simulation_data = main()