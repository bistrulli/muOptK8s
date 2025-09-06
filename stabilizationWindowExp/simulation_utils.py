#!/usr/bin/env python3
"""
Simulation utilities for HPA simulation.
Helper functions for workload scenarios, data logging, and visualization.
"""

import simpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from workload_generator import WorkloadGenerator
    from closed_workload_generator import ClosedWorkloadGenerator
    from application_server import ApplicationServer
    from hpa_controller import HPAController


def create_workload_scenario(env: simpy.Environment, workload_generator: 'WorkloadGenerator') -> simpy.Process:
    """
    Creates a highly fluctuating workload scenario to stress stabilization window.
    Rapid changes between high and low load to trigger frequent scaling decisions.
    """
    # Start with low load
    yield env.timeout(50)
    
    # Fluctuating pattern: rapid changes every 30-60s
    fluctuation_pattern = [
        (60, 5.0),   # High load spike
        (60, 1.0),   # Drop to low
        (60, 6.0),   # Higher spike
        (60, 0.5),   # Very low
        (60, 7.0),   # Very high spike
        (60, 2.0),   # Medium
        (60, 8.0),   # Peak load
        (60, 1.5),   # Low again
        (60, 6.0),   # High again
        (60, 0.8),   # Low
        (60, 5.5),   # Medium-high
        (50, 1.0)    # Final low
    ]
    
    for duration, arrival_rate in fluctuation_pattern:
        workload_generator.set_arrival_rate(arrival_rate)
        yield env.timeout(duration)


def data_logger(env: simpy.Environment, application_server: 'ApplicationServer', 
                workload_generator: 'ClosedWorkloadGenerator', log_data: List[Dict[str, Any]],
                hpa_controller: 'HPAController') -> simpy.Process:
    """
    Logs system state every second for analysis.
    """
    while True:
        yield env.timeout(1.0)  # Log every second
        
        log_entry = {
            'timestamp': env.now,
            'user_count': workload_generator.user_count,
            'current_replicas': application_server.current_replicas,
            'capacity_cores': application_server.cpu_resource.capacity,
            'cpu_utilization': application_server.get_avg_cpu_utilization(),
            'response_time': application_server.get_avg_response_time(),
            'throughput': application_server.get_throughput(10.0),  # Last 10 seconds
            'cumulative_requests': len(application_server.completed_requests),
            'queue_length': len(application_server.cpu_resource.queue)
        }
        log_data.append(log_entry)


def plot_results(log_data: List[Dict[str, Any]], target_utilization: float) -> None:
    """
    Create visualization plots of the simulation results.
    """
    df = pd.DataFrame(log_data)
    # Compute 30s moving average of throughput (logger interval is 1s)
    if 'throughput' in df.columns:
        df['throughput_ma_30s'] = df['throughput'].rolling(window=30, min_periods=1).mean()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('HPA Autoscaling Simulation Results', fontsize=16)
    
    # Plot 1: Replicas over time
    ax1.plot(df['timestamp'], df['current_replicas'], 'b-', linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Number of Replicas')
    ax1.set_title('Replica Count vs Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: CPU Utilization vs Target
    ax2.plot(df['timestamp'], df['cpu_utilization'], 'r-', linewidth=2, label='Actual CPU Utilization')
    ax2.axhline(y=target_utilization, color='g', linestyle='--', linewidth=2, label=f'Target ({target_utilization:.0%})')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('CPU Utilization')
    ax2.set_title('CPU Utilization vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.2)
    
    # Plot 3: Response Time
    ax3.plot(df['timestamp'], df['response_time'], 'purple', linewidth=2)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Response Time (seconds)')
    ax3.set_title('Average Response Time vs Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Arrival Rate and Throughput (with 30s moving average)
    ax4.plot(df['timestamp'], df['user_count'], 'orange', linewidth=2, label='Arrival Rate')
    if 'throughput' in df.columns:
        ax4.plot(df['timestamp'], df['throughput'], color='cyan', linewidth=1, alpha=0.35, label='Throughput (raw)')
    if 'throughput_ma_30s' in df.columns:
        ax4.plot(df['timestamp'], df['throughput_ma_30s'], color='blue', linewidth=2.5, label='Throughput (30s MA)')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Requests/second')
    ax4.set_title('Arrival Rate vs Throughput')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_stabilization_comparison(all_results: List[Dict[str, Any]]) -> None:
    """
    Create comparison plots showing the impact of stabilization window.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Stabilization Window Impact on HPA Performance', fontsize=16)
    
    # Extract comparison data
    windows = [r['downscale_window'] for r in all_results]
    throughputs = [r['cumulative_throughput'] for r in all_results]
    high_load_throughputs = [r['high_load_throughput'] for r in all_results]
    response_times = [r['avg_response_time'] for r in all_results]
    high_load_response_times = [r['high_load_response_time'] for r in all_results]
    scaling_actions = [r['scaling_actions'] for r in all_results]
    blocked_actions = [r['blocked_actions'] for r in all_results]
    
    # Plot 1: High Load Period Throughput vs Downscale Stabilization Window  
    ax1.bar(windows, high_load_throughputs, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Downscale Stabilization Window (seconds)')
    ax1.set_ylabel('High-Load Throughput (req/s)')
    ax1.set_title('Throughput During High Load vs Downscale Window')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: High Load Response Time vs Downscale Stabilization Window
    ax2.bar(windows, high_load_response_times, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Downscale Stabilization Window (seconds)')
    ax2.set_ylabel('High-Load Response Time (seconds)')
    ax2.set_title('Response Time During High Load vs Downscale Window')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scaling Actions vs Stabilization Window
    x_pos = np.arange(len(windows))
    ax3.bar(x_pos - 0.2, scaling_actions, width=0.4, label='Executed', color='green', alpha=0.7)
    ax3.bar(x_pos + 0.2, blocked_actions, width=0.4, label='Blocked', color='red', alpha=0.7)
    ax3.set_xlabel('Stabilization Window (seconds)')
    ax3.set_ylabel('Number of Scaling Actions')
    ax3.set_title('Scaling Actions vs Stabilization Window')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(windows)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Timeline comparison for shortest and longest windows
    short_window_data = pd.DataFrame(all_results[0]['log_data'])
    long_window_data = pd.DataFrame(all_results[-1]['log_data'])
    
    ax4.plot(short_window_data['timestamp'], short_window_data['current_replicas'], 
             label=f'Window={windows[0]}s', linewidth=2)
    ax4.plot(long_window_data['timestamp'], long_window_data['current_replicas'], 
             label=f'Window={windows[-1]}s', linewidth=2)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Number of Replicas')
    ax4.set_title('Replica Timeline Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary table
    print(f"\n{'='*110}")
    print(f"{'Upscale':<10} {'Downscale':<12} {'High-Load':<12} {'High-Load':<12} {'Overall':<12} {'Scaling':<10} {'Blocked':<10}")
    print(f"{'Window(s)':<10} {'Window(s)':<12} {'Throughput':<12} {'Resp Time':<12} {'Throughput':<12} {'Actions':<10} {'Actions':<10}")
    print(f"{'='*110}")
    for r in all_results:
        print(f"{r['upscale_window']:<10} {r['downscale_window']:<12} {r['high_load_throughput']:<12.2f} "
              f"{r['high_load_response_time']:<12.3f} {r['cumulative_throughput']:<12.2f} "
              f"{r['scaling_actions']:<10} {r['blocked_actions']:<10}")
    print(f"{'='*110}")
