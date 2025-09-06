#!/usr/bin/env python3
"""
Warm-up Ramp Experiment (no HPA)

Goal: demonstrate the 30s warm-up effect when increasing replicas while the
arrival rate ramps up. We directly schedule upscales over time and observe
the delayed increase in effective capacity and throughput.
"""

import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from application_server import ApplicationServer
from workload_generator import WorkloadGenerator


def ramp_arrival_rate(env: simpy.Environment, workload: WorkloadGenerator,
                      start_rate: float, end_rate: float, duration: float,
                      num_steps: int = 6) -> simpy.Process:
    """
    Linearly ramp the arrival rate from start_rate to end_rate over 'duration' seconds
    with a small number of large steps (plateaus) so throughput can stabilize.
    """
    total_steps = max(1, int(num_steps))
    step_seconds = float(duration) / float(total_steps)
    rate_delta = (end_rate - start_rate) / float(total_steps)

    # Initialize to start_rate
    workload.set_arrival_rate(start_rate)

    for step in range(1, total_steps + 1):
        yield env.timeout(step_seconds)
        new_rate = start_rate + rate_delta * step
        workload.set_arrival_rate(max(0.0001, new_rate))


def scheduled_upscales(env: simpy.Environment, app: ApplicationServer,
                       schedule: List[Dict[str, int]]) -> simpy.Process:
    """
    Apply a list of upscales at specified times.
    schedule: list of dicts like {'time': 120, 'replicas': 2}
    Note: ApplicationServer enforces a 30s warm-up on increases.
    """
    # Ensure schedule is time-ordered
    schedule_sorted = sorted(schedule, key=lambda x: x['time'])

    current_time = 0.0
    for item in schedule_sorted:
        target_time = float(item['time'])
        if target_time > current_time:
            yield env.timeout(target_time - current_time)
            current_time = target_time
        app.set_replicas(int(item['replicas']))


def experiment_logger(env: simpy.Environment, app: ApplicationServer,
                      workload: WorkloadGenerator, log: List[Dict[str, Any]],
                      interval: float = 1.0) -> simpy.Process:
    """
    Log key metrics periodically.
    """
    while True:
        yield env.timeout(interval)
        log.append({
            't': env.now,
            'arrival_rate': workload.arrival_rate,
            'replicas': app.current_replicas,
            'capacity_cores': app.cpu_resource.capacity,
            'busy_cores_now': app.get_current_cpu_utilization(),
            'throughput_10s': app.get_throughput(10.0),
            'cum_requests': len(app.completed_requests),
            'avg_resp_time': app.get_avg_response_time(),
        })


def run_warmup_ramp_experiment() -> Dict[str, Any]:
    """
    Run the warm-up ramp experiment without HPA.
    """
    # Simulation parameters
    SIM_DURATION = 1200.0
    INITIAL_REPLICAS = 1
    CPUS_PER_REPLICA = 1
    SERVICE_TIME_MEAN = 1.0  # seconds (exponential)

    # Load ramp: from 0.5 req/s to 8.0 req/s over first 480s
    RAMP_START_RATE = 0.5
    RAMP_END_RATE = 8.0
    RAMP_DURATION = 1200.0

    # Scheduled replica increases (observe 30s delayed capacity effect)
    UPSCALE_SCHEDULE = [
        {'time': 400, 'replicas': 4},
        {'time': 800, 'replicas': 8},
        {'time': 1200, 'replicas': 16},
    ]

    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    env = simpy.Environment()

    # Service time function (exponential)
    service_time_func = lambda: np.random.exponential(SERVICE_TIME_MEAN)

    app = ApplicationServer(env, INITIAL_REPLICAS, CPUS_PER_REPLICA, service_time_func)
    workload = WorkloadGenerator(env, app, initial_arrival_rate=RAMP_START_RATE)

    # Processes
    env.process(workload.run())
    env.process(ramp_arrival_rate(env, workload, RAMP_START_RATE, RAMP_END_RATE, RAMP_DURATION, num_steps=3))
    env.process(scheduled_upscales(env, app, UPSCALE_SCHEDULE))

    # Logging
    log: List[Dict[str, Any]] = []
    env.process(experiment_logger(env, app, workload, log, interval=1.0))

    print("=== Warm-up Ramp Experiment (no HPA) ===")
    print(f"Sim duration: {SIM_DURATION}s | Start rate: {RAMP_START_RATE} req/s -> {RAMP_END_RATE} req/s")
    print(f"Upscale schedule: {UPSCALE_SCHEDULE}")

    env.run(until=SIM_DURATION)

    # Post metrics
    total = len(app.completed_requests)
    overall_tput = total / SIM_DURATION
    print(f"Completed: {total} requests | Overall throughput: {overall_tput:.2f} req/s")

    return {
        'log': log,
        'total_requests': total,
        'overall_throughput': overall_tput,
        'schedule': UPSCALE_SCHEDULE,
        'ramp': {
            'start_rate': RAMP_START_RATE,
            'end_rate': RAMP_END_RATE,
            'duration': RAMP_DURATION,
        }
    }


def plot_experiment(results: Dict[str, Any]) -> None:
    log = results['log']
    if not log:
        print("No log data to plot.")
        return

    import pandas as pd
    df = pd.DataFrame(log)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle('Warm-up Ramp Experiment (No HPA)')

    # Top: Rates and Throughput
    ax1.plot(df['t'], df['arrival_rate'], label='Arrival rate (req/s)', color='tab:orange')
    # Moving average over 30s for throughput (logger interval 1s)
    if 'throughput_10s' in df.columns:
        df['throughput_ma_30s'] = df['throughput_10s'].rolling(window=30, min_periods=1).mean()
        ax1.plot(df['t'], df['throughput_10s'], label='Throughput last 10s (raw)', color='tab:blue', alpha=0.35, linewidth=1)
        ax1.plot(df['t'], df['throughput_ma_30s'], label='Throughput (30s MA)', color='navy', linewidth=2.5)
    ax1.set_ylabel('req/s')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Bottom: Replicas and Capacity
    ax2.plot(df['t'], df['replicas'], label='Replicas (commanded)', color='tab:green')
    ax2.plot(df['t'], df['capacity_cores'], label='Active CPU capacity (cores)', color='tab:red')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('count')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def main() -> Dict[str, Any]:
    results = run_warmup_ramp_experiment()
    plot_experiment(results)
    return results


if __name__ == "__main__":
    main()


