#!/usr/bin/env python3
"""
ApplicationServer module for HPA simulation.
Multi-server queuing center (G/G/c) with FIFO discipline representing the application infrastructure.
"""

import simpy
import numpy as np
from typing import Callable
from collections import deque
import random
import time
import sys

# Optional import of LINE solver (for analytical comparison)
try:
    sys.path.append('/Users/emilio-imt/line-solver/python')
    from line_solver import Network, Delay, Queue, Sink, OpenClass, ClosedClass, SchedStrategy
    from line_solver import SolverJMT, SolverMVA, Network as LineNetwork
    from line_solver import Exp as LineExp
    from line_solver import tget
    LINE_AVAILABLE = True
except Exception:
    LINE_AVAILABLE = False


class ApplicationServer:
    """
    Multi-server queuing center (G/G/c) with FIFO discipline representing the application infrastructure.
    """
    
    def __init__(self, env: simpy.Environment, initial_replicas: int, 
                 cpus_per_replica: int, service_time_func: Callable[[], float],
                 warmup_seconds: float = 30.0):
        self.env = env
        self.cpus_per_replica = cpus_per_replica
        self.service_time_func = service_time_func
        self.current_replicas = initial_replicas
        self.warmup_seconds = float(warmup_seconds)
        
        # SimPy resource representing total CPU capacity
        total_cpus = initial_replicas * cpus_per_replica
        self.cpu_resource = simpy.Resource(env, capacity=total_cpus)
        
        # Metrics tracking
        self.completed_requests = []
        self.request_arrivals = []
        self.cpu_usage_samples = deque(maxlen=1000)
        self.request_count = 0
        
    def process_request(self) -> simpy.Process:
        """Process a single request through the application server."""
        request_id = self.request_count
        self.request_count += 1
        arrival_time = self.env.now
        self.request_arrivals.append(arrival_time)
        
        # Request CPU resource (FIFO)
        try:
            with self.cpu_resource.request() as request:
                yield request
                
                # Service the request
                service_start = self.env.now
                wait_time = service_start - arrival_time
                service_time = self.service_time_func()
                
                yield self.env.timeout(service_time)
                
                completion_time = self.env.now
                response_time = completion_time - arrival_time
                
                self.completed_requests.append({
                    'request_id': request_id,
                    'arrival_time': arrival_time,
                    'wait_time': wait_time,
                    'service_time': service_time,
                    'response_time': response_time,
                    'completion_time': completion_time
                })
        except simpy.Interrupt:
            # This can happen if a user process is interrupted while waiting for a resource
            print(f"[{self.env.now:.1f}s] Request {request_id} interrupted and cancelled.")
    
    def get_avg_response_time(self) -> float:
        """Get average response time of all completed requests."""
        if not self.completed_requests:
            return 0.0
        return np.mean([req['response_time'] for req in self.completed_requests])
    
    def get_throughput(self, interval: float) -> float:
        """Get throughput (requests/second) in the last interval seconds."""
        current_time = self.env.now
        cutoff_time = current_time - interval
        
        recent_completions = [req for req in self.completed_requests 
                            if req['completion_time'] >= cutoff_time]
        return len(recent_completions) / interval if interval > 0 else 0.0
    
    def get_avg_cpu_utilization(self) -> float:
        """Get average busy CPU cores over the simulation time (absolute core count)."""
        if not self.completed_requests or not self.cpu_resource.capacity:
            return 0.0
        
        # Calculate total service time across all requests
        total_service_time = sum(req['service_time'] for req in self.completed_requests)
        
        # Calculate total simulation time
        if self.completed_requests:
            simulation_time = self.env.now
            total_capacity = self.cpu_resource.capacity
            
            # Average busy cores = total busy time across cores / simulation time
            avg_busy_cores = total_service_time / simulation_time
            return min(total_capacity, avg_busy_cores)
        
        return 0.0
    
    def get_recent_cpu_utilization(self, interval: float) -> float:
        """Get average busy CPU cores over the last `interval` seconds.
        Approximates busy cores as total service time of requests completed in the window divided by the window length.
        """
        if interval <= 0 or not self.cpu_resource.capacity:
            return 0.0
        if not self.completed_requests:
            return 0.0
        current_time = self.env.now
        cutoff_time = current_time - interval
        recent_reqs = [req for req in self.completed_requests if req['completion_time'] >= cutoff_time]
        if not recent_reqs:
            return 0.0
        total_service_time_recent = sum(req['service_time'] for req in recent_reqs)
        busy_cores_recent = total_service_time_recent / interval
        return min(self.cpu_resource.capacity, busy_cores_recent)
    
    def get_current_cpu_utilization(self) -> float:
        """Get current instantaneous busy CPU cores (absolute core count)."""
        if not self.cpu_resource.capacity:
            return 0.0
        
        current_usage = self.cpu_resource.count
        return float(current_usage)
    
    def set_replicas(self, new_replica_count: int) -> None:
        """Dynamically change the number of replicas with proper warm-up time."""
        old_replicas = self.current_replicas
        old_capacity = self.cpu_resource.capacity
        
        self.current_replicas = new_replica_count
        
        if new_replica_count > old_replicas:
            # Scale UP: new replicas need warm-up time to become active
            print(f"[{self.env.now:.1f}s] SCALE UP: {old_replicas} -> {new_replica_count} replicas")
            print(f"[{self.env.now:.1f}s] WARM-UP: New replicas need {self.warmup_seconds:.0f}s to become active")
            
            # Capacity stays the same until warm-up completes
            # Schedule capacity increase after warm-up
            new_capacity = new_replica_count * self.cpus_per_replica
            self.env.process(self._apply_capacity_after_warmup(new_capacity, self.warmup_seconds))
            
        elif new_replica_count < old_replicas:
            # Scale DOWN: immediate capacity reduction
            new_capacity = new_replica_count * self.cpus_per_replica
            self.cpu_resource._capacity = new_capacity
            print(f"[{self.env.now:.1f}s] SCALE DOWN: {old_replicas} -> {new_replica_count} replicas")
            print(f"[{self.env.now:.1f}s] CAPACITY REDUCED: {old_capacity} -> {new_capacity} CPUs")
    
    def _apply_capacity_after_warmup(self, target_capacity: int, warmup_time: float) -> simpy.Process:
        """Apply full capacity after warm-up period."""
        yield self.env.timeout(warmup_time)
        old_capacity = self.cpu_resource.capacity
        self.cpu_resource._capacity = target_capacity
        print(f"[{self.env.now:.1f}s] WARM-UP COMPLETE: {old_capacity} -> {target_capacity} CPUs now active")


def user_behavior(env: simpy.Environment, app_server: ApplicationServer, 
                  user_id: int, session_duration: float, 
                  think_time_mean: float = 3.0) -> simpy.Process:
    """
    Simulate closed workload user behavior: think-time → request → think-time cycle.
    Think time follows exponential distribution.
    
    Args:
        env: SimPy environment
        app_server: Application server instance
        user_id: Unique user identifier
        session_duration: How long the user session lasts (seconds)
        think_time_mean: Mean think time in seconds (exponential distribution)
    """
    try:
        session_start = env.now
        request_count = 0
        
        # Initial think time before first request (exponential distribution)
        initial_think_time = np.random.exponential(think_time_mean)
        print(f"[{env.now:.1f}s] User {user_id} created, starting with think time: {initial_think_time:.1f}s")
        yield env.timeout(initial_think_time)
        
        while env.now - session_start < session_duration:
            # Submit request to system
            request_count += 1
            print(f"[{env.now:.1f}s] User {user_id} submits request #{request_count}")
            
            # Process request through application server
            yield env.process(app_server.process_request())
            
            # Think time before next request (exponential distribution)
            think_time = np.random.exponential(think_time_mean)
            print(f"[{env.now:.1f}s] User {user_id} request #{request_count} completed, thinking for {think_time:.1f}s")
            yield env.timeout(think_time)
        
        print(f"[{env.now:.1f}s] User {user_id} session ended after {request_count} requests")
    except simpy.Interrupt:
        print(f"[{env.now:.1f}s] User {user_id} was removed from the simulation.")


def stress_test_application(num_users: int = 50, simulation_duration: float = 300.0,
                          initial_replicas: int = 3, cpus_per_replica: int = 2,
                          service_time_mean: float = 0.1, think_time_mean: float = 3.0):
    """
    Stress test the application server with multiple concurrent users using closed workload.
    Both service time and think time follow exponential distributions.
    
    Args:
        num_users: Number of concurrent users to simulate
        simulation_duration: Total simulation time in seconds
        initial_replicas: Initial number of application replicas
        cpus_per_replica: CPU cores per replica
        service_time_mean: Mean service time for requests in seconds (exponential distribution)
        think_time_mean: Mean think time in seconds (exponential distribution)
    """
    print("=" * 60)
    print("APPLICATION SERVER STRESS TEST")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Users: {num_users}")
    print(f"  - Duration: {simulation_duration}s")
    print(f"  - Initial replicas: {initial_replicas}")
    print(f"  - CPUs per replica: {cpus_per_replica}")
    print(f"  - Service time: {service_time_mean:.3f}s (exponential, mean)")
    print(f"  - Think time: {think_time_mean:.1f}s (exponential, mean)")
    print(f"  - Workload type: CLOSED")
    print("=" * 60)
    
    # Create SimPy environment
    env = simpy.Environment()
    
    # Define service time function (exponential distribution)
    def service_time_func():
        return np.random.exponential(service_time_mean)
    
    # Create application server
    app_server = ApplicationServer(
        env=env,
        initial_replicas=initial_replicas,
        cpus_per_replica=cpus_per_replica,
        service_time_func=service_time_func
    )
    
    # Start user sessions (all users start immediately in closed workload)
    for user_id in range(num_users):
        env.process(user_behavior(
            env=env,
            app_server=app_server,
            user_id=user_id,
            session_duration=simulation_duration,
            think_time_mean=think_time_mean
        ))
    
    # Run simulation
    start_time = time.time()
    env.run(until=simulation_duration)
    end_time = time.time()
    
    # Collect and report metrics
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    # Basic metrics
    total_requests = len(app_server.completed_requests)
    avg_response_time = app_server.get_avg_response_time()
    final_throughput = app_server.get_throughput(60.0)  # Last 60 seconds
    avg_busy_cores = app_server.get_avg_cpu_utilization()
    current_busy_cores = app_server.get_current_cpu_utilization()
    avg_throughput = (total_requests / simulation_duration) if simulation_duration > 0 else 0.0
    
    print(f"Total requests processed: {total_requests}")
    print(f"Average response time: {avg_response_time:.3f}s")
    print(f"Final throughput (last 60s): {final_throughput:.2f} req/s")
    print(f"Average throughput (overall): {avg_throughput:.2f} req/s")
    print(f"Average busy CPU cores: {avg_busy_cores:.2f}")
    print(f"Current busy CPU cores: {current_busy_cores:.2f}")
    print(f"Simulation wall time: {end_time - start_time:.2f}s")
    
    # Detailed CPU utilization explanation
    if app_server.completed_requests:
        total_service_time = sum(req['service_time'] for req in app_server.completed_requests)
        simulation_time = app_server.env.now
        total_capacity = app_server.cpu_resource.capacity
        print(f"\nCPU Utilization Details:")
        print(f"  Total service time: {total_service_time:.1f}s")
        print(f"  Simulation time: {simulation_time:.1f}s")
        print(f"  Total CPU capacity: {total_capacity} cores")
        print(f"  Average busy cores = {total_service_time:.1f}s / {simulation_time:.1f}s = {avg_busy_cores:.2f} cores")
    
    # Response time statistics
    if app_server.completed_requests:
        response_times = [req['response_time'] for req in app_server.completed_requests]
        print(f"\nResponse Time Statistics:")
        print(f"  Min: {min(response_times):.3f}s")
        print(f"  Max: {max(response_times):.3f}s")
        print(f"  Median: {np.median(response_times):.3f}s")
        print(f"  95th percentile: {np.percentile(response_times, 95):.3f}s")
        print(f"  99th percentile: {np.percentile(response_times, 99):.3f}s")
    
    # Wait time statistics
    if app_server.completed_requests:
        wait_times = [req['wait_time'] for req in app_server.completed_requests]
        print(f"\nWait Time Statistics:")
        print(f"  Average: {np.mean(wait_times):.3f}s")
        print(f"  Max: {max(wait_times):.3f}s")
        print(f"  95th percentile: {np.percentile(wait_times, 95):.3f}s")
    
    # Service time statistics
    if app_server.completed_requests:
        service_times = [req['service_time'] for req in app_server.completed_requests]
        print(f"\nService Time Statistics (Exponential Distribution):")
        print(f"  Average: {np.mean(service_times):.3f}s")
        print(f"  Expected mean: {service_time_mean:.3f}s")
        print(f"  Standard deviation: {np.std(service_times):.3f}s")
        print(f"  Theoretical std dev: {service_time_mean:.3f}s")
    
    # Analytical comparison using LINE solver (M/M/c closed network)
    try:
        print("\n" + "-" * 60)
        print("ANALYTICAL COMPARISON (LINE: M/M/c CLOSED)")
        print("-" * 60)
        from simulation_utils import solve_with_line_mm_c
        line_results = solve_with_line_mm_c(
            num_users=num_users,
            servers=initial_replicas * cpus_per_replica,
            service_time_mean=service_time_mean,
            think_time_mean=think_time_mean
        )
        print(f"LINE Throughput: {line_results['throughput']:.4f} req/s")
        print(f"LINE Avg response time: {line_results['response_time']:.4f}s")
        print(f"LINE Avg queue length: {line_results['q_len']:.4f}")
        print(f"LINE Busy cores (avg): {line_results['busy_cores']:.4f} cores")
    except Exception as e:
        print("[LINE] Analytical comparison skipped:", str(e))

    print("=" * 60)
    
    return {
        'total_requests': total_requests,
        'avg_response_time': avg_response_time,
        'final_throughput': final_throughput,
        'avg_throughput': avg_throughput,
        'avg_cpu_utilization': avg_busy_cores,
        'completed_requests': app_server.completed_requests
    }


if __name__ == "__main__":
    # Model validation stress test
    print("Starting Application Server Model Validation...")
    
    # Test grid of scenarios (various users and capacities)
    scenarios = []
    for num_users in [1, 3, 5, 10]:
        for replicas in [1, 2]:
            for cpus_per_replica in [1, 2]:
                scenarios.append({
                    'name': f'U{num_users}_R{replicas}_C{cpus_per_replica}',
                    'num_users': num_users,
                    'duration': 2000.0,
                    'replicas': replicas,
                    'cpus_per_replica': cpus_per_replica
                })
    
    results = {}
    comparisons = []
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"RUNNING SCENARIO: {scenario['name']}")
        print(f"{'='*80}")
        
        result = stress_test_application(
            num_users=scenario['num_users'],
            simulation_duration=scenario['duration'],
            initial_replicas=scenario['replicas'],
            cpus_per_replica=scenario['cpus_per_replica'],
            service_time_mean=1.0,  # Mean service time: 1 second (exponential)
            think_time_mean=1.0     # Mean think time: 5 seconds (exponential)
        )
        
        results[scenario['name']] = result

        # Analytical LINE comparison for this scenario
        try:
            from simulation_utils import solve_with_line_mm_c
            line_res = solve_with_line_mm_c(
                num_users=scenario['num_users'],
                servers=scenario['replicas'] * scenario['cpus_per_replica'],
                service_time_mean=1.0,
                think_time_mean=1.0
            )
            # Compute absolute and relative errors
            def rel_err(sim, ana):
                return (sim - ana) / ana if ana != 0 else float('inf')

            comparisons.append({
                'name': scenario['name'],
                'users': scenario['num_users'],
                'servers': scenario['replicas'] * scenario['cpus_per_replica'],
                'sim_tput': result['avg_throughput'],
                'ana_tput': line_res['throughput'],
                'err_tput_abs': result['avg_throughput'] - line_res['throughput'],
                'err_tput_rel': rel_err(result['avg_throughput'], line_res['throughput']),
                'sim_resp': result['avg_response_time'],
                'ana_resp': line_res['response_time'],
                'err_resp_abs': result['avg_response_time'] - line_res['response_time'],
                'err_resp_rel': rel_err(result['avg_response_time'], line_res['response_time']),
                'sim_busy': result['avg_cpu_utilization'],
                'ana_busy': line_res['busy_cores'],
                'err_busy_abs': result['avg_cpu_utilization'] - line_res['busy_cores'],
                'err_busy_rel': rel_err(result['avg_cpu_utilization'], line_res['busy_cores'])
            })
        except Exception as e:
            print(f"[LINE] Skip comparison for {scenario['name']}: {e}")
        
        # Brief pause between scenarios
        time.sleep(1)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SCENARIO COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Scenario':<15} {'Requests':<10} {'Avg Resp (s)':<12} {'Throughput':<12} {'Busy Cores':<12}")
    print("-" * 80)
    
    for name, result in results.items():
        print(f"{name:<15} {result['total_requests']:<10} "
              f"{result['avg_response_time']:<12.3f} "
              f"{result['final_throughput']:<12.2f} "
              f"{result['avg_cpu_utilization']:<12.2f}")
    
    print(f"{'='*80}")
    print("Model validation completed successfully!")

    # Print validation errors vs LINE
    if len(comparisons) > 0:
        print("\n" + "="*80)
        print("VALIDATION ERRORS VS LINE (per scenario)")
        print("="*80)
        header = f"{'Scenario':<15} {'N':<3} {'c':<3} " \
                 f"{'sim_tput':>10} {'ana_tput':>10} {'rel%':>8} " \
                 f"{'sim_resp':>10} {'ana_resp':>10} {'rel%':>8} " \
                 f"{'sim_busy':>10} {'ana_busy':>10} {'rel%':>8}"
        print(header)
        print("-" * len(header))
        for r in comparisons:
            print(f"{r['name']:<15} {r['users']:<3d} {r['servers']:<3d} "
                  f"{r['sim_tput']:>10.3f} {r['ana_tput']:>10.3f} {100.0*r['err_tput_rel']:>8.2f} "
                  f"{r['sim_resp']:>10.3f} {r['ana_resp']:>10.3f} {100.0*r['err_resp_rel']:>8.2f} "
                  f"{r['sim_busy']:>10.3f} {r['ana_busy']:>10.3f} {100.0*r['err_busy_rel']:>8.2f}")
