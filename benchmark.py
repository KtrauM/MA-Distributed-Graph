#!/usr/bin/env python3
import subprocess
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# Graph types to test
GRAPH_TYPES = [
    "gnm-undirected",
    "gnm-directed",
    "gnp-undirected",
    "gnp-directed",
    "rgg2d",
    "rgg3d",
    "grid2d",
    "rdg2d",
    "rdg3d",
    "ba",
    "kronecker",
    "rmat"
]

def is_slurm_environment():
    """Check if running in SLURM environment."""
    return 'SLURM_JOB_ID' in os.environ

def get_processor_list():
    """Get list of processors to test based on environment."""
    if is_slurm_environment():
        return [1, 2, 4, 8, 16, 32, 64]
    else:
        print("Running locally - limiting processor count to 8")
        return [1, 2, 4, 8]

def run_benchmark(num_processors, graph_type):
    """Run the algorithm with specified number of processors and graph type, return runtime."""
    cmd = f"mpirun -n {num_processors} ./build/myapp -g 'type={graph_type};n={2**(18 + num_processors.bit_length() - 1)};m={2**(21 + num_processors.bit_length() - 1)}'"
    print(f"Running command: {cmd}"	)
    start_time = time.time()
    subprocess.run(cmd, shell=True, check=True)
    end_time = time.time()
    return end_time - start_time

def generate_plots(runtimes, graph_type):
    """Generate performance plots for a specific graph type."""
    # Create benchmark_results directory if it doesn't exist
    Path("benchmark_results").mkdir(exist_ok=True)
    
    processors = list(runtimes.keys())
    times = list(runtimes.values())
    
    # Calculate speedup and efficiency
    base_time = times[0]  # Time with 1 processor
    speedup = [base_time / t for t in times]
    efficiency = [s/p for s, p in zip(speedup, processors)]
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Runtime plot
    ax1.plot(processors, times, 'b-o')
    ax1.set_xlabel('Number of Processors')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title(f'Runtime vs Processors - {graph_type}')
    ax1.grid(True)
    
    # Speedup plot
    ax2.plot(processors, speedup, 'g-o')
    ax2.plot(processors, processors, 'r--', label='Ideal Speedup')
    ax2.set_xlabel('Number of Processors')
    ax2.set_ylabel('Speedup')
    ax2.set_title(f'Speedup vs Processors - {graph_type}')
    ax2.legend()
    ax2.grid(True)
    
    # Efficiency plot
    ax3.plot(processors, efficiency, 'm-o')
    ax3.set_xlabel('Number of Processors')
    ax3.set_ylabel('Efficiency')
    ax3.set_title(f'Efficiency vs Processors - {graph_type}')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'benchmark_results/performance_plots_{graph_type}.png')
    plt.close()

def generate_average_plots(all_runtimes):
    """Generate average performance plots across all graph types."""
    # Create benchmark_results directory if it doesn't exist
    Path("benchmark_results").mkdir(exist_ok=True)
    
    processors = list(next(iter(all_runtimes.values())).keys())
    
    # Calculate averages for each number of processors
    avg_times = []
    for num_procs in processors:
        times = np.array([runtimes[num_procs] for runtimes in all_runtimes.values()])
        avg_times.append(np.mean(times))
    
    # Calculate speedup and efficiency
    base_time = avg_times[0]
    speedup = [base_time / t for t in avg_times]
    efficiency = [s/p for s, p in zip(speedup, processors)]
    
    # Create plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Runtime plot
    ax1.plot(processors, avg_times, 'b-o')
    ax1.set_xlabel('Number of Processors')
    ax1.set_ylabel('Average Runtime (seconds)')
    ax1.set_title('Average Runtime vs Processors')
    ax1.grid(True)
    
    # Speedup plot
    ax2.plot(processors, speedup, 'g-o')
    ax2.plot(processors, processors, 'r--', label='Ideal Speedup')
    ax2.set_xlabel('Number of Processors')
    ax2.set_ylabel('Average Speedup')
    ax2.set_title('Average Speedup vs Processors')
    ax2.legend()
    ax2.grid(True)
    
    # Efficiency plot
    ax3.plot(processors, efficiency, 'm-o')
    ax3.set_xlabel('Number of Processors')
    ax3.set_ylabel('Average Efficiency')
    ax3.set_title('Average Efficiency vs Processors')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results/performance_plots_average.png')
    plt.close()

def main():
    # Create benchmark_results directory if it doesn't exist
    Path("benchmark_results").mkdir(exist_ok=True)
    
    # Get processor list based on environment
    processors = get_processor_list()
    all_runtimes = {}
    
    print("Starting benchmark runs...")
    for graph_type in GRAPH_TYPES:
        print(f"\nTesting graph type: {graph_type}")
        runtimes = {}
        for num_procs in processors:
            print(f"Running with {num_procs} processors...")
            runtime = run_benchmark(num_procs, graph_type)
            runtimes[num_procs] = runtime
            print(f"Runtime with {num_procs} processors: {runtime:.2f} seconds")
        
        all_runtimes[graph_type] = runtimes
        
        # Generate plots for this graph type
        print(f"Generating plots for {graph_type}...")
        generate_plots(runtimes, graph_type)
    
    # Generate average plots
    print("\nGenerating average plots...")
    generate_average_plots(all_runtimes)
    
    # Save all results to JSON
    with open('benchmark_results/benchmark_results.json', 'w') as f:
        json.dump(all_runtimes, f, indent=4)
    
    print("\nBenchmark completed. Results saved to:")
    print("- benchmark_results/benchmark_results.json")
    for graph_type in GRAPH_TYPES:
        print(f"- benchmark_results/performance_plots_{graph_type}.png")
    print("- benchmark_results/performance_plots_average.png")

if __name__ == "__main__":
    main()