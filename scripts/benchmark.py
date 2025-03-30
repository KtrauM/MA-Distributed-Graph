#!/usr/bin/env python3
import subprocess
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import shutil  # Add this import at the top of the file

# Graph types to test
GRAPH_TYPES = [
    "gnm-undirected",
    # "gnm-directed",
    # "gnp-undirected",
    # "gnp-directed",
    # "rgg2d",
    # "rgg3d",
    # "grid2d",
    # "rdg2d",
    # "rdg3d",
    # "ba",
    # "kronecker",
    # "rmat"
]

# Programs to benchmark
PROGRAMS = {
    "DistributedGrarrph_bfs": "../build/run_bfs",
    "DistributedGrarrph_cc": "../build/run_cc",
    "havoqgt_cc": "../baseline_frameworks/havoqgt/build/src/run_cc",
    "havoqgt_bfs": "../baseline_frameworks/havoqgt/build/src/run_bfs"
}

OUTPUT_KEY_TYPE = {
    "DistributedGrarrph_bfs": "run_bfs",
    "DistributedGrarrph_cc": "run_cc",
    "havoqgt_cc": "run_cc",
    "havoqgt_bfs": "run_bfs"
}

LOG_PER_PE_NODE_COUNT = 16
LOG_PER_PE_EDGE_COUNT = 19

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

def generate_graph(num_processors, graph_type, log_per_pe_node_count, log_per_pe_edge_count):
    """Generate the graph using the generate_graphs.py script."""
    cmd = f"python3 ./generate_graphs.py {graph_type} {num_processors} --nodes {log_per_pe_node_count} --edges {log_per_pe_edge_count}"
    print(f"Generating graph: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def ingest_graph_for_havoqgt(num_processors, graph_type):
    """Ingest the generated graph into HavoqGT format."""
    # Create havoqgt-databases directory if it doesn't exist
    Path("../graphs/havoqgt-databases").mkdir(exist_ok=True)
    
    # Generate the base graph filename
    base_graph = f"../graphs/{graph_type}_n={2**(LOG_PER_PE_NODE_COUNT + num_processors.bit_length() - 1)}_m={2**(LOG_PER_PE_EDGE_COUNT + num_processors.bit_length() - 1)}_pe"
    
    # Check if the database already exists
    db_path = f"../graphs/havoqgt-databases/{graph_type}_n={2**(LOG_PER_PE_NODE_COUNT + num_processors.bit_length() - 1)}_m={2**(LOG_PER_PE_EDGE_COUNT + num_processors.bit_length() - 1)}"
    if Path(db_path).exists():
        print(f"HavoqGT database already exists at {db_path}")
        return db_path
    
    # Each PE needs to read graph_file.{rank}
    graph_files = " ".join([f"{base_graph}.{i}" for i in range(num_processors)])
    
    # Ingest the graph using HavoqGT's ingest_edge_list
    ingest_cmd = f"mpirun -n {num_processors} ../baseline_frameworks/havoqgt/build/src/ingest_edge_list {graph_files} -o {db_path}"
    print(f"Ingesting graph for HavoqGT: {ingest_cmd}")
    subprocess.run(ingest_cmd, shell=True, check=True)
    return db_path

def run_benchmark(num_processors, graph_type, program):
    """Run the algorithm with specified number of processors and graph type, return runtime."""
    if program == "DistributedGrarrph_bfs" or program == "DistributedGrarrph_cc":
        cmd = f"mpirun -n {num_processors} {PROGRAMS[program]} -g 'type={graph_type};n={2**(LOG_PER_PE_NODE_COUNT + num_processors.bit_length() - 1)};m={2**(LOG_PER_PE_EDGE_COUNT + num_processors.bit_length() - 1)}'"
        # cmd = f"mpirun -n {num_processors} {PROGRAMS[program]} -g 'file;filename={graph_file};input_format=plain-edgelist'"
    else:
        # For HavoqGT programs, use the ingested database
        db_path = ingest_graph_for_havoqgt(num_processors, graph_type)
        cmd = f"mpirun -n {num_processors} {PROGRAMS[program]} -i {db_path}"
    
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    print("---------------STDOUT-----------------")
    print(result.stdout)
    print("---------------STDOUT-----------------")

    
    start_time = time.time()
    try:
        # Find the first '{' and last '}' to extract the JSON
        start_idx = result.stdout.find('{')
        end_idx = result.stdout.rfind('}') + 1
        print(start_idx, end_idx)
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON found in output")
        
        json_str = result.stdout[start_idx:end_idx]
        output_json = json.loads(json_str)
        # Extract runtime from the nested structure
        runtime = output_json["data"]["root"][OUTPUT_KEY_TYPE[program]]["statistics"]["max"][0]
        return runtime
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing program output: {e}")
        print(f"Full program output: {result.stdout}")
        raise
    end_time = time.time()
    return end_time - start_time

def generate_plots(runtimes, graph_type, program):
    """Generate performance plots for a specific graph type and program."""
    # Create benchmark_results directory if it doesn't exist
    Path("../benchmark_results").mkdir(exist_ok=True)
    
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
    ax1.set_title(f'Runtime vs Processors - {graph_type} - {program}')
    ax1.grid(True)
    
    # Speedup plot
    ax2.plot(processors, speedup, 'g-o')
    ax2.plot(processors, processors, 'r--', label='Ideal Speedup')
    ax2.set_xlabel('Number of Processors')
    ax2.set_ylabel('Speedup')
    ax2.set_title(f'Speedup vs Processors - {graph_type} - {program}')
    ax2.legend()
    ax2.grid(True)
    
    # Efficiency plot
    ax3.plot(processors, efficiency, 'm-o')
    ax3.set_xlabel('Number of Processors')
    ax3.set_ylabel('Efficiency')
    ax3.set_title(f'Efficiency vs Processors - {graph_type} - {program}')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../benchmark_results/performance_plots_{graph_type}_{program}.png')
    plt.close()

def generate_average_plots(all_runtimes, program):
    """Generate average performance plots across all graph types for a specific program."""
    # Create benchmark_results directory if it doesn't exist
    Path("../benchmark_results").mkdir(exist_ok=True)
    
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
    ax1.set_title(f'Average Runtime vs Processors - {program}')
    ax1.grid(True)
    
    # Speedup plot
    ax2.plot(processors, speedup, 'g-o')
    ax2.plot(processors, processors, 'r--', label='Ideal Speedup')
    ax2.set_xlabel('Number of Processors')
    ax2.set_ylabel('Average Speedup')
    ax2.set_title(f'Average Speedup vs Processors - {program}')
    ax2.legend()
    ax2.grid(True)
    
    # Efficiency plot
    ax3.plot(processors, efficiency, 'm-o')
    ax3.set_xlabel('Number of Processors')
    ax3.set_ylabel('Average Efficiency')
    ax3.set_title(f'Average Efficiency vs Processors - {program}')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../benchmark_results/performance_plots_average_{program}.png')
    plt.close()

def main():
    # Clear graphs directory at the start of the run
    graphs_dir = Path("../graphs")
    if graphs_dir.exists():
        print("Clearing graphs directory...")
        # Remove entire directory and its contents
        shutil.rmtree(graphs_dir)
    
    # Recreate the empty graphs directory
    graphs_dir.mkdir(exist_ok=True)

    # Create benchmark_results directory if it doesn't exist
    Path("../benchmark_results").mkdir(exist_ok=True)
    
    # Get processor list based on environment
    processors = get_processor_list()
    
    # Run benchmarks for each graph type
    for graph_type in GRAPH_TYPES:
        print(f"\nTesting graph type: {graph_type}")
        
        # Generate graphs once for each processor count
        for num_procs in processors:
            print(f"Generating graph with {num_procs} processors...")
            generate_graph(num_procs, graph_type, LOG_PER_PE_NODE_COUNT, LOG_PER_PE_EDGE_COUNT)
            
            # Run all programs on this graph
            for program in PROGRAMS:
                print(f"Running {program} with {num_procs} processors...")
                runtime = run_benchmark(num_procs, graph_type, program)
                
                # Store results
                if program not in all_runtimes:
                    all_runtimes[program] = {}
                if graph_type not in all_runtimes[program]:
                    all_runtimes[program][graph_type] = {}
                all_runtimes[program][graph_type][num_procs] = runtime
                print(f"Runtime for {program} with {num_procs} processors: {runtime:.2f} seconds")
    
    # Generate plots and save results for each program
    for program in PROGRAMS:
        print(f"\nGenerating plots for {program}...")
        
        # Generate plots for each graph type
        for graph_type in GRAPH_TYPES:
            generate_plots(all_runtimes[program][graph_type], graph_type, program)
        
        # Generate average plots
        generate_average_plots(all_runtimes[program], program)
        
        # Save results to JSON
        with open(f'../benchmark_results/benchmark_results_{program}.json', 'w') as f:
            json.dump(all_runtimes[program], f, indent=4)
    
    print("\nBenchmark completed. Results saved to:")
    for program in PROGRAMS:
        print(f"\nFor {program}:")
        print(f"- ../benchmark_results/benchmark_results_{program}.json")
        print(f"- ../benchmark_results/performance_plots_average_{program}.png")
        for graph_type in GRAPH_TYPES:
            print(f"- ../benchmark_results/performance_plots_{graph_type}_{program}.png")

if __name__ == "__main__":
    all_runtimes = {}  # Initialize the results dictionary
    main()