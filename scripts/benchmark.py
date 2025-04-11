#!/usr/bin/env python3
import subprocess
import math
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import shutil

# Graph types to test
GRAPH_TYPES = [
    "rhg", # gamma 2.8
    "rmat", 
    "gnm-undirected",
    "gnm-directed",
    # "gnp-undirected",
    # "gnp-directed",
    "rgg2d",
    "rgg3d",
    # "grid2d",
    # "rdg2d",
    # "rdg3d",
    # "ba",
    # "kronecker",
]

# Programs to benchmark
PROGRAMS = {
    "DistributedGrarrph_bfs": "../build/run_bfs",
    "DistributedGrarrph_cc": "../build/run_cc",
    "havoqgt_cc": "../baseline_frameworks/havoqgt/build/src/run_cc",
    "havoqgt_bfs": "../baseline_frameworks/havoqgt/build/src/run_bfs",
    "CombBLAS_bfs": "../baseline_frameworks/CombBLAS/build/Applications/tdbfs",
    "CombBLAS_cc": "../baseline_frameworks/CombBLAS/build/Applications/lacc"
}

ALGORITHMS = set([program.split("_")[1] for program in PROGRAMS])

OUTPUT_KEY_TYPE = {
    "DistributedGrarrph_bfs": "run_bfs",
    "DistributedGrarrph_cc": "run_cc",
    "havoqgt_cc": "run_cc",
    "havoqgt_bfs": "run_bfs"
}

LOG_PER_PE_NODE_COUNT = 4
LOG_PER_PE_EDGE_COUNT = 5

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

def generate_graph(num_processors, graph_type, log_per_pe_node_count, log_per_pe_edge_count, distributed_output=True):
    """Generate the graph using the generate_graphs.py script."""
    if distributed_output:
        cmd = f"python3 ./generate_graphs.py {graph_type} {num_processors} --nodes {log_per_pe_node_count} --edges {log_per_pe_edge_count} --distributed-output"
    else:
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
        generator_string = f"type={graph_type};n={2**(LOG_PER_PE_NODE_COUNT + num_processors.bit_length() - 1)};m={2**(LOG_PER_PE_EDGE_COUNT + num_processors.bit_length() - 1)}"
        if graph_type == "rhg":
            generator_string += ";gamma=2.8"
        cmd = f"mpirun -n {num_processors} {PROGRAMS[program]} -g '{generator_string}'"
        # cmd = f"mpirun -n {num_processors} {PROGRAMS[program]} -g 'file;filename={graph_file};input_format=plain-edgelist'"
    elif program == "havoqgt_bfs" or program == "havoqgt_cc":
        # For HavoqGT programs, use the ingested database
        db_path = ingest_graph_for_havoqgt(num_processors, graph_type)
        cmd = f"mpirun -n {num_processors} {PROGRAMS[program]} -i {db_path}"
    elif program == "CombBLAS_bfs" or program == "CombBLAS_cc":
        graph_path = f"../graphs/{graph_type}_n={2**(LOG_PER_PE_NODE_COUNT + num_processors.bit_length() - 1)}_m={2**(LOG_PER_PE_EDGE_COUNT + num_processors.bit_length() - 1)}-singlefile"
        if program == "CombBLAS_bfs":
            cmd = f"mpirun -n {num_processors} {PROGRAMS[program]} Input {graph_path}"
        elif program == "CombBLAS_cc":
            cmd = f"mpirun -n {num_processors} {PROGRAMS[program]} -I triples -M {graph_path}"
    else:
        raise ValueError(f"Unknown program: {program}")

    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if program == "CombBLAS_bfs":
        # Parse CombBLAS BFS output to get minimum BFS time
        bfs_times = []
        for line in output.split('\n'):
            print(line)
            if "BFS time:" in line:
                time_str = line.split("BFS time:")[1].split("seconds")[0].strip()
                bfs_times.append(float(time_str))
        
        if not bfs_times:
            raise ValueError("Could not find any 'BFS time' entries in CombBLAS_bfs output")
        
        return min(bfs_times)
        
    elif program == "CombBLAS_cc":
        # Parse CombBLAS CC output to get total time
        for line in output.split('\n'):
            if "Total time:" in line:
                runtime = float(line.split(":")[1].strip())
                return runtime
        raise ValueError("Could not find 'Total time' in CombBLAS_cc output")
    
    else:
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
            print(output_json["data"]["root"])
            runtime = output_json["data"]["root"][OUTPUT_KEY_TYPE[program]]["statistics"]["max"][0]
            return runtime
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing program output: {e}")
            print(f"Full program output: {result.stdout}")
            raise

def generate_plots(runtimes, graph_type, program):
    """Generate performance plots for a specific graph type and program."""
    # Create benchmark_results directory if it doesn't exist
    Path("../benchmark_results").mkdir(exist_ok=True)
    
    processors = sorted(runtimes.keys())
    times = [runtimes[p] for p in processors]
    
    # Calculate speedup and efficiency
    base_time = times[0]  # Time with 1 processor
    speedup = [base_time / t for t in times]
    efficiency = [s/p for s, p in zip(speedup, processors)]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Runtime plot
    ax1.plot(processors, times, 'b-o')
    ax1.set_xlabel('Number of Processors')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title(f'Runtime vs Processors - {graph_type} - {program}')
    ax1.grid(True)
    
    # Efficiency plot
    ax2.plot(processors, efficiency, 'm-o')
    ax2.set_xlabel('Number of Processors')
    ax2.set_ylabel('Efficiency')
    ax2.set_title(f'Efficiency vs Processors - {graph_type} - {program}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../benchmark_results/performance_plots_{graph_type}_{program}.png')
    plt.close()

def generate_average_plots(all_runtimes, program):
    """Generate average performance plots across all graph types for a specific program."""
    # Create benchmark_results directory if it doesn't exist
    Path("../benchmark_results").mkdir(exist_ok=True)
    
    processors = sorted(next(iter(all_runtimes.values())).keys())
    
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Runtime plot
    ax1.plot(processors, avg_times, 'b-o')
    ax1.set_xlabel('Number of Processors')
    ax1.set_ylabel('Average Runtime (seconds)')
    ax1.set_title(f'Average Runtime vs Processors - {program}')
    ax1.grid(True)
    
    # Efficiency plot
    ax2.plot(processors, efficiency, 'm-o')
    ax2.set_xlabel('Number of Processors')
    ax2.set_ylabel('Average Efficiency')
    ax2.set_title(f'Average Efficiency vs Processors - {program}')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'../benchmark_results/performance_plots_average_{program}.png')
    plt.close()

def generate_combined_comparison_plots(graph_type, algorithm, runtimes_all_programs):
    """
    Generate a combined comparison plot for a given graph type.
    Each program's runtime and efficiency are plotted on the same figure.
    """

    processors_set = []
    for runtime_dict in runtimes_all_programs.values():
        processors_set.extend(runtime_dict.keys())
    processors = sorted(set(processors_set))
    
    # Create combined plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot runtime for each program
    for program, runtime_dict in runtimes_all_programs.items():
        # Ensure we plot in the order of processors from our full list
        plot_processors = [p for p in processors if p in runtime_dict]
        times = [runtime_dict[p] for p in plot_processors]
        ax1.plot(plot_processors, times, marker='o', label=program)
    ax1.set_xlabel("Number of Processors")
    ax1.set_ylabel("Runtime (seconds)")
    ax1.set_title(f"Runtime Comparison on {graph_type}")
    ax1.legend()
    ax1.grid(True)
    
    # Plot efficiency for each program
    for program, runtime_dict in runtimes_all_programs.items():
        plot_processors = [p for p in processors if p in runtime_dict]
        times = [runtime_dict[p] for p in plot_processors]
        base_time = times[0]
        speedup = [base_time / t if t != 0 else 0 for t in times]
        efficiency = [s/p for s, p in zip(speedup, plot_processors)]
        ax2.plot(plot_processors, efficiency, marker='o', label=program)
    ax2.set_xlabel("Number of Processors")
    ax2.set_ylabel("Efficiency")
    ax2.set_title(f"Efficiency Comparison on {graph_type}-{algorithm}")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"../benchmark_results/comparison_plots_{graph_type}_{algorithm}.png")
    plt.close()

def generate_algorithm_comparison_plots(all_runtimes):
    """
    Generate comparison plots that average results for each algorithm type (BFS, CC)
    across all graph types. This shows which implementation of each algorithm 
    performs best on average.
    """
    # Get all available algorithms
    algorithms = set()
    for program in all_runtimes.keys():
        algorithm = program.split("_")[1]
        algorithms.add(algorithm)
    
    # For each algorithm type (bfs, cc)
    for algorithm in algorithms:
        print(f"Generating algorithm comparison plots for {algorithm}...")
        
        # Group runtimes by framework and processor count
        framework_runtimes = {}
        all_processors = set()
        
        # Collect data for this algorithm type
        for program, graph_data in all_runtimes.items():
            if program.split("_")[1] == algorithm:
                framework = program.split("_")[0]
                framework_runtimes[framework] = {}
                
                # Average across all graph types for each processor count
                for graph_type, proc_data in graph_data.items():
                    for proc_count, runtime in proc_data.items():
                        all_processors.add(proc_count)
                        if proc_count not in framework_runtimes[framework]:
                            framework_runtimes[framework][proc_count] = []
                        framework_runtimes[framework][proc_count].append(runtime)
        
        # Convert lists to averages
        for framework in framework_runtimes:
            for proc_count in framework_runtimes[framework]:
                framework_runtimes[framework][proc_count] = np.mean(framework_runtimes[framework][proc_count])
        
        # Create plots
        processors = sorted(all_processors)
        
        # Runtime comparison plot
        plt.figure(figsize=(10, 8))
        for framework, runtime_data in framework_runtimes.items():
            plot_processors = [p for p in processors if p in runtime_data]
            times = [runtime_data[p] for p in plot_processors]
            plt.plot(plot_processors, times, marker='o', label=framework)
        
        plt.xlabel("Number of Processors")
        plt.ylabel("Average Runtime (seconds)")
        plt.title(f"Average Runtime Comparison for {algorithm.upper()} Algorithm")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../benchmark_results/algorithm_comparison_{algorithm}_runtime.png")
        plt.close()
        
        # Efficiency comparison plot
        plt.figure(figsize=(10, 8))
        for framework, runtime_data in framework_runtimes.items():
            plot_processors = [p for p in processors if p in runtime_data]
            times = [runtime_data[p] for p in plot_processors]
            if times:  # Check if we have data
                base_time = times[0]
                speedup = [base_time / t if t != 0 else 0 for t in times]
                efficiency = [s/p for s, p in zip(speedup, plot_processors)]
                plt.plot(plot_processors, efficiency, marker='o', label=framework)
        
        plt.xlabel("Number of Processors")
        plt.ylabel("Average Efficiency")
        plt.title(f"Average Efficiency Comparison for {algorithm.upper()} Algorithm")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../benchmark_results/algorithm_comparison_{algorithm}_efficiency.png")
        plt.close()

def main():
    # Clear graphs directory at the start of the run
    graphs_dir = Path("../graphs")
    # Create benchmark_results directory if it doesn't exist
    Path("../benchmark_results").mkdir(exist_ok=True)
    
    # Get processor list based on environment
    processors = get_processor_list()
    
    # Run benchmarks for each graph type
    for graph_type in GRAPH_TYPES:
        print(f"\nTesting graph type: {graph_type}")
        # Generate graphs once for each processor count
        for num_procs in processors:
            # Clear graphs directory to save space
            if graphs_dir.exists():
                print("Clearing graphs directory...")
                # Remove entire directory and its contents
                shutil.rmtree(graphs_dir)
            
            # Recreate the empty graphs directory
            graphs_dir.mkdir(exist_ok=True)
            print(f"Generating graph with {num_procs} processors...")
            generate_graph(num_procs, graph_type, LOG_PER_PE_NODE_COUNT, LOG_PER_PE_EDGE_COUNT, distributed_output=True)
            generate_graph(num_procs, graph_type, LOG_PER_PE_NODE_COUNT, LOG_PER_PE_EDGE_COUNT, distributed_output=False)
            
            # Run all programs on this graph
            for program in PROGRAMS:
                if (program == "CombBLAS_bfs" or program == "CombBLAS_cc") and math.isqrt(num_procs) * math.isqrt(num_procs) != num_procs:
                    print(f"Skipping {program} with {num_procs} processors because CombBLAS only works on a square logical processor grid")
                    continue
                
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
    
    # Generate combined comparison plots for each graph type across programs
    for graph_type in GRAPH_TYPES:
        for algorithm in ALGORITHMS:    
            runtimes_for_graph_algo = {}
            for program in PROGRAMS:
                if program.split("_")[1] == algorithm and graph_type in all_runtimes.get(program, {}):
                    runtimes_for_graph_algo[program] = all_runtimes[program][graph_type]
            if runtimes_for_graph_algo:
                generate_combined_comparison_plots(graph_type, algorithm, runtimes_for_graph_algo)
                print(f"Combined comparison plot generated for graph type {graph_type}-{algorithm}")
    
    # Generate algorithm comparison plots (averaged over all graph types)
    generate_algorithm_comparison_plots(all_runtimes)
    print("Algorithm comparison plots generated")
    
    print("\nBenchmark completed. Results saved to:")
    for program in PROGRAMS:
        print(f"\nFor {program}:")
        print(f"- ../benchmark_results/benchmark_results_{program}.json")
        print(f"- ../benchmark_results/performance_plots_average_{program}.png")
        for graph_type in GRAPH_TYPES:
            print(f"- ../benchmark_results/performance_plots_{graph_type}_{program}.png")

    print("\nComparison plots:")
    for graph_type in GRAPH_TYPES:
        for algorithm in ALGORITHMS:
            print(f"- ../benchmark_results/comparison_plots_{graph_type}_{algorithm}.png")
    
    print("\nAlgorithm comparison plots:")
    for algorithm in ALGORITHMS:
        print(f"- ../benchmark_results/algorithm_comparison_{algorithm}_runtime.png")
        print(f"- ../benchmark_results/algorithm_comparison_{algorithm}_efficiency.png")

if __name__ == "__main__":
    all_runtimes = {}  # Initialize the results dictionary
    main()