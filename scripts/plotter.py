#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import re
import csv
import glob

# Framework directories under experiment_data
FRAMEWORK_DIRS = [
    "combblas-cc_25_04_29",
    "graph_gen_25_04_29",
    "grarrph-cc_25_04_29",
    "havoqgt-cc_25_04_29"
]

# Map directory names to program names
FRAMEWORK_TO_PROGRAM = {
    "combblas-cc_25_04_29": "CombBLAS_cc",
    "grarrph-cc_25_04_29": "DistributedGrarrph_cc",
    "havoqgt-cc_25_04_29": "havoqgt_cc"
}

# Algorithms
ALGORITHMS = ["cc"]  # Based on the directories provided

# Path configuration - assuming script is in the scripts directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "experiment_data")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "benchmark_results")

def parse_filename(filename):
    """Extract graph type, node count, edge count, and core count from filename."""
    # Base pattern for standard filenames - handles both log and error-log files
    pattern = r"([\w-]+)_n(\d+)_m(\d+)(?:_(\w+\d+))?_\w+-cores(\d+)(?:-error)?-log\.txt"
    match = re.match(pattern, filename)
    if match:
        graph_type = match.group(1)
        log_n = int(match.group(2))
        log_m = int(match.group(3))
        extra_param = match.group(4)  # This could be None if not present
        cores = int(match.group(5))
        
        # If there's an extra parameter, add it to the graph type for uniqueness
        if extra_param:
            graph_type = f"{graph_type}_{extra_param}"
            
        return graph_type, log_n, log_m, cores
    
    print(f"Failed to match pattern on: {filename}")
    return None

def extract_runtime_from_log(framework, log_path):
    """Extract runtime from log file based on framework format."""
    with open(log_path, 'r') as f:
        output = f.read()
    
    if framework == "combblas-cc_25_04_29":
        for line in output.split('\n'):
            if "CC runtime:" in line:
                time_str = line.split("CC runtime:")[1].split("seconds")[0].strip()
                return float(time_str)
        return None
    
    elif framework in ["grarrph-cc_25_04_29", "havoqgt-cc_25_04_29"]:
        try:
            # Find the first '{' and last '}' to extract the JSON
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = output[start_idx:end_idx]
            output_json = json.loads(json_str)
            
            # Determine the output key type based on framework
            if framework == "grarrph-cc_25_04_29":
                key_type = "run_cc"
            elif framework == "havoqgt-cc_25_04_29":
                key_type = "run_cc"
            else:
                return None
                
            # Extract runtime from the nested structure
            runtime = output_json["data"]["root"][key_type]["statistics"]["max"][0]
            return runtime
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing program output for {log_path}: {e}")
            return None
    
    return None

def collect_runtime_data(base_dir=DATA_DIR):
    """Collect runtime data from log files."""
    all_runtimes = {}
    
    for framework in FRAMEWORK_DIRS:
        program = FRAMEWORK_TO_PROGRAM.get(framework)
        if not program:
            continue
            
        if program not in all_runtimes:
            all_runtimes[program] = {}
            
        output_dir = os.path.join(base_dir, framework, "output")
        if not os.path.exists(output_dir):
            print(f"Warning: Directory not found: {output_dir}")
            continue
            
        # Only look for regular log files (not error logs) for runtime data
        log_files = glob.glob(os.path.join(output_dir, "*-log.txt"))
        print(f"Found {len(log_files)} log files in {output_dir}")
        
        for log_file in log_files:
            filename = os.path.basename(log_file)
            file_info = parse_filename(filename)
            
            if not file_info:
                print(f"Warning: Could not parse filename: {filename}")
                continue
                
            graph_type, log_n, log_m, cores = file_info
            
            if graph_type not in all_runtimes[program]:
                all_runtimes[program][graph_type] = {}
                
            runtime = extract_runtime_from_log(framework, log_file)
            
            if runtime is not None:
                all_runtimes[program][graph_type][cores] = runtime
                print(f"Runtime for {program}, {graph_type}, {cores} cores: {runtime:.2f} seconds")
            else:
                print(f"Warning: Could not extract runtime from {filename}")
    
    return all_runtimes

def save_to_csv(all_runtimes, output_dir=OUTPUT_DIR):
    """Save runtime data to CSV files."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save individual program results
    for program in all_runtimes:
        for graph_type in all_runtimes[program]:
            csv_path = os.path.join(output_dir, f"{program}_{graph_type}_runtimes.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Cores', 'Runtime'])
                
                for cores in sorted(all_runtimes[program][graph_type].keys()):
                    runtime = all_runtimes[program][graph_type][cores]
                    writer.writerow([cores, runtime])
                    
            print(f"Saved CSV: {csv_path}")
    
    # Create combined CSV files per graph type and algorithm
    graph_types = set()
    for program in all_runtimes:
        graph_types.update(all_runtimes[program].keys())
    
    for graph_type in graph_types:
        for algorithm in ALGORITHMS:
            programs_for_algo = [p for p in all_runtimes if p.split('_')[1] == algorithm]
            if not programs_for_algo:
                continue
                
            csv_path = os.path.join(output_dir, f"comparison_{graph_type}_{algorithm}_runtimes.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Create header row with program names
                header = ['Cores']
                header.extend(programs_for_algo)
                writer.writerow(header)
                
                # Find all unique core counts
                all_cores = set()
                for program in programs_for_algo:
                    if graph_type in all_runtimes[program]:
                        all_cores.update(all_runtimes[program][graph_type].keys())
                
                # Write data rows
                for cores in sorted(all_cores):
                    row = [cores]
                    for program in programs_for_algo:
                        if graph_type in all_runtimes[program] and cores in all_runtimes[program][graph_type]:
                            row.append(all_runtimes[program][graph_type][cores])
                        else:
                            row.append('')  # Empty if no data for this program/core count
                    writer.writerow(row)
            
            print(f"Saved combined CSV: {csv_path}")

def generate_plots(all_runtimes, output_dir=OUTPUT_DIR):
    """Generate performance plots for each program and graph type."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate individual plots for each program
    for program in all_runtimes:
        for graph_type in all_runtimes[program]:
            processors = sorted(all_runtimes[program][graph_type].keys())
            if not processors:
                continue
                
            times = [all_runtimes[program][graph_type][p] for p in processors]
            
            # Create plots
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            
            # Runtime plot
            ax1.plot(processors, times, 'b-o')
            ax1.set_xlabel('Number of Processors')
            ax1.set_ylabel('Runtime (seconds)')
            ax1.set_title(f'Runtime vs Processors - {graph_type} - {program}')
            ax1.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'performance_plots_{graph_type}_{program}.png'))
            plt.close()
            
            print(f"Generated plot: performance_plots_{graph_type}_{program}.png")

def generate_combined_comparison_plots(all_runtimes, output_dir=OUTPUT_DIR):
    """
    Generate combined comparison plots for each graph type across all programs.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get all graph types
    graph_types = set()
    for program in all_runtimes:
        graph_types.update(all_runtimes[program].keys())
    
    # Generate comparison plots for each graph type and algorithm
    for graph_type in graph_types:
        for algorithm in ALGORITHMS:
            # Collect programs for this algorithm
            runtimes_for_graph_algo = {}
            for program in all_runtimes:
                if program.split("_")[1] == algorithm and graph_type in all_runtimes[program]:
                    runtimes_for_graph_algo[program] = all_runtimes[program][graph_type]
            
            if not runtimes_for_graph_algo:
                continue
            
            # Get all processor counts
            processors_set = []
            for runtime_dict in runtimes_for_graph_algo.values():
                processors_set.extend(runtime_dict.keys())
            processors = sorted(set(processors_set))
            
            if not processors:
                continue
            
            # Create combined plots
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            
            # Plot runtime for each program
            for program, runtime_dict in runtimes_for_graph_algo.items():
                # Ensure we plot in the order of processors from our full list
                plot_processors = [p for p in processors if p in runtime_dict]
                if not plot_processors:
                    continue
                    
                times = [runtime_dict[p] for p in plot_processors]
                ax1.plot(plot_processors, times, marker='o', label=program)
            
            ax1.set_xlabel("Number of Processors")
            ax1.set_ylabel("Runtime (seconds)")
            ax1.set_title(f"Runtime Comparison on {graph_type}")
            ax1.legend()
            ax1.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_plots_{graph_type}_{algorithm}.png"))
            plt.close()
            
            print(f"Generated comparison plot: comparison_plots_{graph_type}_{algorithm}.png")

def main():
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"Script directory: {SCRIPT_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("Collecting runtime data from log files...")
    all_runtimes = collect_runtime_data()
    
    if not all_runtimes:
        print("No runtime data found. Please check the experiment_data directory.")
        return
    
    print("\nSaving runtime data to CSV files...")
    save_to_csv(all_runtimes)
    
    print("\nGenerating individual performance plots...")
    generate_plots(all_runtimes)
    
    print("\nGenerating combined comparison plots...")
    generate_combined_comparison_plots(all_runtimes)
    
    print("\nAnalysis completed. Results saved to:")
    print(f"- CSV files: {OUTPUT_DIR}/*.csv")
    print(f"- Performance plots: {OUTPUT_DIR}/performance_plots_*.png")
    print(f"- Comparison plots: {OUTPUT_DIR}/comparison_plots_*.png")

if __name__ == "__main__":
    main()