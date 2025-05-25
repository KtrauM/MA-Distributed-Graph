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
    "combblas-cc-fileio_25_05_24",
    "grarrph-cc_25_05_24",
    "havoqgt-cc-kagen_25_05_24"
]

# Map directory names to program names
FRAMEWORK_TO_PROGRAM = {
    "combblas-cc-fileio_25_05_24": "CombBLAS_cc",
    "grarrph-cc_25_05_24": "DistributedGrarrph_cc",
    "havoqgt-cc-kagen_25_05_24": "HavoqGT_cc"
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

def extract_runtime_and_numccs_from_log(framework, log_path):
    """Extract runtime and number of CCs from log file based on framework format."""
    with open(log_path, 'r') as f:
        output = f.read()

    if framework.startswith("combblas-cc"):
        for line in output.split('\n'):
            if "CC runtime:" in line:
                time_str = line.split("CC runtime:")[1].split("seconds")[0].strip()
                return float(time_str), None
        return None, None

    elif framework.startswith("grarrph-cc"):
        try:
            start_idx = output.find('{')
            end_idx = output.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                return None, None
            json_str = output[start_idx:end_idx]
            output_json = json.loads(json_str)
            key_type = "run_cc"
            runtime = output_json["data"]["root"][key_type]["statistics"]["max"][0]
            return runtime, None
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing program output for {log_path}: {e}")
            return None, None

    elif framework.startswith("havoqgt-cc"):
        # Look for the line: Num CCs = <num>, largest CC (approx) = <size>, Traversal Time = <runtime>
        numccs = None
        runtime = None
        for line in output.split('\n'):
            m = re.search(r"Num CCs = (\d+), largest CC \(approx\) = (\d+), Runtime = ([0-9.]+)", line)
            if m:
                numccs = int(m.group(1))
                runtime = float(m.group(3))
                break
        return runtime, numccs

    return None, None

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
            key = (graph_type, log_n, log_m)
            
            if key not in all_runtimes[program]:
                all_runtimes[program][key] = {}
                
            runtime, numccs = extract_runtime_and_numccs_from_log(framework, log_file)
            
            if runtime is not None:
                all_runtimes[program][key][cores] = {"runtime": runtime, "numccs": numccs}
                print(f"Runtime for {program}, {graph_type} n={log_n} m={log_m}, {cores} cores: {runtime:.2f} seconds, NumCCs: {numccs}")
            else:
                print(f"Warning: Could not extract runtime from {filename}")
    
    return all_runtimes

def save_to_csv(all_runtimes, output_dir=OUTPUT_DIR):
    """Save runtime data to CSV files."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save individual program results
    for program in all_runtimes:
        for key in all_runtimes[program]:
            graph_type, log_n, log_m = key
            key_str = f"{graph_type}_n{log_n}_m{log_m}"
            csv_path = os.path.join(output_dir, f"{program}_{key_str}_runtimes.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                if program == "HavoqGT_cc":
                    writer.writerow(['Cores', 'Runtime', 'NumCCs'])
                else:
                    writer.writerow(['Cores', 'Runtime'])
                for cores in sorted(all_runtimes[program][key].keys()):
                    entry = all_runtimes[program][key][cores]
                    if program == "HavoqGT_cc":
                        writer.writerow([cores, entry["runtime"], entry["numccs"]])
                    else:
                        writer.writerow([cores, entry["runtime"]])
                    
            print(f"Saved CSV: {csv_path}")
    
    # Create combined CSV files per graph instance and algorithm
    graph_keys = set()
    for program in all_runtimes:
        graph_keys.update(all_runtimes[program].keys())
    
    for key in graph_keys:
        graph_type, log_n, log_m = key
        key_str = f"{graph_type}_n{log_n}_m{log_m}"
        for algorithm in ALGORITHMS:
            programs_for_algo = [p for p in all_runtimes if p.split('_')[1] == algorithm]
            if not programs_for_algo:
                continue
                
            csv_path = os.path.join(output_dir, f"comparison_{key_str}_{algorithm}_runtimes.csv")
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Create header row with program names
                header = ['Cores']
                header.extend(programs_for_algo)
                writer.writerow(header)
                
                # Find all unique core counts
                all_cores = set()
                for program in programs_for_algo:
                    if key in all_runtimes[program]:
                        all_cores.update(all_runtimes[program][key].keys())
                
                # Write data rows
                for cores in sorted(all_cores):
                    row = [cores]
                    for program in programs_for_algo:
                        if key in all_runtimes[program] and cores in all_runtimes[program][key]:
                            row.append(all_runtimes[program][key][cores]["runtime"])
                        else:
                            row.append('')  # Empty if no data for this program/core count
                    writer.writerow(row)
            
            print(f"Saved combined CSV: {csv_path}")

def generate_plots(all_runtimes, output_dir=OUTPUT_DIR):
    """Generate performance plots for each program and graph instance."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate individual plots for each program
    for program in all_runtimes:
        for key in all_runtimes[program]:
            graph_type, log_n, log_m = key
            key_str = f"{graph_type}_n{log_n}_m{log_m}"
            processors = sorted(all_runtimes[program][key].keys())
            if not processors:
                continue
                
            times = [all_runtimes[program][key][p]["runtime"] for p in processors]
            
            # Create plots
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
            
            # Runtime plot
            ax1.plot(processors, times, 'b-o')
            ax1.set_xlabel('Number of Processors')
            ax1.set_ylabel('Runtime (seconds)')
            ax1.set_title(f'Runtime vs Processors - {graph_type} n={log_n} m={log_m} - {program}')
            ax1.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'performance_plots_{key_str}_{program}.png'))
            plt.close()
            
            print(f"Generated plot: performance_plots_{key_str}_{program}.png")

def generate_combined_comparison_plots(all_runtimes, output_dir=OUTPUT_DIR):
    """
    Generate combined comparison plots for each graph instance across all programs.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get all graph keys
    graph_keys = set()
    for program in all_runtimes:
        graph_keys.update(all_runtimes[program].keys())
    
    # Generate comparison plots for each graph instance and algorithm
    for key in graph_keys:
        graph_type, log_n, log_m = key
        key_str = f"{graph_type}_n{log_n}_m{log_m}"
        for algorithm in ALGORITHMS:
            # Collect programs for this algorithm
            runtimes_for_graph_algo = {}
            for program in all_runtimes:
                if program.split("_")[1] == algorithm and key in all_runtimes[program]:
                    runtimes_for_graph_algo[program] = all_runtimes[program][key]
            
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
                    
                times = [runtime_dict[p]["runtime"] for p in plot_processors]
                ax1.plot(plot_processors, times, marker='o', label=program)
            
            ax1.set_xlabel("Number of Processors")
            ax1.set_ylabel("Runtime (seconds)")
            ax1.set_title(f"Runtime Comparison on {graph_type} n={log_n} m={log_m}")
            ax1.legend()
            ax1.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_plots_{key_str}_{algorithm}.png"))
            plt.close()
            
            print(f"Generated comparison plot: comparison_plots_{key_str}_{algorithm}.png")

def generate_all_comparisons_combined_plot(all_runtimes, output_dir=OUTPUT_DIR):
    """
    Generate a single plot containing all comparison graphs arranged in a grid.
    Plots are organized by graph type, with each graph type appearing in the same row.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Get all graph keys
    graph_keys = set()
    for program in all_runtimes:
        graph_keys.update(all_runtimes[program].keys())
    
    # Group plots by graph type
    plots_by_graph_type = {}
    for key in graph_keys:
        graph_type, log_n, log_m = key
        if graph_type not in plots_by_graph_type:
            plots_by_graph_type[graph_type] = []
        plots_by_graph_type[graph_type].append((key, log_n, log_m))
    
    # Sort plots within each graph type by n and m
    for graph_type in plots_by_graph_type:
        plots_by_graph_type[graph_type].sort(key=lambda x: (x[1], x[2]))
    
    # Calculate grid dimensions
    num_rows = len(plots_by_graph_type)
    max_plots_per_row = max(len(plots) for plots in plots_by_graph_type.values())
    
    # Create a figure with subplots arranged in the calculated grid
    fig, axes = plt.subplots(num_rows, max_plots_per_row, figsize=(6*max_plots_per_row, 5*num_rows))
    if num_rows == 1:
        axes = np.array([axes])  # Make it iterable
    axes = axes.flatten()
    
    # Track which subplot we're currently using
    current_subplot = 0
    
    # Generate comparison plots for each graph type
    for graph_type, plots in plots_by_graph_type.items():
        for key, log_n, log_m in plots:
            key_str = f"{graph_type}_n{log_n}_m{log_m}"
            
            # Collect programs for this algorithm
            runtimes_for_graph_algo = {}
            for program in all_runtimes:
                if program.split("_")[1] == "cc" and key in all_runtimes[program]:
                    runtimes_for_graph_algo[program] = all_runtimes[program][key]
            
            if not runtimes_for_graph_algo:
                continue
            
            # Get all processor counts
            processors_set = []
            for runtime_dict in runtimes_for_graph_algo.values():
                processors_set.extend(runtime_dict.keys())
            processors = sorted(set(processors_set))
            
            if not processors:
                continue
            
            # Plot runtime for each program in the current subplot
            ax = axes[current_subplot]
            for program, runtime_dict in runtimes_for_graph_algo.items():
                # Ensure we plot in the order of processors from our full list
                plot_processors = [p for p in processors if p in runtime_dict]
                if not plot_processors:
                    continue
                    
                times = [runtime_dict[p]["runtime"] for p in plot_processors]
                ax.plot(plot_processors, times, marker='o', label=program)
            
            ax.set_xlabel("Number of Processors")
            ax.set_ylabel("Runtime (seconds)")
            ax.set_title(f"Runtime Comparison on {graph_type} n={log_n} m={log_m}")
            ax.legend()
            ax.grid(True)
            
            current_subplot += 1
    
    # Remove any unused subplots
    for i in range(current_subplot, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_comparisons_combined.png"))
    plt.close()
    
    print(f"Generated combined comparison plot: all_comparisons_combined.png")

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
    
    print("\nGenerating all comparisons combined plot...")
    generate_all_comparisons_combined_plot(all_runtimes)
    
    print("\nAnalysis completed. Results saved to:")
    print(f"- CSV files: {OUTPUT_DIR}/*.csv")
    print(f"- Performance plots: {OUTPUT_DIR}/performance_plots_*.png")
    print(f"- Comparison plots: {OUTPUT_DIR}/comparison_plots_*.png")
    print(f"- All comparisons combined plot: {OUTPUT_DIR}/all_comparisons_combined.png")

if __name__ == "__main__":
    main()