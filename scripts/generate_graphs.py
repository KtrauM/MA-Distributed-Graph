#!/usr/bin/env python3
import subprocess
from pathlib import Path
import time
import argparse

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

def generate_graph(graph_type, num_nodes, num_edges, num_procs):
    """Generate a graph using KaGen with the specified parameters."""
    # Create output directory if it doesn't exist
    output_dir = Path("../graphs/")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{graph_type}_n={num_nodes}_m={num_edges}_pe"
    # Create output files
    # for i in range(num_procs):
    #     output_file_with_pe = output_dir / f"{graph_type}_n={num_nodes}_m={num_edges}_pe.{i}"
    #     output_file_with_pe.touch(exist_ok=True)

    # Construct KaGen command
    cmd = [
        "mpirun",
        "-n", str(num_procs),
        "../build/external/KaGen/app/KaGen",
        graph_type,
        "-n", str(num_nodes),
        "-m", str(num_edges),
        "-f", "edgelist",
        "-o", str(output_file),
        "--distributed-output"
    ]
    
    print(f"Generating {graph_type} graph with {num_nodes} nodes and {num_edges} edges...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully generated: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating graph: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate distributed graphs using KaGen')
    parser.add_argument('graph_type', choices=GRAPH_TYPES, help='Type of graph to generate')
    parser.add_argument('num_procs', type=int, help='Number of processors to use')
    parser.add_argument('--nodes', '-n', type=int, help='Number of nodes (in log2) (overrides default calculation)')
    parser.add_argument('--edges', '-m', type=int, help='Number of edges (in log2) (overrides default calculation)')
    args = parser.parse_args()

    num_nodes = 2**(args.nodes + args.num_procs.bit_length() - 1)
    num_edges = 2**(args.edges + args.num_procs.bit_length() - 1)
    
    generate_graph(args.graph_type, num_nodes, num_edges, args.num_procs)

if __name__ == "__main__":
    main()
