#!/usr/bin/env python3
import subprocess
from pathlib import Path
import time
GRAPH_TYPES = [
    # "gnm-undirected",
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
    "rmat"
]

def generate_graph(graph_type, num_nodes, num_edges, num_procs):
    """Generate a graph using KaGen with the specified parameters."""
    # Create output directory if it doesn't exist
    output_dir = Path("graphs/kagen_output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"{graph_type}-n={num_nodes}-m={num_edges}"
    
    # Construct KaGen command
    cmd = [
        "mpirun",
        "-n", str(num_procs),
        "./build/_deps/kagen-build/app/KaGen",
        graph_type,
        "-n", str(num_nodes),
        "-m", str(num_edges),
        "-f", "edgelist",
        "-o", str(output_file),
        # "-o", "output.txt",	
        "--distributed-output"
    ]
    
    print(f"Generating {graph_type} graph with {num_nodes} nodes and {num_edges} edges...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully generated: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating graph: {e}")

def main():
    # Generate graphs for different processor counts
    processor_counts = [1, 2]
    
    for num_procs in processor_counts:
        # Calculate n and m based on processor count (same formula as in benchmark.py)
        num_nodes = 2**(4 + num_procs.bit_length() - 1)
        num_edges = 2**(6 + num_procs.bit_length() - 1)
        
        for graph_type in GRAPH_TYPES[0:1]:
            generate_graph(graph_type, num_nodes, num_edges, num_procs)
            time.sleep(2)

if __name__ == "__main__":
    main()
