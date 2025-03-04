#!/bin/bash
#SBATCH --job-name=dist_graph          # Name of the job
#SBATCH --output=output_%j.log         # Output file (with job ID in the name)
#SBATCH --error=error_%j.log           # Error file (with job ID in the name)
#SBATCH --ntasks=1                     # Number of tasks (processes)
#SBATCH --cpus-per-task=4              # Number of CPU cores per task
#SBATCH --mem=8G                       # Memory required per node
#SBATCH --time=01:00:00                # Wall clock time limit (hh:mm:ss)
#SBATCH --partition=naur               # Partition (queue) to use

set -e

NP=${1:-64}

local_nodes=$((2**18))
local_edges=$((2**21))

global_nodes=$((NP * local_nodes))
global_edges=$((NP * local_edges))

# TODO: Commented out algorithms are not working.
algorithms=(
  # "gnm_undirected"
  # "gnm_directed"
  # "gnp_undirected"
  # "gnp_directed"
  "rgg2d"
  "rgg3d"
  "grid2d"
  # "grid3d"
  "rdg2d"
  "rdg3d"
  # "rhg"
  "ba"
  "kronecker"
  "rmat"
)

for algo in "${algorithms[@]}"; do
  options="${algo};n=${global_nodes};m=${global_edges}"
  echo "--------------------------------------------------" | tee -a ./benchmark_results.txt
  echo "Running generator '${algo}' with options: ${options}" | tee -a ./benchmark_results.txt
  echo "--------------------------------------------------" | tee -a ./benchmark_results.txt
  
  mpirun -n ${NP} ./build/myapp -g "${options}" | tee -a ./benchmark_results.txt
  
  echo "" | tee -a ./benchmark_results.txt
  echo "Completed run for generator: ${algo}"
  echo
done
