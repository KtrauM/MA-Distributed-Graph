#!/bin/bash
#SBATCH --job-name=grarrph                               # Job name
#SBATCH --output=../slurm_logs/output/output_%j.log      # Standard output log
#SBATCH --error=../slurm_logs/error/error_%j.log         # Error log
#SBATCH --time=12:00:00                                  # Time limit (hh:mm:ss)
#SBATCH --partition=hoare                                # Partition
#SBATCH --nodes=1                                        # Number of nodes
#SBATCH --ntasks=1                                       # Total number of MPI tasks
#SBATCH --cpus-per-task=1                                # CPUs per MPI task
#SBATCH --mem=8G                                         # Memory per node
#SBATCH --mail-type=END,FAIL                             # Email notifications
#SBATCH --mail-user=ucjly@student.kit.edu                # Email address

spack load openmpi arch=linux-rocky9-x86_64
source /nfs/home/mkurnaz/benchmark-env/bin/activate
python3 benchmark.py