#!/bin/bash

# Parameters
MAX_JOBS=100  # maximum concurrent jobs
SLEEP_TIME=30  # check every 30 seconds

# List all scripts, sorted properly
cd experiment_data/graph_gen_25_04_29/jobfiles/
scripts=($(ls | sort -V))

# Index for scripts
i=0
total=${#scripts[@]}

# Function to count running jobs
count_jobs() {
    squeue -h -u $USER --states=R,PD | wc -l
}

# Main loop
while [ $i -lt $total ]; do
    script="${scripts[$i]}"
    
    # Check if the script corresponds to an already generated graph
    if [[ "$script" == graph_gen* ]]; then
        # Remove the 'graph_gen-' prefix
        graph_name="${script#graph_gen-}"
        # Replace the last occurrence of '-cores' with '_cores'
        graph_name="${graph_name%-cores*}_cores${graph_name##*-cores}"
        
        # Check if the corresponding graph file/directory exists
        if [ -e "/home/hk-project-toolbox/ucjly/MA-Distributed-Graph/graphs/$graph_name" ]; then
            echo "Skipping $script because /home/hk-project-toolbox/ucjly/MA-Distributed-Graph/graphs/$graph_name already exists."
            ((i++))
            continue
        fi
    fi

    # Check number of current jobs
    running_jobs=$(count_jobs)

    if [ "$running_jobs" -lt "$MAX_JOBS" ]; then
        echo "Submitting $script"
        sbatch "$script"
        ((i++))
    else
        echo "Maximum jobs running ($running_jobs). Waiting..."
        sleep $SLEEP_TIME
    fi
done

echo "All jobs submitted. Waiting for remaining jobs to finish..."

# Optionally: wait until all jobs finish
# while [ "$(count_jobs)" -gt 0 ]; do
#     sleep $SLEEP_TIME
# done

echo "All jobs completed."
