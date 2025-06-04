#!/bin/bash

# First script
echo "Starting DT Standard simulation..."
./experiments/experiment_cmds/dt_simulations/full/dt_standard.sh

# Check if the first script executed successfully
if [ $? -eq 0 ]; then
    echo "DT Standard simulation completed successfully."
else
    echo "DT Standard simulation failed with exit code $?"
    exit 1
fi

# Second script
echo "Starting DT Expanded simulation..."
./experiments/experiment_cmds/dt_simulations/full/dt_expanded.sh

# Check if the second script executed successfully
if [ $? -eq 0 ]; then
    echo "DT Expanded simulation completed successfully."
else
    echo "DT Expanded simulation failed with exit code $?"
    exit 1
fi

echo "All simulations completed successfully!"

# Second script
echo "Starting DT Fitting Time simulation..."
./experiments/experiment_cmds/dt_simulations/full/dt_run_times.sh

# Check if the second script executed successfully
if [ $? -eq 0 ]; then
    echo "DT Fitting Time simulation completed successfully."
else
    echo "DT Fitting Time simulation failed with exit code $?"
    exit 1
fi

echo "All simulations completed successfully!"