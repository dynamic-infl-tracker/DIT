#!/bin/zsh
#SBATCH --job-name=mnist_detection           # Job name
#SBATCH --output=%x_%j.log                   # Output file
#SBATCH --error=%x_%j_err.log                # Error file
#SBATCH --ntasks=1                           # Total number of tasks
#SBATCH --cpus-per-task=6                    # Number of CPUs required per task
#SBATCH --gres=gpu:3                         # Number of GPUs required
#SBATCH --mem=64G                            # Memory
#SBATCH --time=128:00:00                      # Maximum job runtime
#SBATCH --partition=debug                    # Partition to use

# Set initial seed and final seed
INITIAL_SEED=0
FINAL_SEED=15

# Default settings
HOME_DIR="/home/anonymous/codes"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --runpod)
      HOME_DIR="/workspace"
      shift # Move to the next argument
      ;;
    --local_linux)
      HOME_DIR="/home/anonymous/codes"
      shift # Move to the next argument
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set other paths
PYTHON_ENV="$HOME_DIR/dit/.venv/bin/python"
WORK_DIR="$HOME_DIR/dit/experiment"
TRAIN_SCRIPT="$WORK_DIR/train.py"
CLEANSING_SCRIPT="$WORK_DIR/data_cleansing.py"
INFL_SCRIPT="$WORK_DIR/infl.py"

# cleansing
# PYTHON_COMMAND='
#     for model in logreg dnn cnn; do
#         for flip in 20 30 10; do
#             for check in 5 10 15 20 25 30 35 40 45 50; do
#                 # Train the model
#                 $PYTHON_ENV "$TRAIN_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir cleansing/mnist_"$model"_flip_"$flip" --flip "$flip";
                
#                 # Run influence computation
#                 # for type in true sgd icml segment_true dit; do
#                 for type in true segment_true; do
#                     $PYTHON_ENV "$INFL_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir cleansing/mnist_"$model"_flip_"$flip" --flip "$flip" --type "$type";
#                     $PYTHON_ENV "$CLEANSING_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir cleansing/mnist_"$model"_flip_"$flip" --flip "$flip" --type "$type" --check "$check";
#                 done

#                 # for type in dit_first dit_middle dit_last true_first true_middle true_last; do
#                 for type in true_first true_middle true_last; do
#                     $PYTHON_ENV "$CLEANSING_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir cleansing/mnist_"$model"_flip_"$flip" --flip "$flip" --type "$type" --check "$check";
#                 done

#                 # Data cleansing
#             done
#         done
#     done
# '

# detection
PYTHON_COMMAND='
    for model in logreg dnn cnn; do
        for flip in 4 8 12 16; do
            # Training
            $PYTHON_ENV "$TRAIN_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir detection/mnist_"$model"_flip_"$flip" --flip "$flip";
            # Influence
            for type in true segment_true icml sgd dit; do
                $PYTHON_ENV "$INFL_SCRIPT" --target mnist --model "$model" --seed "$seed" --gpu 0 --save_dir detection/mnist_"$model"_flip_"$flip" --flip "$flip" --type "$type";
            done
        done
    done
'

# Print current settings (for debugging)
echo "Current settings:"
echo "HOME_DIR: $HOME_DIR"
echo "PYTHON_ENV: $PYTHON_ENV"
echo "WORK_DIR: $WORK_DIR"
echo "TRAIN_SCRIPT: $TRAIN_SCRIPT"
echo "INFL_SCRIPT: $INFL_SCRIPT"
echo "CLEANSING_SCRIPT: $CLEANSING_SCRIPT"

# Display debug information
echo "========== Debug Info =========="
echo "Current working directory: $(pwd)"
echo "Python interpreter: $PYTHON_ENV"
echo "Initial seed: $INITIAL_SEED"
echo "Final seed: $FINAL_SEED"
echo "==============================="

# Change to the working directory
cd "$WORK_DIR" || exit 1

echo "========== Job Info =========="
echo "Job started at: $(date)"
echo "Job ID: $$"
echo "Node list: $SLURM_JOB_NODELIST"
echo "GPUs: All available"
echo "==============================="

# Get the number of available GPUs
n_gpus=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $n_gpus"

# Create a temporary file to store the next seed value
SEED_FILE="/tmp/next_seed_$$"
echo $INITIAL_SEED > $SEED_FILE

# Define a function to safely get the next seed value
get_next_seed() {
    local next_seed
    local lock_file="/tmp/seed_lock_$$"

    # Attempt to get the lock
    while ! mkdir "$lock_file" 2>/dev/null; do
        sleep 0.1
    done

    # Read and update the seed value
    next_seed=$(<$SEED_FILE)
    echo $((next_seed + 1)) > $SEED_FILE

    # Release the lock
    rmdir "$lock_file"

    echo $next_seed
}

run_experiment() {
    local seed="$1"
    local gpu="$2"

    if [[ -z "$seed" || -z "$gpu" ]]; then
        echo "Error: seed or gpu parameter not set"
        return 1
    fi

    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp][gpu_$gpu] Running with seed=$seed"

    if [ ! -f "$PYTHON_ENV" ]; then
        echo "Python interpreter not found at $PYTHON_ENV"
        return 1
    fi

    if [ ! -f "$TRAIN_SCRIPT" ]; then
        echo "Train script not found at $TRAIN_SCRIPT"
        return 1
    fi

    # Set environment variables to limit GPU memory usage
    export CUDA_VISIBLE_DEVICES=$gpu
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

    # Execute the Python command using the variable
    eval $PYTHON_COMMAND

    local exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "[$timestamp][gpu_$gpu] Error occurred with seed=$seed. Exit status: $exit_status"
        return 1
    fi

    echo "[$timestamp][gpu_$gpu] Task for seed $seed completed"
}

# Maximum number of concurrent processes per GPU
max_processes_per_gpu=2

# Create a function to handle tasks for each GPU
process_gpu_tasks() {
    local gpu=$1
    local pids=()

    while true; do
        # Check the current number of running processes
        while (( ${#pids} >= max_processes_per_gpu )); do
            for pid in $pids; do
                if ! kill -0 $pid 2>/dev/null; then
                    pids=("${(@)pids:#$pid}")
                fi
            done
            sleep 1
        done

        # Get the next seed
        local seed=$(get_next_seed)

        # Check if there are still unprocessed seeds
        if (( seed > FINAL_SEED )); then
            break
        fi

        run_experiment "$seed" "$gpu" &
        pids+=($!)
    done

    # Wait for all processes on this GPU to complete
    for pid in $pids; do
        wait $pid
    done
}

total_start_time=$(date +%s)

# Start a background process for each GPU to handle tasks
for ((gpu=0; gpu<n_gpus; gpu++)); do
    process_gpu_tasks $gpu &
done

# Wait for all GPU tasks to finish
wait

# Clean up temporary files
rm -f $SEED_FILE /tmp/seed_lock_$$

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "[$timestamp][main] All iterations completed."
echo "[$timestamp][main] Total execution time: $total_duration seconds"
