#!/bin/bash
#SBATCH --partition=gpu                 # Specify that you want to use a GPU partition
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks (1 for a single process)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (you might not need more than 1 for this)
#SBATCH --gres=gpu:1                    # Request 1 GPU (can be adjusted based on availability)
#SBATCH --time=10:10:00                 # Maximum runtime (change as needed; 10 minutes is typically reasonable for a simple job)
#SBATCH --job-name=pinns_temp           # Job name
#SBATCH --output=AC-dist-new.log        # Output log file

# Load necessary modules
module load anaconda/3
module load cuda/12.5  

# Activate your Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_tensordiff

# Print start time
echo "Job started at: $(date)"


# Run your Python script
echo "Running Python script..."
python "/home/users/aamit/Project/TensorDiffEq/examples/AC-dist.py"

# Print end time
echo "Job finished at: $(date)"
