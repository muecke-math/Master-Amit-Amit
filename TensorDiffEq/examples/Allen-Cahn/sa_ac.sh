#!/bin/bash
#SBATCH --partition=computedfg                 # Specify that you want to use a GPU partition
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks=1                             # Number of tasks (1 for a single process)
#SBATCH --cpus-per-task=40                     # Number of CPU cores per task (you might not need more than 1 for this)
##SBATCH --gres=gpu:1                          # Request 1 GPU (can be adjusted based on availability)
#SBATCH --time=00:00:00                        # Maximum runtime (change as needed; 10 minutes is typically reasonable for a simple job)
#SBATCH --job-name=sa_ac                       # Job name
#SBATCH --output=/home/users/aamit/Project/TensorDiffEq/examples/Allen-Cahn/ac_baseline.log        # Output log file
#SBATCH --nodelist=dds1                        # Specify the node you want to use

# Load necessary modules
module load anaconda/3
# module load cuda/12.6.1  

# Activate your Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate env_tensordiff

# Print start time
echo "Job started at: $(date)"

export CUDA_HOME=/home/users/aamit/.conda/envs/env_tensordiff
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
# export TF_CUDA_VERSION=11.8
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/nvm

# ls $CUDA_HOME/bin/nvcc
# ls $CUDA_HOME/lib64/libcudart.so
# which nvcc
# nvcc --version
# nvidia-smi
# python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Run your Python script
echo "Running Python script..."
python "/home/users/aamit/Project/TensorDiffEq/examples/Allen-Cahn/AC-SA-new.py"
# python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

# Print end time
echo "Job finished at: $(date)"
