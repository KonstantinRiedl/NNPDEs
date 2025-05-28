#! /bin/bash

#SBATCH --cluster=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --partition=short
#SBATCH --time=12:00:00

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=konstantin.riedl@maths.ox.ac.uk

cd $DATA
cd NNPDEs

module load PyTorch/1.12.0-foss-2022a-CUDA-11.8.0

mkdir -p slurm_logs
mkdir -p results


# Run the Python script
python NNPDE_Analysis.py

