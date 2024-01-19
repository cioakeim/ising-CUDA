#!/bin/bash 

#SBATCH --partition=gpu
#SBATCH --output=v1time.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

module load gcc/9.4.0 cuda/11.2.2-kkrwdua 

cd $HOME/ising-CUDA

make v1time 

$HOME/ising-CUDA/bin/v1time $HOME/plotData
