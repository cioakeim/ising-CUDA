#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --output=cudaTest.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

module load gcc/9.4.0 cuda/11.2.2-kkrwdua 

nvcc -o $HOME/ising-CUDA/test  $HOME/ising-CUDA/cudaTest.cu

$HOME/ising-CUDA/test
