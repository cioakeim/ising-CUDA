#!/bin/bash 

#SBATCH --partition=gpu
#SBATCH --output=cudaTest.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

module load gcc/9.4.0 cuda/11.2.2-kkrwdua 

cd $HOME/ising-CUDA

make v0Test

./bin/v0Test
