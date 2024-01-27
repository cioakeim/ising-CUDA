#!/bin/bash 

#SBATCH --partition=gpu
#SBATCH --output=v2test.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

module load gcc/9.4.0 cuda/11.2.2-kkrwdua 

cd $HOME/ising-CUDA

make v2test 

$HOME/ising-CUDA/bin/v2test
