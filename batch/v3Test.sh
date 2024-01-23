#!/bin/bash 

#SBATCH --partition=gpu
#SBATCH --output=v3test.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

module load gcc/9.4.0 cuda/11.2.2-kkrwdua 

cd $HOME/ising-CUDA

make v3test 

$HOME/ising-CUDA/bin/v3test
