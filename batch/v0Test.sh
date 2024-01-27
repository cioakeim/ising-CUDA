#!/bin/bash 

#SBATCH --partition=gpu
#SBATCH --output=v0test.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00

module load gcc/9.4.0 cuda/11.2.2-kkrwdua 

cd $HOME/ising-CUDA

make v0test

$HOME/ising-CUDA/bin/v0test $1 $2
