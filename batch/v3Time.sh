#!/bin/bash 

#SBATCH --partition=gpu
#SBATCH --output=v3time.stdout 
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

module load gcc/9.4.0 cuda/11.2.2-kkrwdua 

cd $HOME/ising-CUDA

make clean
make v3time 

$HOME/ising-CUDA/bin/v3time $HOME/plotData "$1"
