#!/bin/bash
#SBATCH --job-name=styleGAN2
#SBATCH --partition=oi1
##SBATCH --reservation=oi1
#SBATCH --account=oi1
#SBATCH --ntasks=1
#SBATCH --mail-user=jz79@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/jz79/Welch/training-runs/stylegan2/run_debug
#SBATCH --mem=200G
#SBATCH --cpus-per-task=30
#SBATCH --threads-per-core=1

ml purge
ml GCC/11.2.0
ml Anaconda3/2022.05
source activate qj3-dae3

export OMP_NUM_THREADS=12

python train.py --data /scratch-new/jz79/Welch/trainingset2 --label-dict /scratch-new/jz79/Welch/training_dict.pkl --outdir /scratch-new/jz79/Welch/training-runs/stylegan2 --gpus 4 --batch 224 --workers 56 --tick 1 --snap 50 --glr 0.001 --dlr 0.001
