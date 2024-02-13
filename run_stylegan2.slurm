#!/bin/bash
#SBATCH --job-name=styleGAN2
#SBATCH --partition=oi1
##SBATCH --reservation=oi1
#SBATCH --account=oi1
#SBATCH --ntasks=1
#SBATCH --mail-user=jz79@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jz79/Welch/training-runs/stylegan2/run_debug
#SBATCH --mem=200G
#SBATCH --cpus-per-task=24
#SBATCH --threads-per-core=1
## srun --pty --partition=oi1 --reservation=oi1 --account=oi1 --gres=gpu:8 --time=00:30:00 --mem=64G --cpus-per-task=8 $SHELL
## local: export CC=/usr/bin/gcc-9
## local: export CXX=/usr/bin/g++-9

ml purge
ml GCC/11.2.0
ml Anaconda3/2022.05
#source ~/.bashrc
source activate qj3-dae3

export OMP_NUM_THREADS=12

python train.py --data /scratch/jz79/Welch/trainingset2 --label-dict /scratch/jz79/Welch/InceptionV3-labels.pkl --outdir /scratch/jz79/Welch/training-runs/stylegan2 --gpus 4 --batch 192 --workers 48 --tick 2 --snap 25
