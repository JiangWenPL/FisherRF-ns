#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
##SBATCH --array=0-8
#SBATCH --partition=batch
#SBATCH --qos=normal
##SBATCH -w=kd-a40-0.grasp.maas
#SBATCH --time=12:00:00
#SBATCH --exclude=kd-2080ti-1.grasp.maas,mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,ee-a6000-0.grasp.maas
##SBATCH --exclude=ee-a6000-0.grasp.maas
##SBATCH --exclude=node-a6000-0,node-v100-0
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./outputs/cluster/%x-%j.out

hostname
echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNT '/ Job ID' $SLURM_JOBID

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d

GRAD_THRESH=$1
EXP_NAME=$2

ns-train splatfacto2d --vis viewer+tensorboard  \
            --experiment-name ${EXP_NAME} \
            --data /home/leiboshu/touch-gs-data/bunny_blender_data/ \
            --pipeline.model.densify-size-thresh 0.05 \
            --pipeline.model.densify_grad_thresh ${GRAD_THRESH} \
            --pipeline.model.cull_alpha_thresh 0.05 \
            --pipeline.model.stop_screen_size_at 30000 \
            --pipeline.model.continue_cull_post_densification False blender-data