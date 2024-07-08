#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --array=0-5
#SBATCH --partition=batch
#SBATCH --qos=normal
##SBATCH -w=kd-a40-0.grasp.maas
#SBATCH --time=12:00:00
#SBATCH --exclude=kd-2080ti-1.grasp.maas,mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,ee-a6000-0.grasp.maas,ee-3090-1.grasp.maas
##SBATCH --exclude=ee-a6000-0.grasp.maas
##SBATCH --exclude=node-a6000-0,node-v100-0
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./outputs/cluster/%x-%j.out

hostname
echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNT '/ Job ID' $SLURM_JOBID

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d
cd ~/FisherRF-ns

SCENES=("bicycle" "bonsai" "counter" "kitchen" "room" "stump")
SCENE=${SCENES[$SLURM_ARRAY_TASK_ID]}

SEED=$1
ACTIVE=$2

DATADIR=/mnt/kostas-graid/datasets/boshu/360_v2/${SCENE}

# If directory 'colmap' exists, then skip, else create directory
if [ -x "${DATADIR}/colmap" ]; then
    echo "Directory 'colmap' exists."
else
    mkdir -p ${DATADIR}/colmap
    echo "Directory 'colmap' created."

    ln -s ${DATADIR}/sparse ${DATADIR}/colmap/sparse
fi

ns-train splatfacto2d --vis viewer+tensorboard  \
            --data /mnt/kostas-graid/datasets/boshu/360_v2/${SCENE}/ \
            --machine.seed ${SEED} \
            --pipeline.model.densify_grad_thresh 0.00020 \
            --pipeline.model.stop-screen-size-at 30000 \
            --pipeline.model.stop_split_at 20000 \
            --pipeline.model.num-cluster 1 \
            --pipeline.model.voxel-size 0.004 \
            --pipeline.model.sdf-trunc 0.016 \
            --experiment-name ${SCENE}_${ACTIVE}_seed_${SEED} \
            --pipeline.model.lambda_normal 0.05 \
            --pipeline.model.depth-trunc 3.0 \
            --viewer.quit-on-train-completion True \
            --pipeline.datamanager.start_img_num 4 \
            --pipeline.datamanager.batch_select_num 4 \
            --pipeline.datamanager.final_img_num 20 \
            --pipeline.datamanager.select_step 3000 \
            --pipeline.datamanager.select-method ${ACTIVE} \
            --pipeline.model.continue_cull_post_densification False colmap
