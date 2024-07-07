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
DTU_SCENES=("scan24" "scan37" "scan40" "scan55" "scan63" "scan65" )
# "scan24" "scan37"  "scan40" "scan55" "scan63" "scan65" "scan69" "scan83" "scan97" "scan105"

SCENE=${DTU_SCENES[$SLURM_ARRAY_TASK_ID]}
SCENE_ID=${SCENE:4}
echo "Scene ID is ${SCENE_ID}"

EXP_NAME=$1
DATADIR=/mnt/kostas-graid/datasets/boshu/DTU/DTU/${SCENE}

# If directory 'colmap' exists, then skip, else create directory
if [ -x "${DATADIR}/colmap" ]; then
    echo "Directory 'colmap' exists."
else
    mkdir -p ${DATADIR}/colmap
    echo "Directory 'colmap' created."

    ln -s ${DATADIR}/sparse ${DATADIR}/colmap/sparse
fi

cd ~/FisherRF-ns

ns-train splatfacto2d --vis viewer+tensorboard  \
            --data /mnt/kostas-graid/datasets/boshu/DTU/DTU/${SCENE}/ \
            --pipeline.model.densify-grad-thresh 0.00015 \
            --pipeline.model.stop-screen-size-at 30000 \
             --pipeline.model.stop_split_at 24000 \
            --pipeline.model.num-cluster 1 \
            --pipeline.model.voxel-size 0.004 \
            --pipeline.model.sdf-trunc 0.016 \
            --experiment-name  ${SCENE}_${EXP_NAME} \
            --pipeline.model.lambda_dist 1000 \
            --pipeline.model.lambda_normal 0.05 \
            --pipeline.model.depth-trunc 3.0 \
            --pipeline.model.background_color black \
            --viewer.quit-on-train-completion True \
            --pipeline.datamanager.camera_res_scale_factor 0.5 \
            --pipeline.datamanager.start_img_num 4 \
            --pipeline.datamanager.batch_select_num 1 \
            --pipeline.datamanager.final_img_num 10 \
            --pipeline.datamanager.select_step 3000 \
            --pipeline.datamanager.select-method Fisher \
            --pipeline.model.continue_cull_post_densification False colmap


python scripts/dtu_eval.py --exp_dir outputs/${SCENE}_${EXP_NAME}/splatfacto2d/ \
        --scan_id ${SCENE_ID} \
        --DTU /mnt/kostas-graid/datasets/boshu/DTU/SampleSet/MVSData \
        --dtu /mnt/kostas-graid/datasets/boshu/DTU/DTU/
