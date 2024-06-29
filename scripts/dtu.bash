source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d

ns-train splatfacto2d --vis viewer+tensorboard  \
            --data /mnt/kostas-graid/datasets/boshu/DTU/DTU/scan65/ \
            --pipeline.model.densify-size-thresh 0.05 \
            --pipeline.model.densify-grad-thresh 0.0002 \
            --pipeline.model.cull-alpha-thresh 0.05 \
            --pipeline.model.stop-screen-size-at 30000 \
            --pipeline.model.num-cluster 1 \
            --pipeline.model.voxel-size 0.004 \
            --pipeline.model.sdf-trunc 0.016 \
            --experiment-name scan65_all_distnormal_1000 \
            --pipeline.model.lambda_dist 1000 \
            --pipeline.model.depth-trunc 3.0 \
            --pipeline.model.background_color black \
            --viewer.quit-on-train-completion True \
            --pipeline.model.continue_cull_post_densification False colmap

            # --pipeline.datamanager.camera-res-scale-factor 2.0 \

# ns-eval splatfacto2d \
#             --data /mnt/kostas-graid/datasets/boshu/360_v2/bicycle/ \
#             --load-checkpoint outputs/bicycle/splatfacto2d/2024-06-17_142230/nerfstudio_models/step-000029999.ckpt colmap