source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d

ns-train splatfacto2d --vis viewer+tensorboard  \
            --experiment-name depth_pearson \
            --data /mnt/kostas-graid/datasets/boshu/touch-gs-data/bunny_blender_data/ \
            --pipeline.model.densify-grad-thresh 0.00015 \
            --pipeline.model.stop-screen-size-at 30000 \
            --experiment_name bunny_blender_mesh_normal \
            --pipeline.model.num-cluster 1 \
            --pipeline.model.voxel-size 0.004 \
            --pipeline.model.sdf-trunc 0.016 \
            --pipeline.model.depth-trunc 3.0 \
            --pipeline.model.background_color black \
            --viewer.quit-on-train-completion True \
            --pipeline.model.continue_cull_post_densification False nerfstudio-data

            # --pipeline.model.lambda_dist 1000 \

# ns-eval splatfacto2d \
#             --data /mnt/kostas-graid/datasets/boshu/360_v2/bicycle/ \
#             --load-checkpoint outputs/bicycle/splatfacto2d/2024-06-17_142230/nerfstudio_models/step-000029999.ckpt colmap