source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d

ns-train splatfacto2d --vis viewer+tensorboard  \
            --experiment-name depth_pearson \
            --data /mnt/kostas-graid/datasets/boshu/touch-gs-data/bunny_blender_data/ \
            --pipeline.model.densify-size-thresh 0.05 \
            --pipeline.model.densify_grad_thresh 0.0002 \
            --pipeline.model.cull_alpha_thresh 0.05 \
            --pipeline.model.depth_lambda 0.1 \
            --pipeline.model.depth_loss_type Pearson \
            --pipeline.model.stop_screen_size_at 30000 \
            --viewer.quit-on-train-completion True \
            --pipeline.model.continue_cull_post_densification False nerfstudio-data

# ns-eval splatfacto2d \
#             --data /mnt/kostas-graid/datasets/boshu/360_v2/bicycle/ \
#             --load-checkpoint outputs/bicycle/splatfacto2d/2024-06-17_142230/nerfstudio_models/step-000029999.ckpt colmap