source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d

ns-train splatfacto2d --vis viewer+tensorboard  \
            --data /mnt/kostas-graid/datasets/boshu/360_v2/garden/ \
            --pipeline.model.densify-size-thresh 0.05 \
            --pipeline.model.densify_grad_thresh 0.00015 \
            --pipeline.model.cull_alpha_thresh 0.05 \
            --pipeline.model.stop_screen_size_at 30000 \
            --pipeline.model.continue_cull_post_densification False colmap

# ns-eval splatfacto2d \
#             --data /mnt/kostas-graid/datasets/boshu/360_v2/bicycle/ \
#             --load-checkpoint outputs/bicycle/splatfacto2d/2024-06-17_142230/nerfstudio_models/step-000029999.ckpt colmap