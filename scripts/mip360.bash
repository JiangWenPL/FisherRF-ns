ns-train splatfacto2d --vis viewer+tensorboard  \
            --data /mnt/kostas-graid/datasets/boshu/360_v2/bicycle/ \
            --pipeline.model.densify-size-thresh 0.05 \
            --pipeline.model.densify_grad_thresh 0.00015 \
            --pipeline.model.cull_alpha_thresh 0.05 \
            --pipeline.model.stop_screen_size_at 30000 \
            --pipeline.model.continue_cull_post_densification False colmap