source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d

ns-train splatfacto2d --vis viewer+tensorboard  \
            --data /mnt/kostas-graid/datasets/boshu/DTU/DTU/scan40/ \
            --pipeline.model.densify-grad-thresh 0.00012 \
            --pipeline.model.stop-screen-size-at 30000 \
            --pipeline.model.num-cluster 1 \
            --pipeline.model.voxel-size 0.004 \
            --pipeline.model.sdf-trunc 0.016 \
            --experiment-name scan40_test_grad_00012 \
            --pipeline.model.lambda_dist 1000 \
            --pipeline.model.lambda_normal 0.05 \
            --pipeline.model.depth-trunc 3.0 \
            --pipeline.model.background_color black \
            --viewer.quit-on-train-completion True \
            --pipeline.datamanager.camera_res_scale_factor 0.5 \
            --pipeline.model.continue_cull_post_densification False colmap

            # --pipeline.datamanager.start_img_num 4 \
            # --pipeline.datamanager.batch_select_num 1 \
            # --pipeline.datamanager.select_step 3000 \

ns-train splatfacto2d --vis viewer+tensorboard  \
            --data /mnt/kostas-graid/datasets/boshu/DTU/DTU/scan40/ \
            --pipeline.model.densify-grad-thresh 0.00015 \
            --pipeline.model.stop-screen-size-at 30000 \
             --pipeline.model.stop_split_at 24000 \
            --pipeline.model.num-cluster 1 \
            --pipeline.model.voxel-size 0.004 \
            --pipeline.model.sdf-trunc 0.016 \
            --experiment-name scan40_Fisher_s4 \
            --pipeline.model.lambda_dist 1000 \
            --pipeline.model.lambda_normal 0.05 \
            --pipeline.model.depth-trunc 3.0 \
            --pipeline.model.background_color black \
            --viewer.quit-on-train-completion True \
            --pipeline.datamanager.camera_res_scale_factor 0.5 \
            --pipeline.datamanager.start_img_num 4 \
            --pipeline.datamanager.batch_select_num 4 \
            --pipeline.datamanager.final_img_num 10 \
            --pipeline.datamanager.select_step 3000 \
            --pipeline.datamanager.select-method Fisher \
            --pipeline.model.continue_cull_post_densification False colmap

            

# ns-eval splatfacto2d \
#             --data /mnt/kostas-graid/datasets/boshu/360_v2/bicycle/ \
#             --load-checkpoint outputs/bicycle/splatfacto2d/2024-06-17_142230/nerfstudio_models/step-000029999.ckpt colmap