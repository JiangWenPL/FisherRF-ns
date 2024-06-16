ns-train splatfacto2d --vis viewer+tensorboard  \
            --data /mnt/kostas-graid/datasets/boshu/360_v2/bicycle/ \
            --optimizers.means.optimizer.lr 0.0007 \
            --pipeline.model.densify-size-thresh 0.05 colmap