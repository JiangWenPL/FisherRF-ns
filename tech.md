

## Latest command

```
ns-train splatfacto --logging.local-writer.max-log-size 0 --logging.steps-per-log 100 --pipeline.model.background-color white --pipeline.model.near-plane 2. --pipeline.model.far-plane 6. --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-average-appearance-embedding False --pipeline.model.distortion-loss-mult 0 --pipeline.model.disable-scene-contraction True  --vis tensorboard --data /mnt/kostas-graid/datasets/nerf_synthetic/lego/ --experiment-name debug-lego-W  --timestamp main  --relative-model-dir=nerfstudio_models  --max-num-iterations=30000  blender-data
```


### Workflow


### Debugging:

set `disable=True` inside `base_pipeline.py` to use ipdb within progress bar