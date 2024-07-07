source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d

SCENES=("scan24" "scan37" "scan55" "scan63" "scan65" "scan69")

for SCENE in "${SCENES[@]}"
do
    SCENE_ID=${SCENE:4}
    echo "SCENE ID is ${SCENE_ID}"
    python scripts/dtu_eval.py --exp_dir outputs/${SCENE}_test_grad_00012/splatfacto2d/ \
                                --scan_id ${SCENE_ID} \
                                --DTU /mnt/kostas-graid/datasets/boshu/DTU/SampleSet/MVSData \
                                --dtu /mnt/kostas-graid/datasets/boshu/DTU/DTU/
done
    

