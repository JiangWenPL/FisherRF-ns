source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate gaussian2d

SCENES=("24" "65" "69" "83" "97" "105")

for SCENE in "${SCENES[@]}"
do
    python scripts/dtu_eval.py --exp_dir outputs/scan${SCENE}/splatfacto2d/ --scan_id ${SCENE} \
                                --DTU /mnt/kostas-graid/datasets/boshu/DTU/SampleSet/MVSData \
                                --dtu /mnt/kostas-graid/datasets/boshu/DTU/DTU/
done
    

