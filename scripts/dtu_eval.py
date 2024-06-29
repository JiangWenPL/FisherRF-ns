import os
import os.path as osp
import argparse

arg = argparse.ArgumentParser()
arg.add_argument("--exp_dir", type=str, default="models")
arg.add_argument("--scan_id", type=int, default=0)
arg.add_argument('--dtu', required=True, type=str)
arg.add_argument('--DTU', required=True, type=str)
opt = arg.parse_args()

# get all the experiments under the dir
experiments = os.listdir(opt.exp_dir)
experiments.sort()
latest_exp = experiments[-1]
run_dir = osp.join(opt.exp_dir, latest_exp)
print(f"run dir: {run_dir}")

# export the Mesh Here
config_file = osp.join(run_dir, "config.yml")
print(f"ns-export gaussian-splat --load-config {config_file} --output-dir {run_dir}")
os.system(f"ns-export gaussian-splat --load-config {config_file} --output-dir {run_dir}")

# run the evaluation code for DTU
script_dir = "/home/leiboshu/2d-gaussian-splatting/scripts"
string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {run_dir}/fuse_post.ply " + \
            f"--scan_id {opt.scan_id} --output_dir {run_dir} " + \
            f"--mask_dir {opt.dtu} " + \
            f"--DTU {opt.DTU}"
print(string)
os.system(string)
