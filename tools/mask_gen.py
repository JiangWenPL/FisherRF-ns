import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import os.path as osp
import glob

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

IMG_DIR = "/mnt/kostas-graid/datasets/boshu/touch-gs-data/bunny_blender_data/imgs"
POINTS_DIR = "/mnt/kostas-graid/datasets/boshu/touch-gs-data/bunny_blender_data/points"

MASKS_DIR = IMG_DIR.replace("imgs", "masks")
os.makedirs(MASKS_DIR, exist_ok=True)

img_files = glob.glob(osp.join(IMG_DIR, "*.png"))
img_files.sort()

points_files = glob.glob(osp.join(POINTS_DIR, "*.npz"))
points_files.sort()

for img_file, point_file in zip(img_files, points_files):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = np.load(point_file)
    coords = points["points"]
    labels = np.ones((coords.shape[0], ))

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
    )
    
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_mask = best_mask.astype(np.uint8) * 255

    mask_filename = osp.join(MASKS_DIR, osp.basename(img_file))
    cv2.imwrite(mask_filename, best_mask)

    
    
    