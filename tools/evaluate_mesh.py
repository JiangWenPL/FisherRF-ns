import numpy as np
import open3d as o3d
import trimesh
import torch
from skimage.morphology import binary_dilation, disk
import sklearn.neighbors as skln

from tqdm import tqdm
import cv2
import os
import glob
import json
import torch.nn.functional as F

@ torch.no_grad()
def cull_scan(mesh_path, mask_dir, extrinsics, intrinsics):
    # hard-coded image shape
    W, H = 900, 900

    # load mask
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        masks.append(mask)
    n_images = len(masks)

    # load mesh
    mesh = trimesh.load(mesh_path)

    # load transformation matrix
    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    sampled_masks = []
    for i in tqdm(range(n_images), desc="Culling mesh given masks"):
        pose = torch.from_numpy(extrinsics[i]).cuda().float()
        intrinsic = torch.from_numpy(intrinsics[i]).cuda().float()
        # pose = torch.inverse(pose)

        w2c = pose[:3, :]

        # transform and project
        cam_points = intrinsic @ w2c @ vertices
        pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
        pix_coords = pix_coords.permute(1, 0)
        pix_coords[..., 0] /= W - 1
        pix_coords[..., 1] /= H - 1
        pix_coords = (pix_coords - 0.5) * 2
        valid = ((pix_coords > -1.) & (pix_coords < 1.)).all(dim=-1).float()

        # dialate mask similar to unisurf
        maski = masks[i].astype(np.float32) / 255.
        maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

        sampled_mask = \
            F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)
        sampled_mask = sampled_mask.squeeze()

        sampled_mask = sampled_mask + (1. - valid)
        sampled_masks.append(sampled_mask)

    sampled_masks = torch.stack(sampled_masks, -1)
    # filter

    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)

    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)

    mesh.export("cull.ply")
    del mesh

if __name__ == "__main__":
    mesh_path = "NSModel/fuse_post.ply"
    result_mesh_file = "result_mesh.ply"

    # load json file
    with open("NSModel/cameras.json", "r") as f:
        data = json.load(f)

    extrinsics = []
    intrinsics = []

    for cam_idx, cam_data in enumerate(data):
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.array(cam_data["rotation"])
        extrinsic[:3, 3] = np.array(cam_data["position"])
        extrinsics.append(extrinsic)

        intrinsic = np.eye(3)
        intrinsic[0, 0] = cam_data["fx"]
        intrinsic[1, 1] = cam_data["fy"]
        intrinsic[0, 2] = cam_data["width"] / 2
        intrinsic[1, 2] = cam_data["height"] / 2
        intrinsics.append(intrinsic)

    # cull_scan(mesh_path, "masks", extrinsics, intrinsics)

    # for bunny only
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    # Load a ply point cloud
    mesh = o3d.io.read_triangle_mesh("cull.ply")
    # mesh = mesh.compute_vertex_normals()
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_vertices -= np.mean(mesh_vertices, axis=0, keepdims=True)
    mesh_vertices = mesh_vertices @ rot
    mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    pcd = mesh.sample_points_uniformly(number_of_points=20_000)
    pcd_points = np.asarray(pcd.points)

    stl = o3d.io.read_triangle_mesh("bunny_rightcoord.STL")
    stl_vertex = np.asarray(stl.vertices)
    stl_vertex = stl_vertex / 1000
    stl_vertex -= np.mean(stl_vertex, axis=0, keepdims=True)
    stl.vertices = o3d.utility.Vector3dVector(stl_vertex)
    stl_pcd = stl.sample_points_uniformly(number_of_points=20_000)
    stl_points = np.asarray(stl_pcd.points)

    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=0.001, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(stl_points)

    dist_d2s, idx_d2s = nn_engine.kneighbors(pcd_points, n_neighbors=1, return_distance=True)
    max_dist = 1
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    print(mean_d2s)

    nn_engine.fit(pcd_points)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_points, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    print(mean_s2d)

    print("CD: " + str(mean_d2s + mean_s2d))