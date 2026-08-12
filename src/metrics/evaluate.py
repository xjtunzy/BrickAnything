"""
2-21
gt mesh or pc
gen brick
"""

import numpy as np
import trimesh
from skimage import measure
import torch
import json
import os, argparse
#from render_pointcloud import render_pointcloud_xyz_to_png
#from metrics.PyTorchEMD.emd import earth_mover_distance

def calc_chamfer_loss(vertices_gt, vertices_recon):
    dist1 = torch.cdist(vertices_gt, vertices_recon, p=2).min(dim=1)[0]
    dist2 = torch.cdist(vertices_recon, vertices_gt, p=2).min(dim=1)[0]
    chamfer_loss = dist1.mean() + dist2.mean()
    return chamfer_loss

def calc_chamfer_torch(x, y):
    # x: [N,3], y:[M,3]
    d = torch.cdist(x, y, p=2)
    return d.min(dim=1)[0].mean() + d.min(dim=0)[0].mean()

def calc_emd_loss(vertices_gt, vertices_recon):
    """
    vertices_gt: [N, 3] Tensor
    vertices_recon: [N, 3] Tensor
    """
    gt_input = vertices_gt.unsqueeze(0).contiguous()
    recon_input = vertices_recon.unsqueeze(0).contiguous()
    dist = earth_mover_distance(gt_input, recon_input, transpose=False)
    emd_loss = dist[0]
    return emd_loss


def voxel2mesh(voxels, res):
    voxels = np.flip(voxels, axis=1)
    voxels = np.swapaxes(voxels, 1, 2)
    vol = voxels.astype(np.float32)
    if vol.max() < 0.5:
        print("Warning: Voxel volume is empty.")
        return trimesh.Trimesh()
    try:
        verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5)
    except ValueError:
        verts, faces, normals, _ = measure.marching_cubes(vol, level=0.1)
    verts = verts / res
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    return mesh

def seq2voxel(brick_path,res):
    assert os.path.exists(brick_path)
    with open(brick_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    seq = data['Seq']
    assert len(seq)%5==0
    occ = np.zeros((res,res,res))
    for i in range(len(seq)//5):
        idx = i*5
        x1 = seq[idx]
        y1 = seq[idx+1]
        z1 = seq[idx+2]
        x2 = seq[idx+3]
        y2 = seq[idx+4]
        #print(f"{x1} {x2+1}||{y1} {y2+1}||{z1}")
        occ[x1:x2+1,y1:y2+1,z1]=1
    #print(occ)

    return occ

def brick2voxel(bricks,res):
    occ = np.zeros((res,res,res))
    for b in bricks:
        x1 = b[2]
        y1 = b[3]
        z1 = b[4]
        x2 = b[2]+b[0]-1
        y2 = b[3]+b[1]-1
        occ[x1:x2+1,y1:y2+1,z1]=1
    return occ

def normalize_point_cloud(points):
    """ 将点云中心化并缩放到单位球/立方体内，确保 CD 计算公平 """
    centroid = torch.mean(points, dim=0)
    points -= centroid
    max_dist = torch.max(torch.sqrt(torch.sum(points**2, dim=1)))
    points /= max_dist
    return points


def normalize_with_gt(gt_points, gen_points):
    # 用 GT 的中心和尺度统一归一化（非常重要）
    centroid = gt_points.mean(dim=0, keepdim=True)
    gt_points = gt_points - centroid
    gen_points = gen_points - centroid

    scale = torch.norm(gt_points, dim=1).max()
    gt_points = gt_points / (scale + 1e-8)
    gen_points = gen_points / (scale + 1e-8)
    return gt_points, gen_points

def evaluate(brick_path,gt_path,res,name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt_mesh = trimesh.load(gt_path)
    if isinstance(gt_mesh, trimesh.Scene):
        if len(gt_mesh.geometry) == 0:
            raise ValueError("加载的场景中没有几何体！")
        gt_mesh = trimesh.util.concatenate([
            geom for geom in gt_mesh.geometry.values() 
            if isinstance(geom, trimesh.Trimesh)
        ])
    voxels = brick2voxel(brick_path, res)
    gen_mesh = voxel2mesh(voxels, res)
    if len(gen_mesh.vertices) == 0:
        print("Error: Generated mesh has no vertices.")
        return
    print('Finish mesh generation!')
    gt_points = gt_mesh.sample(10000)
    if not os.path.exists(f'visual/{name}/gt.png'):render_pointcloud_xyz_to_png(gt_points,f'visual/{name}/gt.png',points_per_object=8192)
    gt_points = torch.tensor(gt_points, device=device, dtype=torch.float32)
    gt_points_norm = normalize_point_cloud(gt_points)
    gen_points = gen_mesh.sample(10000)
    render_pointcloud_xyz_to_png(gen_points,f'visual/{name}/gen_{res}.png',points_per_object=8192)
    gen_points = torch.tensor(gen_points, device=device, dtype=torch.float32)
    gen_points_norm = normalize_point_cloud(gen_points)
    chamfer_loss = calc_chamfer_loss(gt_points_norm,gen_points_norm)
    print(f"chamfer_loss: {chamfer_loss.item():.6f}")
    try:
        emd_loss = calc_emd_loss(gt_points_norm, gen_points_norm)
        print(f"EMD Loss: {emd_loss.item():.6f}")
    except Exception as e:
        print(f"EMD calculation failed: {e}")

def evaluate_func(bricks,gt_p):
    res = 20
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    voxels = brick2voxel(bricks, res)
    gen_mesh = voxel2mesh(voxels, res)
    if len(gen_mesh.vertices) == 0:
        print("Error: Generated mesh has no vertices.")
        return torch.tensor(float('nan'), device=device)
    #print('Finish mesh generation!')
    gt_points = gt_p[:,:3]
    #gt_points = torch.tensor(gt_points, device=device, dtype=torch.float32)
    #print(f"gt_points shape: {gt_points.shape}")
    gt_points_norm = normalize_point_cloud(gt_points)
    gen_points = gen_mesh.sample(8192)
    gen_points = torch.tensor(gen_points, device=device, dtype=torch.float32)
    gen_points_norm = normalize_point_cloud(gen_points)
    chamfer_loss = calc_chamfer_loss(gt_points_norm,gen_points_norm)
    #print(f"chamfer_loss: {chamfer_loss.item():.6f}")
    # try:
    #     emd_loss = calc_emd_loss(gt_points_norm, gen_points_norm)
    #     print(f"EMD Loss: {emd_loss.item():.6f}")
    # except Exception as e:
    #     print(f"EMD calculation failed: {e}")
    return chamfer_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser("BrickAnything", add_help=False)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--res', default=None, type=int)
    args = parser.parse_args()
    brick_path = f'/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/src/mesh2brick_v2/brickmodel/{args.name}/cpp/{args.res}/seq.json'
    gt_path = f'/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/src/mesh2brick_v2/data/{args.name}.obj'
    evaluate(brick_path,gt_path,args.res,args.name)
