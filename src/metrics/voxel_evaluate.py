"""
3-29
gt mesh or pc
gen brick
"""
import numpy as np
import trimesh
from skimage import measure
import json
import math
import os, argparse
from scipy.spatial import cKDTree
from metrics.voxelization_v1 import voxelize_mesh, voxelize_pc_npy
from tqdm import tqdm

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



def voxel_iou(voxel_pred, voxel_gt):
    """
    计算两个 20x20x20 voxel 的 IoU

    Args:
        voxel_pred: np.ndarray, shape=(20, 20, 20)
        voxel_gt:   np.ndarray, shape=(20, 20, 20)

    Returns:
        float: IoU
    """
    assert voxel_pred.shape == (20, 20, 20)
    assert voxel_gt.shape == (20, 20, 20)

    pred = voxel_pred.astype(bool)
    gt = voxel_gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return intersection / union


def voxel_cd(voxel_pred, voxel_gt):
    """
    计算两个 20x20x20 voxel 的 Chamfer Distance

    Args:
        voxel_pred: np.ndarray, shape=(20, 20, 20)
        voxel_gt:   np.ndarray, shape=(20, 20, 20)

    Returns:
        float: CD
    """
    assert voxel_pred.shape == (20, 20, 20)
    assert voxel_gt.shape == (20, 20, 20)

    pts_pred = np.argwhere(voxel_pred > 0).astype(np.float32)
    pts_gt = np.argwhere(voxel_gt > 0).astype(np.float32)

    if len(pts_pred) == 0 and len(pts_gt) == 0:
        return 0.0
    if len(pts_pred) == 0 or len(pts_gt) == 0:
        return float("inf")

    tree_pred = cKDTree(pts_pred)
    tree_gt = cKDTree(pts_gt)

    dist_pred_to_gt, _ = tree_gt.query(pts_pred, k=1)
    dist_gt_to_pred, _ = tree_pred.query(pts_gt, k=1)

    cd = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()
    return cd


def save_voxel_comparison_png(
    pred,
    gt,
    out_path,
    title="",
    dpi=140,
):
    """
    将 pred（生成 brick 体素）与 gt（mesh/pc 体素）画在同一张 3D 图里并保存为 PNG。

    颜色：绿=TP（两者同占），橙=FP（仅 pred），蓝=FN（仅 gt）。
    需要 matplotlib（无 GUI 时用 Agg 后端写文件）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    assert pred.shape == gt.shape

    tp = np.logical_and(pred, gt)
    fp = np.logical_and(pred, ~gt)
    fn = np.logical_and(~pred, gt)
    filled = np.logical_or(pred, gt)

    fig = plt.figure(figsize=(6.5, 5.8))
    ax = fig.add_subplot(111, projection="3d")

    if filled.any():
        colors = np.zeros(pred.shape + (4,), dtype=np.float64)
        colors[..., 3] = 0.0
        colors[tp] = (0.15, 0.82, 0.28, 0.88)
        colors[fp] = (0.95, 0.42, 0.12, 0.88)
        colors[fn] = (0.25, 0.45, 0.95, 0.88)
        ax.voxels(filled, facecolors=colors, edgecolor="k", linewidth=0.12)
    else:
        ax.text2D(0.5, 0.5, "empty pred & gt", transform=ax.transAxes, ha="center")

    n = pred.shape[0]
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_zlim(0, n)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title or "green=TP  orange=FP(pred)  blue=FN(gt)")
    ax.view_init(elev=22, azim=42)

    out_path = os.fspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)


def evaluate(brick_path, gt_path, mode, return_volumes=False):
    brick_occ = seq2voxel(brick_path, 20)
    if mode == "mesh":
        voxels = voxelize_mesh(gt_path, (20, 20, 20))
    elif mode == "pc":
        voxels = voxelize_pc_npy(gt_path, (20, 20, 20))
    else:
        raise ValueError(f"mode must be 'mesh' or 'pc', got {mode!r}")
    iou = voxel_iou(brick_occ, voxels)
    cd = voxel_cd(brick_occ, voxels)
    if return_volumes:
        return iou, cd, brick_occ, voxels
    return iou, cd

def evaluate_func(bricks,gt_path, mode, return_volumes=True):
    res = 20
    brick_occ = brick2voxel(bricks, res)
    if mode == "mesh":
        voxels = voxelize_mesh(gt_path, (20, 20, 20))
    elif mode == "pc":
        voxels = voxelize_pc_npy(gt_path, (20, 20, 20))
    else:
        raise ValueError(f"mode must be 'mesh' or 'pc', got {mode!r}")
    #render_cubes2png(bricks, gt_path, colorm=[24,107,239], vs_size=0.5, sample_count=256, out_width=800, out_height=600, lookat_1=3, lookat_2=3, lookat_3=3, trim_img=0)
    iou = voxel_iou(brick_occ, voxels)
    cd = voxel_cd(brick_occ, voxels)
    if return_volumes:
        return iou, cd, brick_occ, voxels
    return iou, cd , voxels


def write_voxel_metrics_to_gen_json(brick_json_path, iou, cd):
    """把体素 IoU / CD 写进 *_gen.json（与 Seq 等同文件），cd 为 inf 时存 null。"""
    with open(brick_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["iou"] = float(iou)
    if isinstance(cd, float) and math.isinf(cd):
        data["cd"] = None
    else:
        data["cd"] = float(cd)
    with open(brick_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxel IoU/CD vs mesh or pc npy.")
    parser.add_argument("--gt_path", type=str, default="/mnt/nas/yanfeng/results/BrickAnything/inference_vail/03-30-20-16-47/vail_pc")
    parser.add_argument("--brick_path", type=str, default="/mnt/nas/yanfeng/results/BrickAnything/inference_vail/03-30-20-16-47")
    parser.add_argument("--viz_dir", type=str, default="/mnt/nas/yanfeng/results/BrickAnything/inference_vail/03-30-20-16-47", help="若设置，则每个样本保存 pred vs gt 体素对比 PNG 到此目录")
    args, _unknown = parser.parse_known_args()

    gt_path = args.gt_path
    brick_path = args.brick_path
    viz_dir = args.viz_dir

    for file in tqdm(os.listdir(gt_path)):
        # vail 点云文件名为 {uid}_pc.npy，与 {uid}_gen.json 对齐时需去掉后缀 _pc
        stem = os.path.splitext(file)[0]
        uid = stem.removesuffix("_pc")
        brick_json = os.path.join(brick_path, f"{uid}_gen.json")
        print(f"brick_json: {brick_json}")
        if not os.path.exists(brick_json):
            continue

        gt_file = os.path.join(gt_path, file)
        if viz_dir:
            iou, cd, brick_occ, gt_vox = evaluate(brick_json, gt_file, "pc", return_volumes=True)
            viz_path = os.path.join(viz_dir, f"{uid}_voxel_compare.png")
            save_voxel_comparison_png(
                brick_occ,
                gt_vox,
                viz_path,
                title=f"{uid}  IoU={iou:.4f}  CD={cd}",
            )
        else:
            iou, cd = evaluate(brick_json, gt_file, "pc")
        write_voxel_metrics_to_gen_json(brick_json, iou, cd)
        print(f"{uid} iou: {iou}, cd: {cd} (written to json)")

