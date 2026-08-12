import trimesh
import numpy as np
import os
from PIL import Image, ImageDraw

from trimesh import grouping, transformations as tr
from trimesh.voxel import base as voxel_base
from trimesh.voxel import encoding as voxel_enc


def _load_trimesh_for_voxelize(filename):
    """
    GLB/glTF 里除三角网格外还可能有 Path3D（曲线）等；force='mesh' 有时会错误地得到 Path3D，
    而 Path3D 没有 voxelized。这里按 Scene 加载，只合并 Trimesh。
    """
    loaded = trimesh.load(filename, force="scene")
    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        dumped = loaded.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh) and len(dumped.faces) > 0:
            mesh = dumped
        else:
            meshes = [
                g
                for g in loaded.geometry.values()
                if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0
            ]
            if not meshes:
                raise ValueError(
                    f"No triangular mesh in {filename!r} "
                    f"(got {type(dumped).__name__} from dump; geometry may be paths/points only)."
                )
            mesh = (
                trimesh.util.concatenate(meshes)
                if len(meshes) > 1
                else meshes[0]
            )
    else:
        raise TypeError(
            f"Cannot voxelize {filename!r}: unsupported type {type(loaded).__name__}"
        )
    if len(mesh.faces) == 0:
        raise ValueError(f"Empty mesh after load: {filename!r}")
    return mesh


def load_pc_npy(path, mmap_mode=None):
    """
    读取 .npy 点云为 (N, 3) float64。
    支持形状 (N, 3)、(N, K)（取前三列 xyz）、或 (3, N)。
    """
    arr = np.load(os.fspath(path), mmap_mode=mmap_mode)
    arr = np.asanyarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"point cloud npy must be 2-D, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        pts = arr.T
    elif arr.shape[1] >= 3:
        pts = arr[:, :3]
    else:
        raise ValueError(
            f"need at least 3 coordinates per point, got shape {arr.shape}"
        )
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError(f"empty point cloud: {path!r}")
    return pts


def _voxel_grid_from_points_world(points, pitch):
    """世界坐标点量化体素，与 mesh voxelize_subdivide 的 round(p / pitch) 一致。"""
    points = np.asanyarray(points, dtype=np.float64)
    hit = np.round(points / float(pitch)).astype(np.int64)
    unique, _ = grouping.unique_rows(hit)
    occupied_index = hit[unique]
    origin_index = occupied_index.min(axis=0)
    origin_position = origin_index.astype(np.float64) * float(pitch)
    return voxel_base.VoxelGrid(
        voxel_enc.SparseBinaryEncoding(occupied_index - origin_index),
        transform=tr.scale_and_translate(scale=pitch, translate=origin_position),
    )


def translate_voxels(voxels, dx=0, dy=0, dz=0, fill=0):
    """
    离散平移体素数组，不回卷；越界部分丢弃，新空出来位置用 fill 填充。
    voxels: ndarray, shape (X,Y,Z)
    dx,dy,dz: int，沿 X/Y/Z 方向平移的体素数。正数表示索引增大方向。
    """
    out = np.full_like(voxels, fill)
    X, Y, Z = voxels.shape

    # 源区间
    xs0, xs1 = max(0, -dx), min(X, X - dx)
    ys0, ys1 = max(0, -dy), min(Y, Y - dy)
    zs0, zs1 = max(0, -dz), min(Z, Z - dz)

    # 目标区间
    xd0, xd1 = xs0 + dx, xs1 + dx
    yd0, yd1 = ys0 + dy, ys1 + dy
    zd0, zd1 = zs0 + dz, zs1 + dz

    if xs0 < xs1 and ys0 < ys1 and zs0 < zs1:
        out[xd0:xd1, yd0:yd1, zd0:zd1] = voxels[xs0:xs1, ys0:ys1, zs0:zs1]
    return out

def move_to_corner(voxels, corner="000", margin=0):
    # corner 用 'xyz' 三位字符表示目标角：'0' 表示靠近索引0端，'1' 表示靠近最大端
    idx = np.argwhere(voxels > 0)
    if len(idx) == 0:
        return voxels
    minv = idx.min(axis=0)  # (xmin,ymin,zmin)
    maxv = idx.max(axis=0)  # (xmax,ymax,zmax)
    X,Y,Z = voxels.shape

    # 计算要移到目标角所需的 dx,dy,dz
    d = []
    for a, N in zip(range(3), (X,Y,Z)):
        if corner[a] == "0":
            d.append(-minv[a] + margin)
        else:
            d.append((N-1 - maxv[a]) - margin)
    return translate_voxels(voxels, dx=d[0], dy=d[1], dz=d[2], fill=0)


def voxelize_mesh(
    filename,
    size=(96, 96, 96)
):
    #print(f'载入模型: {filename}')
    mesh = _load_trimesh_for_voxelize(filename)

    # 标准化 mesh 到单位立方体，避免尺寸问题
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.scale)

    # 设置 voxel grid 尺寸（最大尺寸为 size 的最小值）
    pitch = 1.0 / max(size)  # 单个体素边长
    vg = mesh.voxelized(pitch=pitch)
    vg_filled = vg.fill()  # 实心填充
    voxels = vg_filled.matrix  #（x,y,z）

    # 将体素裁剪/填充到目标 shape
    target_shape = size
    voxels = _resize_voxel(voxels, target_shape)
    voxels = np.swapaxes(voxels, 1, 2)
    voxels = np.flip(voxels,axis=1)
    voxels = move_to_corner(voxels, corner="000", margin=0)
    return voxels


def voxelize_pc_npy(pts, size=(96, 96, 96), fill=True):
    """
    从 .npy 读取点云并体素化；归一化与后处理与 voxelize_mesh 一致。

    Parameters
    ----------
    npy_path : str or path-like
        点云 .npy，见 load_pc_npy。
    size : (int, int, int)
        输出体素网格 shape。
    fill : bool
        False：仅标记含点的体素。True：VoxelGrid.fill()，行为接近 mesh 的实心填充。
    """
    pts = pts[:,:3].detach().float().cpu().numpy()
    pc = trimesh.PointCloud(vertices=pts)
    pc.apply_translation(-pc.centroid)
    s = float(pc.scale)
    if s <= 0 or not np.isfinite(s):
        s = 1.0
    pc.apply_scale(1.0 / s)

    pitch = 1.0 / max(size)
    vg = _voxel_grid_from_points_world(pc.vertices, pitch=pitch)
    if fill:
        vg = vg.fill()
    voxels = vg.matrix.astype(np.uint8)

    target_shape = size
    voxels = _resize_voxel(voxels, target_shape)
    voxels = np.swapaxes(voxels, 1, 2)
    voxels = np.flip(voxels, axis=1)
    voxels = move_to_corner(voxels, corner="000", margin=0)
    return voxels


def _resize_voxel(voxels, target_shape):
    """将任意大小的体素矩阵 resize 到固定大小（用 0 padding 或裁剪）"""
    result = np.zeros(target_shape, dtype=np.uint8)
    z, y, x = voxels.shape
    tz, ty, tx = target_shape

    min_z = min(z, tz)
    min_y = min(y, ty)
    min_x = min(x, tx)

    result[:min_z, :min_y, :min_x] = voxels[:min_z, :min_y, :min_x]
    return result

def save_slices_pillow(occ, out_dir="layer_ps",idx='2', axis=2, cell_px=24):
    """
    使用 Pillow 精确生成每一层切片，确保格子比例完美。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 定义颜色 (R, G, B)
    bg_color = (95, 95, 95)      # "#5f5f5f" 
    fill_color = (242, 214, 214) # "#f2d6d6"
    grid_color = (31, 35, 48)    # "#1f2330"
    grid_width = 1               # 网格线宽度

    n = occ.shape[axis]
    for k in range(n):
        # 提取切片
        sl = np.take(occ, k, axis=axis).astype(bool)
        H, W = sl.shape

        # 计算最终图像尺寸：格子总像素 + 网格线宽度
        # 如果你希望网格线压在格子边缘，可以稍微调整计算方式
        img_w = W * cell_px
        img_h = H * cell_px
        
        # 1. 创建画布
        img = Image.new("RGB", (img_w, img_h), bg_color)
        draw = ImageDraw.Draw(img)

        # 2. 填充填充色区域 (occ == 1)
        # 优化：通过遍历 True 的索引来画矩形
        rows, cols = np.where(sl)
        for r, c in zip(rows, cols):
            # 计算当前格子的左上角和右下角坐标
            x0, y0 = c * cell_px, r * cell_px
            x1, y1 = x0 + cell_px, y0 + cell_px
            draw.rectangle([x0, y0, x1, y1], fill=fill_color)

        # 3. 绘制网格线
        # 画垂直线
        for x in range(0, img_w + 1, cell_px):
            draw.line([(x, 0), (x, img_h)], fill=grid_color, width=grid_width)
        # 画水平线
        for y in range(0, img_h + 1, cell_px):
            draw.line([(0, y), (img_w, y)], fill=grid_color, width=grid_width)

        # 4. 保存
        img.save(os.path.join(out_dir, f'{idx}',f"layer{k}.png"))


if __name__ == '__main__':
    import argparse
    len1 = 128
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='输入的 mesh 文件路径 (.ply / .obj)')
    parser.add_argument('--size0', type=int, default=len1)
    parser.add_argument('--size1', type=int, default=len1)
    parser.add_argument('--size2', type=int, default=len1)

    args = parser.parse_args()

    voxel = voxelize_mesh(
        filename=args.input,
        size=(args.size0, args.size1, args.size2)
    )

    save_slices_pillow(voxel)

    