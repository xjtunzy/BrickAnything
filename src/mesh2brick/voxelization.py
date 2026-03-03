import trimesh
import numpy as np
import os

def voxelize_mesh(
    filename,
    output_json_path='./voxel_json/',
    output_numpy_path='./voxel_numpy/',
    output_binvox_path='./voxel_binvox/',
    size=(96, 96, 96)
):
    #print(f'载入模型: {filename}')
    mesh = trimesh.load(filename, force='mesh')

    # 标准化 mesh 到单位立方体，避免尺寸问题
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / mesh.scale)

    # 设置 voxel grid 尺寸（最大尺寸为 size 的最小值）
    pitch = 1.0 / max(size)  # 单个体素边长
    vg = mesh.voxelized(pitch=pitch)
    vg_filled = vg.fill()  # 实心填充
    voxels = vg_filled.matrix  # (Z, Y, X)

    # 将体素裁剪/填充到目标 shape
    target_shape = size
    voxels = _resize_voxel(voxels, target_shape)
    voxels_xyz = np.transpose(voxels, (2, 1, 0))  # (X, Y, Z)
    # 输出基础名
    # base_name = os.path.splitext(os.path.basename(filename))[0]

    # # 保存为 .npy
    # os.makedirs(output_numpy_path, exist_ok=True)
    # np.save(os.path.join(output_numpy_path, base_name + ".npy"), voxels_xyz)
    # print(f"已保存 numpy 文件: {os.path.join(output_numpy_path, base_name + '.npy')}")

    # print(f"体素化完成！尺寸: {voxels.shape}")
    return voxels_xyz


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


if __name__ == '__main__':
    import argparse
    len = 20
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='输入的 mesh 文件路径 (.ply / .obj)')
    parser.add_argument('--json_dir', type=str, default='./voxel_json/')
    parser.add_argument('--numpy_dir', type=str, default='./voxel_numpy/')
    parser.add_argument('--binvox_dir', type=str, default='./voxel_binvox/')
    parser.add_argument('--size0', type=int, default=len)
    parser.add_argument('--size1', type=int, default=len)
    parser.add_argument('--size2', type=int, default=len)

    args = parser.parse_args()

    _,base_name = voxelize_mesh(
        filename=args.input,
        output_json_path=args.json_dir,
        output_numpy_path=args.numpy_dir,
        output_binvox_path=args.binvox_dir,
        size=(args.size0, args.size1, args.size2)
    )
    # import open3d as o3d
    # import numpy as np

    # # 加载体素矩阵
    # voxels = np.load(rf'voxel_numpy/{base_name}.npy')

    # # 获取体素坐标
    # z, y, x = np.where(voxels > 0)

    # # 体素边长（可调）
    # voxel_size = 1.0

    # # 这些占据体素的中心坐标（换成世界坐标系）
    # pts = np.vstack([x, y, z]).T.astype(np.float32) * voxel_size

    # # 构造点云
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])
    # # 用“点云 → 体素网格”的工厂方法生成 VoxelGrid
    # min_bound = pts.min(0) - voxel_size * 0.5
    # max_bound = pts.max(0) + voxel_size * 0.5
    # vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
    #     pcd, voxel_size=voxel_size, min_bound=min_bound, max_bound=max_bound
    # )
    # o3d.visualization.draw_geometries([vg])