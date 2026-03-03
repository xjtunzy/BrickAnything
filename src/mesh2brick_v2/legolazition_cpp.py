import numpy as np
import sys
sys.path.append("/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/src/mesh2brick_v2/cpp/build")
import voxel2brick_cpp
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from brick_structure import Brick
import os
#from voxelization_v1 import voxelize_mesh
from voxelization_v2 import voxelize_mesh
import open3d as o3d
import json
import time
from PIL import Image, ImageDraw


#tmp code
import collections
def is_overlap(a_range, b_range):
    return max(a_range[0], b_range[0]) < min(a_range[1], b_range[1])

def get_components(bricks_list):
    n = len(bricks_list)
    adj = collections.defaultdict(list)
    # 建立邻接表：Z轴层级相邻且投影重叠的视为相连
    for i in range(n):
        x1, y1, z1, sx1, sy1 = bricks_list[i]
        for j in range(i + 1, n):
            x2, y2, z2, sx2, sy2 = bricks_list[j]
            # 只有 Z 轴相邻才算垂直连通
            if abs(z1 - z2) == 1:
                if is_overlap((x1, x1+sx1), (x2, x2+sx2)) and \
                   is_overlap((y1, y1+sy1), (y2, y2+sy2)):
                    adj[i].append(j)
                    adj[j].append(i)
    
    # BFS 寻找组件
    visited = [-1] * n
    comp_id = 0
    for i in range(n):
        if visited[i] == -1:
            queue = collections.deque([i])
            visited[i] = comp_id
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = comp_id
                        queue.append(v)
            comp_id += 1
    return visited, comp_id

def legobrick(voxels, base_name, idx, size, output_dir='brickmodel'):
    atc = time.time()
    voxels_u8 = np.ascontiguousarray(voxels, dtype=np.uint8)
    res = voxel2brick_cpp.solve(voxels_u8, max_iter=100, seed=12345)
    A = int(res["components"])
    bricks_cpp = list(res["bricks"])  # [(x,y,z,sx,sy), ...]
    etc = time.time()
    cost_time = etc - atc
    print(f"cost_time: {cost_time:.2f} s")

    #_____debug code : visuliaze diff components_______________________________________
    comp_labels, total_comps = get_components(bricks_cpp)
    ldraw_colors = [86, 18, 19, 14, 13, 20, 23, 26, 27, 25, 9, 10, 11, 12, 15]
    filepath = os.path.join(output_dir, base_name, f'{idx}', f'{size}')
    os.makedirs(filepath, exist_ok=True)
    bricks = []
    seq = []
    lines_out = []
    for i,(x, y, z, sx, sy) in enumerate(bricks_cpp):
        brick = Brick(h=sx, w=sy, x=x, y=y, z=z)
        color_idx = comp_labels[i] % len(ldraw_colors)
        current_color = ldraw_colors[color_idx]
        line = brick.to_ldr(color=current_color)
        lines_out.append(line)
        bricks.append((x, y, z, sx, sy))
    bricks.sort(key=lambda t: (t[0], t[1], t[2]))
    for b in bricks:
        seq.append(int(b[0]))
        seq.append(int(b[1]))
        seq.append(int(b[2]))
        seq.append(int(b[0] + b[3] - 1))
        seq.append(int(b[1] + b[4] - 1))

    filepath2 = os.path.join(filepath, "brick.ldr")
    with open(filepath2, "w", encoding="utf-8") as f:
        f.writelines(lines_out)

    json2save = {
        'Components': A,
        'Cost_time': cost_time,
        'len': len(seq),
        'Seq': seq
    }
    filepath3 = os.path.join(filepath, "seq.json")
    with open(filepath3, "w", encoding="utf-8") as f:
        json.dump(json2save, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument("--res", type=int, required=True)
    args = parser.parse_args()
    print(f"download model: {args.name}")
    size = args.res
    obj_path = f"data/{args.name}.obj"
    voxel1 = voxelize_mesh(filename=obj_path, size=(size, size, size))
    brick1 = legobrick(voxel1, args.name, idx='cpp', size=size)
