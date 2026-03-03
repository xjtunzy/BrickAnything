import numpy as np
import sys
sys.path.append("/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/src/mesh2brick_v2/cpp/build")
import voxel2brick_cpp
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh2brick_v2.brick_structure import Brick
import os
from mesh2brick_v2.voxelization_v1 import voxelize_mesh
#from voxelization_v2 import mesh2voxel
import open3d as o3d
import json
import time
from PIL import Image, ImageDraw


def legobrick(voxels,base_name,category_id,output_dir):
    atc = time.time()
    voxels_u8 = np.ascontiguousarray(voxels, dtype=np.uint8)
    res = voxel2brick_cpp.solve(voxels_u8, max_iter=100, seed=12345)
    A = int(res["components"])
    bricks_cpp = list(res["bricks"])  # [(x,y,z,sx,sy), ...]
    etc = time.time()
    cost_time = etc - atc
    #print(f"cost_time: {cost_time:.2f} s")
    filepath = os.path.join(output_dir,category_id,base_name)
    os.makedirs(filepath,exist_ok=True)
    bricks = []
    seq = []
    lines_out = []
    for x, y, z, sx, sy in bricks_cpp:
        brick = Brick(h=sx, w=sy, x=x, y=y, z=z)
        line = brick.to_ldr(color=86)
        lines_out.append(line)
        bricks.append((x, y, z, sx, sy))
    bricks.sort(key=lambda t: (t[2],t[0],t[1]))
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
    return filepath2
