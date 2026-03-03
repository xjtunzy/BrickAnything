import numpy as np
#from legoblockgraph import LegoBlockGraph
from voxel2brick import LegoBlockGraph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from brick_structure import Brick
import os
from voxelization_v1 import voxelize_mesh
from voxelization_v2 import mesh2voxel
import open3d as o3d
import json
import time
from PIL import Image, ImageDraw
#from render_bricks import render_bricks

def legobrick(voxels,base_name,idx,size,output_dir='brickmodel'):
    #voxels = np.rot90(voxels, k=1, axes=(1, 0))
    #voxels = np.swapaxes(voxels, 1, 2) #交换z，y轴
    #voxels[0,0,0]=1
    #voxels[0,0,18]=1

    save_slices_pillow(voxels,idx='3',axis=0)
    atc = time.time()
    g = LegoBlockGraph.from_numpy(voxels)    
    g.merge_to_maximal()  
    A = g.generate_single_component_analysis(max_iter=500)
    etc = time.time()
    cost_time = etc-atc
    print(f"cost_time: {cost_time:.2f} s")
    filepath = os.path.join(output_dir,base_name,f'{idx}',f'{size}')
    os.makedirs(filepath,exist_ok=True)

    bricks = []
    seq = []
    #将砖块导出为ldr文件
    lines_out = [] #存放修改过后的行
    for b in g.blocks:
        brick = Brick(h=b.sx,w=b.sy,x=b.x,y=b.y,z=b.z)
        #print(brick)
        line = brick.to_ldr(color=86)
        #print(line)
        lines_out.append(line)
        #print(f"postion: {b.x}\t{b.y}\t{b.z}\tsize: {b.sx}\t{b.sy}")
        bricks.append((b.x,b.y,b.z,b.sx,b.sy))
    bricks.sort(key=lambda t: (t[0], t[1], t[2]))
    #print(bricks)
    for b in bricks:
        seq.append(int(b[0]))
        seq.append(int(b[1]))
        seq.append(int(b[2]))
        seq.append(int(b[0]+b[3]-1))
        seq.append(int(b[1]+b[4]-1))
        #seq.append(int(b[2]))
    #seq.append("&")
    filepath2 = os.path.join(filepath,"brick.ldr")
    filepath3 = os.path.join(filepath,'render.png')
    with open(filepath2, "w", encoding="utf-8") as f:
            f.writelines(lines_out)
    
    #导出序列文件为json格式
    json2save = {}
    json2save['Components'] = A
    json2save['Cost_time'] = cost_time
    json2save['len'] = len(seq)
    json2save['Seq'] = seq
    filepath3 = os.path.join(filepath,"seq.json")
    with open(filepath3,"w",encoding="utf-8") as f:
        json.dump(json2save,f)

def save_slices_pillow(occ, out_dir="layer_ps",idx='1', axis=2, cell_px=24):
    """
    使用 Pillow 精确生成每一层切片，确保格子比例完美。
    """
    out_dir = os.path.join(out_dir,f'{idx}')
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
        #print(f'k: {k}\n{sl}')
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
        img.save(os.path.join(out_dir,f"layer{k}.png"))


if __name__ == "__main__":
    #读入体素化的结果
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument("--res", type=int, required=True)
    args = parser.parse_args()
    print(f"download model: {args.name}")
    size = args.res
    obj_path = f"data/{args.name}.obj"
    voxel1 = voxelize_mesh(
        filename=obj_path,
        size=(size,size,size)
    )
    # mesh = o3d.io.read_triangle_mesh(obj_path)
    # voxel2 = mesh2voxel(mesh,(size,size,size))
    #判断x轴朝向
    # voxel1[0,0,0]=1
    # voxel1[2,0,0]=1
    # voxel1[6,0,0]=1
    # voxel1[0,0,4] = 1
    save_slices_pillow(voxel1,idx='1')
    #save_slices_pillow(voxel2,idx='2')
    brick1 = legobrick(voxel1,args.name,idx='py',size=size)