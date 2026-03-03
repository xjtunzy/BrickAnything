import numpy as np
from mesh2brick.legoblockgraph import LegoBlockGraph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh2brick.brick_structure import Brick
import os
import json



def legobrick(voxels,base_name,category_id,output_dir):
    voxels = np.rot90(voxels, k=1, axes=(1, 0))
    g = LegoBlockGraph.from_numpy(voxels)    
    g.merge_to_maximal()  
    A, _ = g.component_analysis()   #判断出模型有几个散件
    #print("components:", A)
    g.generate_single_component_analysis(max_iter=500)
    A, _ = g.component_analysis()   
    #print("components:", A)
    filepath = os.path.join(output_dir,category_id,base_name)
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
    with open(filepath2, "w", encoding="utf-8") as f:
            f.writelines(lines_out)

    #导出序列文件为json格式
    json2save = {}
    json2save['Components'] = A
    json2save['len'] = len(seq)
    json2save['Seq'] = seq
    filepath3 = os.path.join(filepath,"seq.json")
    with open(filepath3,"w",encoding="utf-8") as f:
        json.dump(json2save,f)
    
    return filepath2



if __name__ == "__main__":
    #读入体素化的结果
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="/mnt/data/yanfeng/n_project/brick_gen/DiffBrick/src/mesh2brick/voxel_numpy/0c643864f9ae458c94b2703dc56ad3f8.npy",type=str)
    args = parser.parse_args()
    print(f"download model: {args.name}")
    voxels = np.load(rf'{args.name}').astype("bool")
    voxels = np.rot90(voxels, k=1, axes=(1, 0))
    g = LegoBlockGraph.from_numpy(voxels)    
    g.merge_to_maximal()  
    A, _ = g.component_analysis()   #判断出模型有几个散件
    print("components:", A)
    g.generate_single_component_analysis(max_iter=500)
    A, _ = g.component_analysis()   
    print("components:", A)

    #将砖块导出为ldr文件
    lines_out = [] #存放修改过后的行
    for b in g.blocks:
        brick = Brick(h=b.sx,w=b.sy,x=b.x,y=b.y,z=b.z)
        #print(brick)
        line = brick.to_ldr(color=86)
        #print(line)
        lines_out.append(line)
        print(f"postion: {b.x}\t{b.y}\t{b.z}\tsize: {b.sx}\t{b.sy}")
    filepath2 = rf"/mnt/data/yanfeng/n_project/brick_gen/DiffBrick/dataset/brickmodel/0c643864f9ae458c94b2703dc56ad3f8.ldr"
    with open(filepath2, "w", encoding="utf-8") as f:
            f.writelines(lines_out)