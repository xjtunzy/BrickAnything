import numpy as np
import json
import os
from brickanything_train.render_bricks import render_bricks
from pathlib import Path
from mesh2brick.brick_structure import Brick
from brickanything_train.render_pointcloud import render_pointcloud_xyz_to_png

def seq2brick(seq,bricks):
    assert len(seq)%5==0,f"len of seq is error:{len(seq)}"
    for i in range(0,len(seq)//5):
        idx1 = 5*i
        idx2 = 5*i+1
        idx3 = 5*i+2
        idx4 = 5*i+3
        idx5 = 5*i+4
        x = seq[idx1]
        y = seq[idx2]
        z = seq[idx3]
        h = seq[idx4]-x+1
        w = seq[idx5]-y+1
        #匹配砖块型号
        #print(f"{h}*{w} {x},{y},{z}")
        bricks.append([h,w,x,y,z])

def brick2ldr(bricks,ldr_path):
    lines_out = [] #存放修改过后的行
    for b in bricks:
        brick = Brick(h=b[0],w=b[1],x=b[2],y=b[3],z=b[4])
        #print(brick)
        line = brick.to_ldr(color=225)
        #print(line)
        lines_out.append(line)

    with open(ldr_path, "w", encoding="utf-8") as f:
            f.writelines(lines_out)


if __name__ == "__main__":
    data_dir = "/mnt/nas/yanfeng/data/n_project/shapenet_20/02801938/1f6046149060eb81cbde89e0c48a01bf"
    out_dir = "/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/test_output"
    ldr_path = os.path.join(out_dir,"brick.ldr")
    ldr_dir = os.path.join(data_dir,"brick.ldr")
    render_path1 = os.path.join(out_dir,"t1.png")
    render_path2 = os.path.join(out_dir,"t2.png")
    render_path3 = os.path.join(out_dir,'t3.png')
    json_path = os.path.join(data_dir,"seq.json")
    with open(json_path,'r') as f:
        seq = json.load(f)['Seq']
        #seq = seq[:-1]
    bricks = []
    seq2brick(seq,bricks)
    brick2ldr(bricks,ldr_dir)
    #render bricks
    render_bricks(ldr_dir,render_path1)
    #render_bricks(ldr_path,render_path2)
    #render_pointcloud_xyz_to_png()
