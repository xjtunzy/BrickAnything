from multiprocessing import Pool, cpu_count # 引入多进程模块
import os
import numpy as np
import argparse
import random
import tqdm
import json
import time
from mesh2brick_v2 import voxelization_v1,legolazition_func
from brickanything_train.render_bricks import render_bricks

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane',     '02773838': 'bag',          '02801938': 'basket',
    '02808440': 'bathtub',      '02818832': 'bed',          '02828884': 'bench',
    '02876657': 'bottle',       '02880940': 'bowl',         '02924116': 'bus',
    '02933112': 'cabinet',      '02747177': 'can',          '02942699': 'camera',
    '02954340': 'cap',          '02958343': 'car',          '03001627': 'chair',
    '03046257': 'clock',        '03207941': 'dishwasher',   '03211117': 'monitor',
    '04379243': 'table',        '04401088': 'telephone',    '02946921': 'tin_can',
    '04460130': 'tower',        '04468005': 'train',        '03085013': 'keyboard',
    '03261776': 'earphone',     '03325088': 'faucet',       '03337140': 'file',
    '03467517': 'guitar',       '03513137': 'helmet',       '03593526': 'jar',
    '03624134': 'knife',        '03636649': 'lamp',         '03642806': 'laptop',
    '03691459': 'speaker',      '03710193': 'mailbox',      '03759954': 'microphone',
    '03761084': 'microwave',    '03790512': 'motorcycle',   '03797390': 'mug',
    '03928116': 'piano',        '03938244': 'pillow',       '03948459': 'pistol',
    '03991062': 'pot',          '04004475': 'printer',      '04074963': 'remote_control',
    '04090263': 'rifle',        '04099429': 'rocket',       '04225987': 'skateboard',
    '04256520': 'sofa',         '04330267': 'stove',        '04530566': 'vessel',
    '04554684': 'washer',       '02992529': 'cellphone',
    '02843684': 'birdhouse',    '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


def process_single_mesh(item_info):
    """
    item_info 是一个字典，包含处理所需的所有参数
    """
    obj_model_path = item_info['obj_model_path']
    len_val = item_info['len']
    base_name = item_info['base_name']
    category_id = item_info['category_id']
    out_dir = item_info['out_dir']
    
    # 检查是否已经处理过
    if os.path.exists(os.path.join(out_dir, category_id, base_name)):
        return f"Skipped: {base_name}"

    try:
        # 执行体素化
        voxel = voxelization_v1.voxelize_mesh(
            obj_model_path,
            size=(len_val, len_val, len_val)
        )
        # 执行乐高化
        ldr_path = legolazition_func.legobrick(voxel, base_name, category_id, out_dir)
        return f"Success: {base_name}"
    except Exception as e:
        return f"Error in {base_name}: {e}"


def process_shapenet_mesh(data_dir, category, out_dir, leng):
    if "all" in category:
        category = list(synsetid_to_cate.values())
    
    # 1. 收集所有待处理的任务
    task_list = []
    for category_name in category:
        category_id = cate_to_synsetid[category_name]
        shape_net_base_dir = os.path.join(data_dir, category_id)
        
        if not os.path.exists(shape_net_base_dir):
            continue

        for cur_cat in sorted(os.listdir(shape_net_base_dir)):
            obj_model_path = os.path.join(shape_net_base_dir, cur_cat, "models", "model_normalized.obj")
            if not os.path.exists(obj_model_path):
                continue
            
            # 将任务参数打包
            task_list.append({
                'obj_model_path': obj_model_path,
                'len': leng,
                'base_name': cur_cat,
                'category_id': category_id,
                'out_dir': out_dir
            })

    # 2. 使用多进程池并行执行
    #num_processes = min(cpu_count(),8)  # 获取你电脑的 CPU 核心数
    num_processes = 4 
    print(f"Starting limited core processing (4 cores). Total tasks: {len(task_list)}")
    
    # 2. 强制每任务重启进程 (maxtasksperchild=1)
    # 3. 显式设置 chunksize=1
    with Pool(processes=num_processes, maxtasksperchild=50) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(process_single_mesh, task_list, chunksize=1), 
                          total=len(task_list), 
                          desc="Processing Bricks"):
            pass
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/mnt/nas/yanfeng/data/n_project/ShapeNet",
                        help='Path to the ShapeNet dataset root folder.')
    parser.add_argument('--output_folder', default="/mnt/nas/yanfeng/data/n_project/BrickAnything_64",type=str, 
                        help='Path to the output folder to save processed data.')
    parser.add_argument('--output_folder_json', default="/mnt/nas/yanfeng/data/n_project/BrickAnything_partition_note",type=str, 
                        help='Path to the output folder to save processed data json.')
    parser.add_argument('--category', type=str, default=['washer','all'],
                        help='Path to the output folder to save processed data.')
    parser.add_argument('--split_test',type=float,default=0.1,
                        help="10/100 elements of the hole dataset are belong to test dataset")
    parser.add_argument('--leng',type=int,default=64)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    process_shapenet_mesh(args.data_root, args.category, args.output_folder, args.leng)