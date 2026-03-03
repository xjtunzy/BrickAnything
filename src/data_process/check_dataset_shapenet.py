import os
import numpy as np
import argparse
import random
import tqdm
import json
import time
from mesh2brick import voxelization,legolazition
from brickanything_train.render_bricks import render_bricks
from seq2brick import seq2brick
from mesh2brick.brick_library import dimensions_to_brick_id

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


def check_dataset(data_dir,category):
    #需要-1
    bad_dataset = []
    if "all" in category:
        category = list(synsetid_to_cate.values())
        print("here")
    for category_name in category:
        category_id = cate_to_synsetid[category_name]
        brick_base_dir = os.path.join(data_dir,category_id)
        assert os.path.exists(brick_base_dir)
        print(f"brick_dir: {brick_base_dir}")
        for cur_cat in tqdm.tqdm(sorted(os.listdir(brick_base_dir)),f"{category_name}"):
            base_name = cur_cat  #base_name is like 10155655850468db78d106ce0a280f87
            if os.path.exists(os.path.join(brick_base_dir,base_name,'seq.json')):
                with open(os.path.join(brick_base_dir,base_name,'seq.json'), "r", encoding="utf-8") as f:
                    data = json.load(f)
                seq = data['Seq']
                for e in seq:
                    if type(e)==str:
                        if category_name not in bad_dataset:bad_dataset.append(f'{category_name}')
                        break
                    elif e<0:
                        if category_name not in bad_dataset:bad_dataset.append(f'{category_name}')
                        break
        print(bad_dataset)
                
def check_dataset_v2(data_dir,category):
    #需要-1
    bad_dataset = []
    if "all" in category:
        category = list(synsetid_to_cate.values())
        print("here")
    for category_name in category:
        category_id = cate_to_synsetid[category_name]
        brick_base_dir = os.path.join(data_dir,category_id)
        assert os.path.exists(brick_base_dir)
        print(f"brick_dir: {brick_base_dir}")
        for cur_cat in tqdm.tqdm(sorted(os.listdir(brick_base_dir)),f"{category_name}"):
            base_name = cur_cat  #base_name is like 10155655850468db78d106ce0a280f87
            bricks = []
            if os.path.exists(os.path.join(brick_base_dir,base_name,'seq.json')):
                with open(os.path.join(brick_base_dir,base_name,'seq.json'), "r", encoding="utf-8") as f:
                    data = json.load(f)
                seq = data['Seq']
            seq2brick(seq,bricks)
            for b in bricks:
                dimensions_to_brick_id(b[0],b[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/mnt/nas/yanfeng/data/n_project/ShapeNet",
                        help='Path to the ShapeNet dataset root folder.')
    parser.add_argument('--output_folder', default="/mnt/nas/yanfeng/data/n_project/BrickAnything",type=str, 
                        help='Path to the output folder to save processed data.')
    parser.add_argument('--output_folder_json', default="/mnt/nas/yanfeng/data/n_project/BrickAnything_partition_note",type=str, 
                        help='Path to the output folder to save processed data json.')
    parser.add_argument('--category', type=str, default=['all'],
                        help='Path to the output folder to save processed data.')
    parser.add_argument('--split_test',type=float,default=0.1,
                        help="10/100 elements of the hole dataset are belong to test dataset")
    parser.add_argument('--leng',type=int,default=20)
    args = parser.parse_args()

    #os.makedirs(args.output_folder)

    random.seed(0)
    np.random.seed(0)

    #process_shapenet_mesh(args.data_root,args.category,args.output_folder,args.leng)
    #split_dataset(args.data_root,args.category,args.output_folder_json,args.split_test)
    check_dataset_v2(args.output_folder,args.category)
