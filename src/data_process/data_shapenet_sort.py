import json
import os
import numpy as np
from tqdm import tqdm
import argparse

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

#zyx
def sort_bricks_1(seq_root,out_root):
    category = list(synsetid_to_cate.values())
    Error_id = []
    for category_name in category:
        category_id = cate_to_synsetid[category_name]
        shape_net_base_dir = os.path.join(seq_root,category_id)
        shapenet_out_dir = os.path.join(out_root,category_id)
        for cur_cat in tqdm(sorted(os.listdir(shape_net_base_dir)),f"{category_name}"):
            base_name = cur_cat  #base_name is like 10155655850468db78d106ce0a280f87
            json_path = os.path.join(shape_net_base_dir,base_name,'seq.json')
            if  not os.path.exists(json_path):
                Error_id.append(json_path)
                continue
            json2save_dir = os.path.join(shapenet_out_dir,base_name)
            
            os.makedirs(json2save_dir,exist_ok=True)
            json2save_path = os.path.join(json2save_dir,'seq.json')
            if os.path.exists(json2save_path):continue
            with open(json_path,'r') as f:
                seq = json.load(f)
            brick_seq = seq['Seq']
            brick_stable = seq['Stable']
            brick_seq_len = seq['len']
            bricks = []
            assert len(brick_seq)%5==0,f"len of seq is error:{len(brick_seq)}"
            for i in range(0,len(brick_seq)//5):
                idx1 = 5*i
                idx2 = 5*i+1
                idx3 = 5*i+2
                idx4 = 5*i+3
                idx5 = 5*i+4
                x = brick_seq[idx1]
                y = brick_seq[idx2]
                z = brick_seq[idx3]
                h = brick_seq[idx4]-x+1
                w = brick_seq[idx5]-y+1
                #匹配砖块型号
                #print(f"{h}*{w} {x},{y},{z}")
                bricks.append([x,y,z,h,w])
            bricks.sort(key=lambda b: (b[2], b[1], b[0]))
            brick_new_seq = []
            for b in bricks:
                brick_new_seq.append(int(b[0]))
                brick_new_seq.append(int(b[1]))
                brick_new_seq.append(int(b[2]))
                brick_new_seq.append(int(b[0] + b[3] - 1))
                brick_new_seq.append(int(b[1] + b[4] - 1))
            json2save = {
                'Stable': brick_stable,
                'len': brick_seq_len,
                'Seq': brick_new_seq
            }
            with open(json2save_path, "w", encoding="utf-8") as f:
                json.dump(json2save, f)
    print(Error_id)
            

                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default='shapenet_20_zyx',type=str)
    parser.add_argument("--seq_root", default="/mnt/nas/yanfeng/data/n_project/shapenet_20", type=str)
    parser.add_argument("--out_dir", default="/mnt/nas/yanfeng/data/n_project", type=str)
    args = parser.parse_args()
    out_root = os.path.join(args.out_dir,args.mode)
    sort_bricks_1(args.seq_root,out_root)

