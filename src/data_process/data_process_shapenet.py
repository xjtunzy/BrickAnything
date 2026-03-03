import os
import numpy as np
import argparse
import random
import tqdm
import json
import time
from mesh2brick import voxelization,legolazition
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


def process_shapenet_mesh(data_dir,category,out_dir,len):
    if "all" in category:
        category = list(synsetid_to_cate.values())
    for category_name in category:
        category_id = cate_to_synsetid[category_name]
        shape_net_base_dir = os.path.join(data_dir,category_id)
        i = 0
        for cur_cat in tqdm.tqdm(sorted(os.listdir(shape_net_base_dir)),f"{category_name}"):
            base_name = cur_cat  #base_name is like 10155655850468db78d106ce0a280f87
            if os.path.exists(os.path.join(out_dir,category_id,base_name)):continue
            #time.sleep(0.01)
            obj_model_path = os.path.join(shape_net_base_dir,cur_cat,"models","model_normalized.obj")
            #print(obj_model_path)
            if not os.path.exists(obj_model_path):continue

            voxel = voxelization_v1.voxelize_mesh(
            obj_model_path,
            size=(len,len,len)
            )
            
            ldr_path = legolazition_func.legobrick(voxel,base_name,category_id,out_dir)
            png_path = os.path.join(out_dir,category_id,base_name,"render.png")
            
            #render_bricks(ldr_path,png_path)


def split_dataset(data_dir,category,out_dir,test_percent):
    
    te_list = []
    tr_list = []
    if "all" in category:
        category = list(synsetid_to_cate.values())
        print("here")
    for cate_idx,subd in enumerate(category):
        data_list = []
        cate_id = cate_to_synsetid[subd]
        sub_path = os.path.join(data_dir,cate_id)
        print(sub_path)
        #data_list.extend(os.listdir(sub_path))
        for name in os.listdir(sub_path):
            data_list.append(os.path.join(cate_id, name))
        test_dataset_len = int(len(data_list)*test_percent)
        print(f"data_list len: {len(data_list)}")
        print(f"test_len : {test_dataset_len}")
        te_idx = np.random.choice(data_list,test_dataset_len,replace=False) 
        #print(f"te_idx: {type(te_idx)}")
        te_list_temp = te_idx.tolist()
        print(f"te_set: {te_list_temp}")
        tr_list_temp = [x for x in data_list if x not in te_list_temp]
        te_list.extend(te_list_temp)
        tr_list.extend(tr_list_temp)
        #print(f"tr_list: {type(tr_list)}")
    

    with open(os.path.join(out_dir, "train_ids.json"), "w") as f:
        json.dump(tr_list, f)
    with open(os.path.join(out_dir, "test_ids.json"), "w") as f:
        json.dump(te_list, f)

def refine_dataset(data_dir,category):
    #去掉末尾截止符-1
    if "all" in category:
        category = list(synsetid_to_cate.values())
        print("here")
    for category_name in category:
        category_id = cate_to_synsetid[category_name]
        brick_base_dir = os.path.join(data_dir,category_id)
        if not os.path.exists(brick_base_dir):continue
        print(f"brick_dir: {brick_base_dir}")
        for cur_cat in tqdm.tqdm(sorted(os.listdir(brick_base_dir)),f"{category_name}"):
            base_name = cur_cat  #base_name is like 10155655850468db78d106ce0a280f87
            name_id = os.path.join(category_id,base_name)
            if os.path.exists(os.path.join(brick_base_dir,base_name,'seq.json')):
                with open(os.path.join(brick_base_dir,base_name,'seq.json'), "r", encoding="utf-8") as f:
                    data = json.load(f)
                seq = data['Seq']
                seq = seq[:-1]
                new_seq = seq
                if not len(seq)%5==0:continue
                #assert len(seq)%5==0 ,f"len of seq: {len(seq)}"
                for i in range(0,len(seq)//5):
                    idx1 = 3 + i*5
                    idx2 = 4 + i*5
                    new_seq[idx1] -= 1
                    new_seq[idx2] -= 1
                data['Seq']=new_seq
                data['len']=len(new_seq)
                with open(os.path.join(brick_base_dir,base_name,'seq.json'), "w", encoding="utf-8") as f:
                     json.dump(data, f, ensure_ascii=False, indent=2)
                # print("done:")
                # print("old_len:", len(seq), "new_len:", len(new_seq))

                    
def refine_dataset_v2(data_dir,category):
    #增加name标签
    if "all" in category:
        category = list(synsetid_to_cate.values())
        print("here")
    for category_name in category:
        category_id = cate_to_synsetid[category_name]
        brick_base_dir = os.path.join(data_dir,category_id)
        if not os.path.exists(brick_base_dir):continue
        #print(f"brick_dir: {brick_base_dir}")
        for cur_cat in tqdm.tqdm(sorted(os.listdir(brick_base_dir)),f"{category_name}"):
            base_name = cur_cat  #base_name is like 10155655850468db78d106ce0a280f87
            name_id = os.path.join(category_id,base_name)
            if os.path.exists(os.path.join(brick_base_dir,base_name,'seq.json')):
                with open(os.path.join(brick_base_dir,base_name,'seq.json'), "r", encoding="utf-8") as f:
                    data = json.load(f)
                data['name'] = name_id
                with open(os.path.join(brick_base_dir,base_name,'seq.json'), "w", encoding="utf-8") as f:
                     json.dump(data, f, ensure_ascii=False, indent=2)
                # print("done:")
                # print("old_len:", len(seq), "new_len:", len(new_seq))
            
def search_component_1():
    pass

                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="/mnt/nas/yanfeng/data/n_project/ShapeNet",
                        help='Path to the ShapeNet dataset root folder.')
    parser.add_argument('--output_folder', default="/mnt/nas/yanfeng/data/n_project/BrickAnything_32",type=str, 
                        help='Path to the output folder to save processed data.')
    parser.add_argument('--output_folder_json', default="/mnt/nas/yanfeng/data/n_project/BrickAnything_partition_note",type=str, 
                        help='Path to the output folder to save processed data json.')
    parser.add_argument('--category', type=str, default=['washer','all'],
                        help='Path to the output folder to save processed data.')
    parser.add_argument('--split_test',type=float,default=0.1,
                        help="10/100 elements of the hole dataset are belong to test dataset")
    parser.add_argument('--leng',type=int,default=32)
    args = parser.parse_args()

    #os.makedirs(args.output_folder)

    random.seed(0)
    np.random.seed(0)

    process_shapenet_mesh(args.data_root,args.category,args.output_folder,args.leng)
    #split_dataset(args.data_root,args.category,args.output_folder_json,args.split_test)
    #refine_dataset(args.output_folder,args.category)
    #refine_dataset_v2(args.output_folder,args.category)




        
