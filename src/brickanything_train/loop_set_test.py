import os
import json
import numpy as np
#from brickanything_train.eval_cond_gpt import evaluate as evaluate_cond_gpt
from torch.utils.data import Dataset

import trimesh
import networkx as nx
from torch.utils.data import DataLoader
from render_pointcloud import render_pointcloud_xyz_to_png




# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


def normalize_to_unit_box(vertices, eps=1e-8):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    center = (bounds[0] + bounds[1]) / 2.0
    scale = np.maximum((bounds[1] - bounds[0]).max(), eps)
    v = (vertices - center[None, :]) / scale  # roughly [-0.5, 0.5]
    #v = np.clip(v, -0.5, 0.5)
    return v

def sample_surface_points(vertices, faces, sample_num=4096):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, force="mesh", merge_primitives=True, process=False)
    pts, face_idx = mesh.sample(sample_num, return_index=True)
    nrm = mesh.face_normals[face_idx]

    pts = pts.astype(np.float16)
    pts = np.clip(pts, -0.9995, 0.9995)
    nrm = nrm.astype(np.float16)
    return np.concatenate([pts, nrm], axis=-1)  # (N,6)



class BrickDataset(Dataset):
    def __init__(self,
                 data_dir,
                 mesh_dir,
                 categorys,
                 split_set="train",
                 pc_num = 8192,
                 max_len =1600*9*0.7 ):
        super().__init__()  
        self.data_dir = data_dir
        self.mesh_dir = mesh_dir
        self.categorys = categorys
        self.split_set=split_set
        self.pc_num = pc_num
        self.max_seq_len = int(max_len)      # 统一长度
        self.pad_id = -1


        self.data = []
        #读取划分文件
        with open("/mnt/nas/yanfeng/data/n_project/BrickAnything_partition_note/train_ids.json","r") as f:
            train_idx = json.load(f)
        with open("/mnt/nas/yanfeng/data/n_project/BrickAnything_partition_note/test_ids.json","r") as f:
            test_idx = json.load(f)
        
        self.subdirs = [cate_to_synsetid[c] for c in self.categorys]

        for cate_idx,subd in enumerate(self.subdirs):
            # NOTE: [subd] here is synset id  ,like 02880940
            sub_path = os.path.join(data_dir,subd)

            if not os.path.isdir(sub_path):
                print("Directory missing : %s" % sub_path)
                continue
            
            count = 0
            max_len = 0
            for name_id in os.listdir(sub_path):
                search_name = os.path.join(subd,name_id)
                #print(search_name)
                count +=1
                if self.split_set == "train":
                    if search_name in train_idx:
                        self.data.append(search_name)
            
            #print(self.data)
            #print(len(self.data)/count)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_path = self.data[idx]
        print(f"data_path: {data_path}")
        data_dict = {}
        seq_path = os.path.join(self.data_dir,data_path,"seq.json")
        mesh_path = os.path.join(self.mesh_dir,data_path,"models","model_normalized.obj")
        assert os.path.exists(seq_path), f"路径 {seq_path} 不存在"
        assert os.path.exists(mesh_path), f"路径 {mesh_path} 不存在"
        mesh = trimesh.load(mesh_path, force="mesh")

        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)

        v = normalize_to_unit_box(v)
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_unreferenced_vertices()
        v = np.clip(v, -0.5, 0.5)
        

        #pc_normal = sample_surface_points(np.asarray(m.vertices), np.asarray(m.faces), self.pc_num)
        mv = np.asarray(m.vertices) * (2 * 0.9995)
        mv = np.clip(mv, -0.9995, 0.9995)   # 和上一份一致（可选，但建议加）
        mf = np.asarray(m.faces)
        pc_normal = sample_surface_points(mv, mf, self.pc_num)
        data_dict["pc_normal"] = pc_normal  # (pc_num,6), float16

        with open(seq_path,"r") as f:
            seq = json.load(f)['Seq']
            seq =  seq[:-1]
        seq = np.asarray(seq,dtype=np.int64)
        pad_seq = np.ones((self.max_seq_len,), dtype=np.int64) * self.pad_id
        L = min(len(seq), self.max_seq_len)
        pad_seq[:L] = seq[:L]
        
        data_dict['sequence'] = pad_seq
        data_dict['seq_len'] = L

        return data_dict
    

if __name__ == "__main__":
    data_dir = "/mnt/nas/yanfeng/data/n_project/BrickAnything"
    mesh_dir = "/mnt/nas/yanfeng/data/n_project/ShapeNet"
    cur_data_dir = "03001627/1ac6531a337de85f2f7628d6bf38bcc4"
    subdir = ['chair']
    print(f"subdir: {subdir}")
    dataset = BrickDataset(data_dir,mesh_dir,subdir)

    print("Dataset length:", len(dataset))

    sample = dataset[4]

    print("Keys:", sample.keys())
    print("Sequence type:", type(sample["sequence"]), "len:", len(sample["sequence"]))
    print("PC shape:", sample["pc_normal"].shape)
    print("PC dtype:", sample["pc_normal"].dtype)
    print("PC range:", sample["pc_normal"][:, :3].min(), sample["pc_normal"][:, :3].max())
    render_pointcloud_xyz_to_png(sample["pc_normal"][:,:3],"/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/test_output/pc.png",2048)






