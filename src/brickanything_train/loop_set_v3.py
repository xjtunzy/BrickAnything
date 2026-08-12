import os
import json
import numpy as np
import trimesh
from brickanything_train.eval_cond_gpt import evaluate as evaluate_cond_gpt


def normalize_to_unit_box(vertices, eps=1e-8):
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    center = (bounds[0] + bounds[1]) / 2.0
    scale = np.maximum((bounds[1] - bounds[0]).max(), eps)
    return (vertices - center[None, :]) / scale  # roughly [-0.5, 0.5]

def sample_surface_points(vertices, faces, sample_num=4096):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, force="mesh", merge_primitives=True, process=False)
    pts, face_idx = mesh.sample(sample_num, return_index=True)
    nrm = mesh.face_normals[face_idx]
    pts = pts.astype(np.float16)
    pts = np.clip(pts, -0.9995, 0.9995)
    nrm = nrm.astype(np.float16)
    return np.concatenate([pts, nrm], axis=-1)  # (N,6)


#需要添加id映射逻辑


class Dataset():
    """
    读取 split/train.json or split/test.json
    每条样本包含 npz_path,npz 内含 vertices/faces/seq
    """
    def __init__(self,
                 split,          # e.g. /.../split/train.json
                 n_discrete_size,
                 mode,
                 pc_num=8192,
                 max_seq_len=int(1000*5*1),
                 pad_id=-1,
                 renormalize=False,
                 clean_mesh=False,
                 augment=False):
        self.split = split
        self.pc_num = pc_num
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)
        self.renormalize = renormalize
        self.clean_mesh = clean_mesh
        self.augment = bool(augment)
        print(f"augment: {self.augment}")
        if mode == "sft_shapenet":
            split_json = os.path.join('/mnt/nas/yanfeng/results/BrickAnything/dataset/brickanything_sft/split',f'{split}.json')
        elif mode == "robot":
            split_json = os.path.join('/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/dataset/shapenet_20_v1/split',f'{split}.json')
        elif mode == 'all':
            split_json = os.path.join('/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/dataset/shapenet_20_v3/split',f'{split}.json')
        elif mode == 'brickanything':
            split_json = os.path.join('/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/dataset/brickanything_dataset_v1_zyx/split',f'{split}.json')
        elif mode == 'brickanything_v1':
            split_json = os.path.join('/mnt/nas/yanfeng/results/BrickAnything/dataset/brickanything_v1/split',f'{split}.json')
        elif mode == 'brickanything_tree_v1':
            #tree
            split_json = os.path.join('/mnt/nas/yanfeng/results/BrickAnything/dataset/brickanything_v3/split',f'{split}.json')
        self.eval_func = evaluate_cond_gpt
        with open(split_json, "r") as f:
            self.items = json.load(f)  # list[dict], each has npz_path
        
        # Token layout (n_discrete_size=20): root xyz in [0,20), f in [20,44),
        # m in [44,56), h/w in [56,61), EOP=61.
        self.F_offset = n_discrete_size
        self.M_offset = n_discrete_size + 24
        self.H_W = {'1': 56, '2': 57, '4': 58, '6': 59, '8': 60}
        self.Eop = 61

    def id_mapping(self, seq):
        if len(seq) < 5:
            raise ValueError(f"Invalid connectivity sequence length: {len(seq)}")

        mapped_seq = []

        # Root format: [x, y, z, h, w]. Coordinates keep their original ids.
        mapped_seq.extend(seq[:3].tolist())
        mapped_seq.append(self.H_W[str(int(seq[3]))])
        mapped_seq.append(self.H_W[str(int(seq[4]))])

        child_pos = 0  # 0:f, 1:h, 2:w, 3:m
        for token in seq[5:]:
            token = int(token)
            if token == 101:
                mapped_seq.append(self.Eop)
                child_pos = 0
                continue

            if child_pos == 0:
                mapped_seq.append(token + self.F_offset)
            elif child_pos == 1:
                mapped_seq.append(self.H_W[str(token)])
            elif child_pos == 2:
                mapped_seq.append(self.H_W[str(token)])
            else:
                mapped_seq.append(token + self.M_offset)

            child_pos = (child_pos + 1) % 4

        if child_pos != 0:
            raise ValueError("Invalid connectivity sequence: unfinished [f, h, w, m] group before sequence end.")

        return np.asarray(mapped_seq, dtype=np.int64)

    def __len__(self):
        return len(self.items)

    def _augment_vertices(self, vertices):
        # Random z-axis rotation keeps canonical upright assumption.
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_z = np.array(
            [[cos_t, -sin_t, 0.0],
             [sin_t,  cos_t, 0.0],
             [0.0,    0.0,   1.0]],
            dtype=np.float32,
        )
        vertices = vertices @ rot_z.T

        # Mild random scaling and translation.
        scale = np.random.uniform(0.95, 1.05)
        vertices = vertices * scale
        trans = np.random.uniform(-0.02, 0.02, size=(1, 3)).astype(np.float32)
        vertices = vertices + trans
        return np.clip(vertices, -0.5, 0.5)

    def __getitem__(self, idx):
        item = self.items[idx]
        npz_path = item["npz_path"]
        d = np.load(npz_path, allow_pickle=False)
        v = d["vertices"].astype(np.float32)
        f = d["faces"].astype(np.int64)
        seq = d["seq"].astype(np.int64)

        # 可选：再次清洗（防止有些 mesh 预处理时漏掉）
        if self.clean_mesh:
            m = trimesh.Trimesh(vertices=v, faces=f, process=False)
            m.merge_vertices()
            m.update_faces(m.nondegenerate_faces())
            m.update_faces(m.unique_faces())
            m.remove_unreferenced_vertices()
            v = np.asarray(m.vertices, dtype=np.float32)
            f = np.asarray(m.faces, dtype=np.int64)

        # 你的预处理已经做过归一化并 clip 到 [-0.5,0.5]，一般不需要再做
        if self.renormalize:
            v = normalize_to_unit_box(v)
            v = np.clip(v, -0.5, 0.5)

        if self.augment and self.split == "train":
            v = self._augment_vertices(v)

        # 采样点云（这里把范围扩到 [-0.9995,0.9995]，保持与你原代码一致）
        mv = v * (2 * 0.9995)
        mv = np.clip(mv, -0.9995, 0.9995)
        pc_normal = sample_surface_points(mv, f, self.pc_num)

        if self.augment and self.split == "train":
            jitter = np.random.normal(0.0, 0.002, size=pc_normal[:, :3].shape).astype(np.float16)
            pc_normal[:, :3] = np.clip(pc_normal[:, :3] + jitter, -0.9995, 0.9995)

        # pad seq
        if len(seq) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds max_seq_len {self.max_seq_len}: {npz_path}"
            )
        L = min(len(seq), self.max_seq_len)

        #id映射
        seq = self.id_mapping(seq)
        #print(f'seq_new: {seq}')
        pad_seq = np.full((self.max_seq_len,), self.pad_id, dtype=np.int64)
        pad_seq[:L] = seq[:L]
        #print(f'pad_seq: {pad_seq}')
        return {
            "pc_normal": pc_normal,     # (pc_num,6) float16
            "sequence": pad_seq,        # (max_seq_len,) int64
            "seq_len": L,
            "model_name": item.get("uid", os.path.basename(npz_path).split(".")[0]),
        }