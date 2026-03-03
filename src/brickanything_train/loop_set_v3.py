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

class Dataset():
    """
    读取 split/train.json or split/test.json
    每条样本包含 npz_path,npz 内含 vertices/faces/seq
    """
    def __init__(self,
                 split,          # e.g. /.../split/train.json
                 pc_num=8192,
                 max_seq_len=int(1600*9*0.7),
                 pad_id=-1,
                 renormalize=False,
                 clean_mesh=True):
        self.pc_num = pc_num
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)
        self.renormalize = renormalize
        self.clean_mesh = clean_mesh
        split_json = os.path.join('/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/dataset/shapenet_20_v1/split',f'{split}.json')
        self.eval_func = evaluate_cond_gpt
        with open(split_json, "r") as f:
            self.items = json.load(f)  # list[dict], each has npz_path

    def __len__(self):
        return len(self.items)

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

        # 采样点云（这里把范围扩到 [-0.9995,0.9995]，保持与你原代码一致）
        mv = v * (2 * 0.9995)
        mv = np.clip(mv, -0.9995, 0.9995)
        pc_normal = sample_surface_points(mv, f, self.pc_num)

        # pad seq
        L = min(len(seq), self.max_seq_len)
        assert L%5==0
        pad_seq = np.full((self.max_seq_len,), self.pad_id, dtype=np.int64)
        pad_seq[:L] = seq[:L]

        return {
            "pc_normal": pc_normal,     # (pc_num,6) float16
            "sequence": pad_seq,        # (max_seq_len,) int64
            "seq_len": L,
            "model_name": item.get("uid", os.path.basename(npz_path).split(".")[0]),
        }