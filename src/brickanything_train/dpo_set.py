import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh


class DPODataset:
    def __init__(
        self,
        dpo_pairs_json: str,
        pc_dir: str,
        mesh_items_json: Optional[str] = None,
        pc_num: int = 8192,
        max_seq_len: int = int(1000 * 5),
        pad_id: int = -1,
        allow_mesh_fallback: bool = False,
    ):
        self.dpo_pairs_json = dpo_pairs_json
        self.pc_dir = pc_dir
        self.mesh_items_json = mesh_items_json
        self.pc_num = int(pc_num)
        self.max_seq_len = int(max_seq_len)
        self.pad_id = int(pad_id)
        self.allow_mesh_fallback = bool(allow_mesh_fallback)

        with open(dpo_pairs_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        self.items: List[Dict[str, Any]] = d["pairs"] if isinstance(d, dict) else d
        if not isinstance(self.items, list) or len(self.items) == 0:
            raise ValueError(f"Invalid DPO json: {dpo_pairs_json}")
        self.uid_to_item: Dict[str, Dict[str, Any]] = {}
        if self.mesh_items_json is not None:
            with open(self.mesh_items_json, "r", encoding="utf-8") as f:
                mesh_items = json.load(f)
            if not isinstance(mesh_items, list) or len(mesh_items) == 0:
                raise ValueError(f"Invalid mesh items json: {self.mesh_items_json}")
            for mesh_item in mesh_items:
                if "uid" not in mesh_item or "npz_path" not in mesh_item:
                    continue
                self.uid_to_item[str(mesh_item["uid"])] = mesh_item

    def __len__(self):
        return len(self.items)

    def _sample_surface_points(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            force="mesh",
            merge_primitives=True,
            process=False,
        )
        pts, face_idx = mesh.sample(self.pc_num, return_index=True)
        nrm = mesh.face_normals[face_idx]
        pts = np.clip(pts.astype(np.float16), -0.9995, 0.9995)
        nrm = nrm.astype(np.float16)
        return np.concatenate([pts, nrm], axis=-1)

    def _load_pc_npy(self, uid: str) -> np.ndarray:
        pc_path = os.path.join(self.pc_dir, f"{uid}_pc.npy")
        if not os.path.isfile(pc_path):
            raise FileNotFoundError(pc_path)
        pc_normal = np.load(pc_path, allow_pickle=False)
        if pc_normal.dtype != np.float16:
            pc_normal = pc_normal.astype(np.float16)
        if pc_normal.ndim != 2:
            raise ValueError(f"Invalid point cloud shape for {uid}: {pc_normal.shape}")
        # infer_dpo --save_xyz_only => (N,3); model expects xyz+normal (N,6)
        if pc_normal.shape[1] == 3:
            n = pc_normal.shape[0]
            zeros = np.zeros((n, 3), dtype=np.float16)
            pc_normal = np.concatenate([pc_normal, zeros], axis=-1)
        elif pc_normal.shape[1] != 6:
            raise ValueError(
                f"Expected point cloud (N,3) or (N,6) for {uid}, got {pc_normal.shape}"
            )
        return pc_normal

    def _pad_seq(self, seq: List[int]) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.int64)
        if seq.shape[0] > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        if seq.shape[0] % 5 != 0:
            trim_len = (seq.shape[0] // 5) * 5
            seq = seq[:trim_len]
        out = np.full((self.max_seq_len,), self.pad_id, dtype=np.int64)
        out[: seq.shape[0]] = seq
        return out

    def __getitem__(self, idx: int):
        item = self.items[idx]
        uid = str(item["uid"])
        mesh_item = self.uid_to_item.get(uid)
        npz_path = item.get("npz_path") if isinstance(item, dict) else None
        if npz_path is None and mesh_item is not None:
            npz_path = mesh_item.get("npz_path")

        pc_path = os.path.join(self.pc_dir, f"{uid}_pc.npy")
        if os.path.isfile(pc_path):
            pc_normal = self._load_pc_npy(uid)
        elif self.allow_mesh_fallback and npz_path is not None:
            d = np.load(npz_path, allow_pickle=False)
            v = d["vertices"].astype(np.float32)
            f = d["faces"].astype(np.int64)
            mv = np.clip(v * (2 * 0.9995), -0.9995, 0.9995)
            pc_normal = self._sample_surface_points(mv, f)
        else:
            raise FileNotFoundError(
                f"Point cloud not found: {pc_path}. "
                f"Run infer_dpo with --save_pc to generate vail_pc, or pass "
                f"--allow_mesh_fallback 1 with mesh npz paths."
            )

        chosen_seq = self._pad_seq(item["chosen_seq"])
        rejected_seq = self._pad_seq(item["rejected_seq"])
        out: Dict[str, Any] = {
            "uid": uid,
            "pc_normal": pc_normal,
            "chosen_sequence": chosen_seq,
            "rejected_sequence": rejected_seq,
            "iou_gap": np.float32(item.get("iou_gap", 1.0)),
            "chosen_len": int((chosen_seq != self.pad_id).sum()),
            "rejected_len": int((rejected_seq != self.pad_id).sum()),
        }
        gt_raw = item.get("gt_seq")
        if gt_raw is not None:
            out["gt_sequence"] = self._pad_seq(gt_raw)
        return out
