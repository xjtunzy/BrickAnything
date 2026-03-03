import os
import json
import time
import torch
import numpy as np
from collections import defaultdict
from brickanything_train.misc import SmoothedValue

from plyfile import PlyData, PlyElement
import trimesh


def to_jsonable(x):
    # tensor -> python number / list
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return x.item() if x.numel() == 1 else x.tolist()
    # numpy scalar 等也可按需扩展
    return x

def calc_chamfer_loss(vertices_gt, vertices_recon):
    dist1 = torch.cdist(vertices_gt, vertices_recon, p=2).min(dim=1)[0]
    dist2 = torch.cdist(vertices_recon, vertices_gt, p=2).min(dim=1)[0]
    chamfer_loss = dist1.mean() + dist2.mean()
    return chamfer_loss

def write_gt(vertices, triangles, save_path ):
    face_mask = triangles[:, 0] != -1
    triangles = triangles[face_mask].cpu()
    vertice_mask = ~(vertices == -1).all(dim=1)
    gt_mesh = vertices[vertice_mask].cpu()

    scene_mesh = trimesh.Trimesh(vertices=gt_mesh, faces=triangles, force="mesh", merge_primitives=True)
    scene_mesh.merge_vertices()
    scene_mesh.update_faces(scene_mesh.nondegenerate_faces())
    scene_mesh.update_faces(scene_mesh.unique_faces())
    scene_mesh.remove_unreferenced_vertices()
    scene_mesh.fix_normals()

    write_mesh_with_color(scene_mesh, save_path)

def write_mesh_with_color(mesh, save_path):
    num_faces = len(mesh.faces)
    brown_color = np.array([255, 165, 0, 255], dtype=np.uint8)
    face_colors = np.tile(brown_color, (num_faces, 1))
    mesh.visual.face_colors = face_colors
    mesh.export(save_path)

@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    dataset_loader,
    accelerator,
    logger,
    curr_train_iter=-1,
    test_only = False,
):
    do_generate = False
    num_batches = len(dataset_loader)
    logger.info(f"Start evaluating on {num_batches} batches, data samples: {len(dataset_loader.dataset)}")
    time_delta = SmoothedValue(window_size=10)
    before_eval_time = time.time()
    model.eval()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""
    storage_dir = os.path.join(args.checkpoint_dir, "eval_logs")
    # 只让主进程建目录，避免多进程竞争
    if accelerator.is_main_process:
        os.makedirs(storage_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    loss_gather = defaultdict(list)
    for curr_iter, batch_data_label in enumerate(dataset_loader):
        #print("here2")
        curr_time = time.time()
        if "vertices" in batch_data_label or 1:
            loss_outputs = model(batch_data_label)
            for key, value in loss_outputs.items():
                if 'loss' in key.lower():
                    gathered_value = accelerator.gather(value)
                    loss_gather["val_"+key].append(gathered_value.mean().item())
        time_delta.update(time.time() - curr_time)
        mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        # logger.info(
        #     f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; " +
        #     f"Evaluating on iter: {curr_train_iter}; "
        #     f"Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
        # )
    loss_avg = {
        key: torch.tensor(loss_list, dtype=torch.float32).mean().item() \
            for key, loss_list in loss_gather.items()
    }
    print(f"loss_avg: {loss_avg}")
    if accelerator.is_main_process:
        loss_avg_json = {k: to_jsonable(v) for k, v in loss_avg.items()}
        out_path = os.path.join(storage_dir, f"loss_avg_{curr_train_iter}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(loss_avg_json, f, ensure_ascii=False, indent=2)
        logger.info(f"saved: {out_path}")
    accelerator.wait_for_everyone()
    return {},loss_avg