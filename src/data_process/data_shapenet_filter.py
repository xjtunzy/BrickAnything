import os
import json
import time
import argparse
import random
import logging
import numpy as np
import trimesh
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -------------------------
# 1) 扫描 seq.json，筛 Stable==1，并直接返回 seq_arr（避免二次读文件）
# -------------------------
def collect_stable_samples(seq_root, stable_key="Stable", stable_value=1, max_workers=32):
    seq_paths = []
    for syn_ent in os.scandir(seq_root):
        if not syn_ent.is_dir():
            continue
        synset = syn_ent.name
        for mid_ent in os.scandir(syn_ent.path):
            if not mid_ent.is_dir():
                continue
            p = os.path.join(mid_ent.path, "seq.json")
            if os.path.isfile(p):
                seq_paths.append((synset, mid_ent.name, p))

    def _check_one(item):
        synset, model_id, p = item
        try:
            with open(p, "r") as f:
                seq = json.load(f)
            if int(seq.get(stable_key, 0)) == int(stable_value):
                seq_arr = np.asarray(seq["Seq"], dtype=np.int32)
                return (synset, model_id, p, seq_arr)
        except Exception as e:
            return ("__ERR__", p, str(e), None)
        return None

    samples = []
    err_cnt = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_check_one, item) for item in seq_paths]
        for fu in tqdm(as_completed(futures), total=len(futures), desc="Scanning seq.json"):
            r = fu.result()
            if r is None:
                continue
            if r[0] == "__ERR__":
                err_cnt += 1
                if err_cnt <= 20:
                    logging.warning(f"Failed to read {r[1]}: {r[2]}")
                continue
            samples.append(r)

    logging.info(f"Stable samples: {len(samples)}/{len(seq_paths)} (errors={err_cnt})")
    return samples


# -------------------------
# 2) 在 mesh_root 中定位 mesh 文件
# -------------------------
def find_mesh_file(mesh_root, synset, model_id):
    base = os.path.join(mesh_root, synset, model_id, "models")
    if not os.path.isdir(base):
        return None
    mesh_path = os.path.join(base, "model_normalized.obj")
    return mesh_path if os.path.exists(mesh_path) else None


# -------------------------
# 3) mesh 清洗 + 归一化
# -------------------------
def clean_and_normalize(mesh: trimesh.Trimesh):
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)

    bounds = np.array([v.min(axis=0), v.max(axis=0)], dtype=np.float32)
    center = (bounds[0] + bounds[1]) / 2.0
    scale = float((bounds[1] - bounds[0]).max())
    if scale < 1e-12:
        raise ValueError("degenerate mesh scale too small")

    v = (v - center[None, :]) / scale
    v = np.clip(v, -0.5, 0.5)
    return v, f


# -------------------------
# 4) 单样本处理：保存 raw_npz/<uid>.npz
#   不保存 vertices_num / faces_num
# -------------------------
def process_one(args_tuple):
    """
    args_tuple = (synset, model_id, seq_json_path, seq_arr, mesh_root, out_raw_dir)
    """
    synset, model_id, seq_json_path, seq_arr, mesh_root, out_raw_dir = args_tuple
    uid = f"{synset}_{model_id}"
    out_path = os.path.join(out_raw_dir, uid + ".npz")
    tmp_path = os.path.join(out_raw_dir, uid + ".tmp")

    if os.path.exists(out_path):
        return 0

    try:
        fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return 0

    try:
        mesh_path = find_mesh_file(mesh_root, synset, model_id)
        if mesh_path is None:
            return 0

        mesh = trimesh.load(mesh_path, force="mesh")
        if not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
            return 0

        v_norm, f_clean = clean_and_normalize(mesh)

        np.savez(
            out_path,
            uid=uid,
            synset=synset,
            model_id=model_id,
            vertices=v_norm,
            faces=f_clean,
            seq=seq_arr,
            seq_len=np.int32(seq_arr.shape[0]),
        )
        return 1
    except Exception as e:
        logging.error(f"[{uid}] failed: {e}")
        return 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# -------------------------
# 5) 生成索引 split/train.json, split/test.json
#    test = 10%（可配置 test_ratio）
#    只存 uid/synset/model_id/npz_path
# -------------------------
def write_split_index(raw_dir, out_split_dir, test_ratio=0.1, seed=0):
    os.makedirs(out_split_dir, exist_ok=True)
    npz_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(".npz")])

    items = []
    for p in tqdm(npz_files, desc="Building index"):
        try:
            d = np.load(p, allow_pickle=True)
            items.append({
                "uid": str(d["uid"]),
                "synset": str(d["synset"]),
                "model_id": str(d["model_id"]),
                "npz_path": p,
            })
        except Exception as e:
            logging.warning(f"skip {p}: {e}")

    n = len(items)
    if n == 0:
        raise ValueError("No samples found in raw_npz")

    n_test = int(round(n * float(test_ratio)))
    n_test = max(1, n_test)  # 至少 1
    n_test = min(n - 1, n_test)  # 至少留 1 个给 train

    rng = np.random.default_rng(seed)
    test_idx = set(rng.choice(n, size=n_test, replace=False).tolist())
    test = [items[i] for i in range(n) if i in test_idx]
    train = [items[i] for i in range(n) if i not in test_idx]

    with open(os.path.join(out_split_dir, "train.json"), "w") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_split_dir, "test.json"), "w") as f:
        json.dump(test, f, ensure_ascii=False, indent=2)

    logging.info(f"Saved index: train={len(train)}, test={len(test)} (ratio={test_ratio}) -> {out_split_dir}")


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seq_root", default="/mnt/nas/yanfeng/data/n_project/shapenet_20_zyx", type=str)
    parser.add_argument("--mesh_root", default="/mnt/nas/yanfeng/data/n_project/ShapeNet", type=str)
    parser.add_argument("--out_dir", default="/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/dataset/shapenet_20_v2", type=str)

    parser.add_argument("--stable_key", default="Stable", type=str)
    parser.add_argument("--stable_value", default=1, type=int)

    parser.add_argument("--workers", default=cpu_count(), type=int)
    parser.add_argument("--seed", default=0, type=int)

    # seq.json 扫描并行度
    parser.add_argument("--seq_scan_workers", default=14, type=int)

    # test 占比
    parser.add_argument("--test_ratio", default=0.1, type=float)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_raw_dir = os.path.join(args.out_dir, "raw_npz")
    out_split_dir = os.path.join(args.out_dir, "split")
    os.makedirs(out_raw_dir, exist_ok=True)
    os.makedirs(out_split_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    samples = collect_stable_samples(
        args.seq_root,
        stable_key=args.stable_key,
        stable_value=args.stable_value,
        max_workers=args.seq_scan_workers,
    )

    tasks = [(synset, mid, seqp, seq_arr, args.mesh_root, out_raw_dir)
             for (synset, mid, seqp, seq_arr) in samples]

    logging.info(f"Processing {len(tasks)} samples with {args.workers} workers")
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        done = pool.map(process_one, tasks)
    logging.info(f"Done. saved={sum(done)} time={time.time()-t0:.1f}s")

    write_split_index(out_raw_dir, out_split_dir, test_ratio=args.test_ratio, seed=args.seed)


if __name__ == "__main__":
    main()