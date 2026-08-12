import os, argparse, json, math
import torch
import time
import trimesh
import numpy as np
import datetime
import yaml
import logging
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
import transformers
from pathlib import Path

from seq2brick import brick2ldr
from mesh_to_pc import process_mesh_to_pc
from BrickAnything.models.brickanything_tree_mode import BrickAnything
from metrics.evaluate import evaluate_func
from metrics.voxel_evaluate import evaluate_func as evaluate_voxel

# Optional Blender rendering (not required for generation / LDR export).
try:
    from brickanything_train.render_bricks import render_bricks as _render_bricks
except Exception:
    _render_bricks = None

# inference mode only support batch_size = 1
REPO_ROOT = Path(__file__).resolve().parent.parent


def save_voxel_png(voxels, out_path, title=None, dpi=140):
    """Render occupancy voxel grid to PNG using matplotlib only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occ = np.asarray(voxels).astype(bool)
    if occ.ndim != 3:
        raise ValueError(f"voxels must be 3D, got shape={occ.shape}")

    fig = plt.figure(figsize=(6.0, 5.6))
    ax = fig.add_subplot(111, projection="3d")

    if occ.any():
        colors = np.zeros(occ.shape + (4,), dtype=np.float64)
        colors[occ] = (0.12, 0.47, 0.92, 0.92)
        ax.voxels(occ, facecolors=colors, edgecolor="k", linewidth=0.10)
    else:
        ax.text2D(0.5, 0.5, "empty voxel", transform=ax.transAxes, ha="center")

    nx, ny, nz = occ.shape
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=24, azim=40)
    ax.set_title(title or "Voxel Occupancy")

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)

class Dataset:
    def __init__(self, input_type, input_list, mc=False, mc_level = 7, seed=None):
        super().__init__()
        self.data = []
        if input_type == 'pc_normal':
            for input_path in input_list:
                # load npy
                cur_data = np.load(input_path)
                # sample 8192
                assert cur_data.shape[0] >= 8192, "input pc_normal should have at least 8192 points"
                idx = np.random.choice(cur_data.shape[0], 8192, replace=False)
                cur_data = cur_data[idx]
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})

        elif input_type == 'mesh':
            mesh_list = []
            mesh_paths = []
            for input_path in input_list:
                # load mesh files (obj/ply/stl/glb/gltf, etc.)
                cur_data = trimesh.load(input_path, process=False, force="mesh")
                if isinstance(cur_data, trimesh.Scene):
                    # 合并 scene 中所有几何体为一个 mesh
                    cur_data = trimesh.util.concatenate(
                        [g for g in cur_data.geometry.values()]
                    )
                mesh_list.append(cur_data)
                mesh_paths.append(input_path)
            if mc:
                print("First Marching Cubes and then sample point cloud, need several minutes...")
            pc_list, _ = process_mesh_to_pc(mesh_list, marching_cubes=mc, mc_level=mc_level, seed=seed)
            for input_path, cur_data in zip(mesh_paths, pc_list):   
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0], 'mesh_path': input_path})
        print(f"dataset total data samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}
        raw_pc_normal = self.data[idx]['pc_normal']
        data_dict['sampled_pc_normal'] = raw_pc_normal.astype(np.float32)
        data_dict['pc_normal'] = raw_pc_normal
        data_dict['mesh_path'] = self.data[idx].get('mesh_path', None)
        # normalize pc coor
        pc_coor = data_dict['pc_normal'][:, :3]
        normals = data_dict['pc_normal'][:, 3:]
        bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
        pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
        assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), "normals should be unit vectors, something wrong"
        data_dict['pc_normal'] = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
        data_dict['uid'] = self.data[idx]['uid']
        return data_dict

def get_args():
    parser = argparse.ArgumentParser("BrickAnything", add_help=False)

    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "model_config" / "opt_tree_mode.yaml"),
    )

    parser.add_argument('--input_dir', default=None, type=str)
    parser.add_argument('--input_path', default=None, type=str)

    parser.add_argument('--out_dir', default="inference_out", type=str)

    parser.add_argument(
        '--input_type',
        choices=['mesh','pc_normal'],
        default='pc',
        help="Type of the asset to process (default: pc)"
    )

    parser.add_argument("--batchsize_per_gpu", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--trseed", default=26, type=int)
    parser.add_argument("--mc", default=False, action="store_true")
    parser.add_argument("--mc_level", default=7, type=int)
    parser.add_argument(
        "--save_sampled_pc",
        default=False,
        action="store_true",
        help="Save sampled point cloud (.npy) in output folder.",
    )
    parser.add_argument(
        "--do_render",
        default=False,
        action="store_true",
        help="Render LDR to PNG if Blender/ImportLDraw tooling is available locally.",
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    cur_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    checkpoint_dir = os.path.join(str(REPO_ROOT / "result"), args.out_dir, cur_time)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with open(args.config,'r',encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    #set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(checkpoint_dir, "inference.log"), encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting inference at {cur_time}")
    logger.info(f"Config: {cfg}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Batch size per GPU: {args.batchsize_per_gpu}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"MC: {args.mc}")
    logger.info(f"MC level: {args.mc_level}")
    logger.info(f"Input type: {args.input_type}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"TRSeed: {args.trseed}")
    logger.info(f"Save sampled point cloud: {args.save_sampled_pc}")
    
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        project_dir=checkpoint_dir,
        kwargs_handlers=[kwargs]
    )
    transformers.set_seed(args.trseed)
    
    print('loading pth file .....')
    model = BrickAnything(cfg['model'])
    model = model.to(torch.float16)
    ckpt_path = cfg['ckpt_path']
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    print('create dataset .....')
    # create dataset
    if args.input_dir is not None:
        input_list = sorted(os.listdir(args.input_dir))
        # only ply, obj or npy
        if args.input_type == 'pc_normal':
            input_list = [os.path.join(args.input_dir, x) for x in input_list if x.endswith('.npy')]
        else:
            mesh_exts = ('.ply', '.obj', '.stl', '.glb', '.gltf')
            input_list = [os.path.join(args.input_dir, x) for x in input_list if x.lower().endswith(mesh_exts)]
        set_seed(args.seed)
        dataset = Dataset(args.input_type, input_list, args.mc, args.mc_level)
    elif args.input_path is not None:
        set_seed(args.seed)
        dataset = Dataset(args.input_type, [args.input_path], args.mc, args.mc_level, args.seed)
    else:
        raise ValueError("input_dir or input_path must be provided.")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize_per_gpu, 
        drop_last = False,
        shuffle = False,
    )

    if accelerator.state.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    dataloader, model = accelerator.prepare(dataloader, model)
    begin_time = time.time()
    logger.info("Generation Start!!!")
    with accelerator.autocast():
        for curr_iter, batch_data_label in enumerate(dataloader):
            curr_time = time.time()
            outputs = model(batch_data_label['pc_normal'])
            end_time1 = time.time()
            #print(f"outputs: {outputs}")
            batch_size = len(outputs["gen_brick"])
            for batch_id in range(batch_size):
                recon_bricks = outputs["gen_brick"][batch_id]
                seq = outputs["seq_batch"][batch_id].tolist()
                ldr_save_path = os.path.join(checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_gen.ldr')
                photo_save_path = os.path.join(checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_gen.png')
                voxels_save_path = os.path.join(checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_voxels.png')
                brick_occ_save_path = os.path.join(checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_brick_occ.png')
                json_save_path = os.path.join(checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_gen.json')
                sampled_pc_save_path = os.path.join(checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_sampled_pc.npy')
                logger.info(f"Gen time: {end_time1 - curr_time}")
                if args.save_sampled_pc:
                    sampled_pc = batch_data_label["sampled_pc_normal"][batch_id].detach().cpu().numpy()
                    np.save(sampled_pc_save_path, sampled_pc)
                    logger.info(f"Saved sampled point cloud: {sampled_pc_save_path}")
                brick2ldr(recon_bricks,ldr_save_path)
                chamfer_loss = evaluate_func(recon_bricks,batch_data_label['pc_normal'][batch_id])
                if 'mesh_path' in batch_data_label and batch_data_label['mesh_path'][batch_id] is not None:
                    logger.info(f'mesh evaluate ...')
                    voxel_iou, voxel_cd, brick_occ,voxels = evaluate_voxel(recon_bricks,batch_data_label['mesh_path'][batch_id], mode='mesh')
                else:
                    voxel_iou, voxel_cd, brick_occ, voxels = evaluate_voxel(recon_bricks,batch_data_label['pc_normal'][batch_id], mode='pc')
                if chamfer_loss is None:
                    chamfer_loss_json = None
                else:
                    _cl = float(chamfer_loss.item())
                    chamfer_loss_json = None if math.isnan(_cl) or math.isinf(_cl) else _cl
                json2save = {
                    'gen_time':end_time1 - curr_time,
                    #'regeneration_count': outputs["regeneration_count"],
                    "chamfer_loss": chamfer_loss_json,
                    "voxel_iou": float(voxel_iou),
                    "voxel_cd": float(voxel_cd),
                    "Seq": seq,
                }
                with open(json_save_path, "w", encoding="utf-8") as f:
                    json.dump(json2save, f, ensure_ascii=False)
                logger.info(f'regeneration count: {outputs["regeneration_count"]}')
                if chamfer_loss is None:
                    logger.warning('chamfer loss: unavailable (evaluate_func returned None)')
                elif math.isnan(float(chamfer_loss.item())):
                    logger.warning('chamfer loss: nan (empty brick voxel mesh)')
                else:
                    logger.info(f'chamfer loss: {chamfer_loss.item():.6f}')
                logger.info(f'voxel iou: {voxel_iou}')
                logger.info(f'voxel cd: {voxel_cd}')
                save_voxel_png(voxels, voxels_save_path, title=f'{batch_data_label["uid"][batch_id]} voxel')
                save_voxel_png(brick_occ, brick_occ_save_path, title=f'{batch_data_label["uid"][batch_id]} brick_occ')
                if args.do_render:
                    if _render_bricks is None:
                        logger.warning(
                            "Skipping render: local render tooling unavailable."
                        )
                    else:
                        _render_bricks(ldr_save_path, photo_save_path)
                logger.info(f"{ldr_save_path} Over!!")
    end_time = time.time()
    logger.info(f"Total time: {end_time - begin_time}")
    