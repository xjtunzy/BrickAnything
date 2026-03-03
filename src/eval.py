import os, argparse

import datetime
from brickanything_train.engine import do_train 
from brickanything_train.models.single_gpt import SingleGPT  
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging
from tqdm import tqdm
import importlib
from accelerate.utils import DistributedDataParallelKwargs
from mesh2brick.brick_structure import Brick
from seq2brick import brick2ldr
from brickanything_train.render_bricks import render_bricks
import torch

def make_args_parser():
    parser = argparse.ArgumentParser("BrickAnything", add_help=False)

    parser.add_argument("--input_pc_num", default=8192, type=int) # 输入点云点数
    parser.add_argument("--max_vertices", default=800, type=int) # mesh 顶点上限（通常会影响 token 序列长度）

    parser.add_argument("--warm_lr_epochs", default=1, type=int)
    parser.add_argument("--num_beams", default=1, type=int)
    parser.add_argument("--max_seq_ratio", default=0.70, type=float)  # 序列长度比例（常用于裁剪/采样策略）

    ##### Model Setups #####
    parser.add_argument(
        '--pretrained_tokenizer_weight',
        default=None,
        type=str,
        help="The weight for pre-trained vqvae"
    )

    parser.add_argument('--llm', default="facebook/opt-125m", type=str, help="The LLM backend")
    parser.add_argument("--gen_n_max_bricks", default=1600, type=int, help="max number of triangles")  
    ##### Training #####
    parser.add_argument("--eval_every_iteration", default=1000, type=int)
    parser.add_argument("--save_every", default=250, type=int)
    parser.add_argument("--generate_every_data", default=1, type=int)

    ##### Testing #####
    parser.add_argument(
        "--clip_gradient", default=1., type=float,
        help="Max L2 norm of the gradient"
    )
    parser.add_argument("--pad_id", default=-1, type=int, help="padding id")
    parser.add_argument("--dataset", default='loop_set_v2', help="dataset list split by ','")

    parser.add_argument("--n_discrete_size", default=20, type=int, help="discretized 3D space")
    parser.add_argument("--data_n_max_bricks", default=1600, type=int, help="max number of triangles")
    parser.add_argument("--n_max_bricks", default=1600, type=int, help="max number of triangles")
    parser.add_argument("--n_min_bricks", default=40, type=int, help="max number of triangles")

    parser.add_argument("--shift_scale", default=0.1, type=float)  #设置平移量
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    
    parser.add_argument('--data_dir', default="/mnt/nas/yanfeng/data/n_project/BrickAnything", type=str, help="data path")
    parser.add_argument('--mesh_dir', default="/mnt/nas/yanfeng/data/n_project/ShapeNet", type=str, help="data path")
    parser.add_argument('--categorys',default=['camera'], type=list, help="data categorys" )

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--base_lr", default=1e-4, type=float)
    parser.add_argument("--final_lr", default=6e-5, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    

    parser.add_argument("--no_aug", default=False, action="store_true")#开启数据增强
    parser.add_argument("--checkpoint_dir", default="default", type=str)#指定保存 checkpoint（模型权重、optimizer 状态等）的目录名/路径
    parser.add_argument("--log_every", default=10, type=int)#控制多少次 iteration/step 打印一次日志（loss、lr、速度、显存等）
    parser.add_argument("--test_only", default=False, action="store_true")#变 True，脚本通常会 跳过训练，只跑测试/评估/推理，例如加载 checkpoint，然后在 test set 上评估指标或生成结果。
    parser.add_argument("--generate_every_iteration", default=6, type=int)#控制训练期间每隔多少次 iteration 做一次“生成（generate）”

    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--start_eval_after", default=-1, type=int)  #从第几步开始做评估
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)
    parser.add_argument(
        "--criterion", default=None, type=str,
        help='metrics for saving the best model'
    )#用哪个指标作为“最好”的判断依据

    parser.add_argument('--pretrained_weights', default=None, type=str)#含义：指定一个 checkpoint 路径，用来在训练开始前把模型权重加载进来
    parser.add_argument('--eval_dataset', default=None, type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    base_dir = "/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/gpt_output/training_trial_13_00-31-17"
    ckpt_path = "/mnt/data/yanfeng/n_project/brick_gen/BrickAnything/gpt_output/training_trial_13_00-31-17/checkpoint_7000.pth"
    args = make_args_parser()
    out_dir = os.path.join(base_dir,'visual',f'{args.eval_dataset}')
    os.makedirs(out_dir,exist_ok=True)
    dataset_module = importlib.import_module(f'brickanything_train.{args.dataset}')
    train_dataset = dataset_module.Dataset(args, split_set="train")
    test_dataset = dataset_module.Dataset(args, split_set="test")
    train_uids = train_dataset.data
    test_uids = test_dataset.data
    intersection_list = list(set(train_uids).intersection(set(test_uids)))
    assert len(intersection_list)==0  ,f"intersection_list not zero: {len(intersection_list)}"
    #print("intersection_list:", len(intersection_list))
    print(f"len of train_dataset: {len(train_uids)}\nlen of test_dataset: {len(test_dataset)}")
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batchsize_per_gpu,
        drop_last = True,
        shuffle = True,
    )
    #print(f"train dataloader len :{len(dataloaders['train'])}")
    dataloaders['test'] = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchsize_per_gpu,
        drop_last = True,
        shuffle = False,
    )
    print("Instantiate model.......")
    model = SingleGPT(args)
    model = model.to(torch.float16)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
    model = model.to("cuda")

    #遍历所有测试集
    if args.eval_dataset == "test":
        batch_data = dataloaders['test']
    else:
        batch_data = dataloaders['train']
    for batch in tqdm(batch_data):
        gen_bricks,data_names = model.generate(batch)
        for bricks,id in zip(gen_bricks,data_names):
            tmp_path = os.path.join(out_dir,id)
            os.makedirs(tmp_path,exist_ok=True)
            ldr_path = os.path.join(tmp_path,'brick_eval.ldr')
            render_path = os.path.join(tmp_path,'eval.png')
            brick2ldr(bricks,ldr_path)
            render_bricks(ldr_path,render_path)






