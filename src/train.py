import os, argparse

import datetime
from brickanything_train.engine import do_train 
from brickanything_train.models.single_gpt import SingleGPT  
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging
import importlib
from accelerate.utils import DistributedDataParallelKwargs
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
    parser.add_argument("--dataset", default='loop_set_v3', help="dataset list split by ','")

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

    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__file__)

    args = make_args_parser()

    cur_time = datetime.datetime.now().strftime("%d_%H-%M-%S")
    wandb_name = args.checkpoint_dir + "_" +cur_time
    args.checkpoint_dir = os.path.join("gpt_output", wandb_name)
    print("checkpoint_dir:", args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.precision,
        log_with="wandb",#指定日志后端用 Weights & Biases (wandb)
        project_dir=args.checkpoint_dir,
        kwargs_handlers=[kwargs]
    )
    if "default" not in args.checkpoint_dir:
        accelerator.init_trackers(
            project_name="GPT",
            config=vars(args),
            init_kwargs={"wandb": {"name": wandb_name}}
        )

    set_seed(args.seed, device_specific=True)

    dataset_module = importlib.import_module(f'brickanything_train.{args.dataset}')

    train_dataset = dataset_module.Dataset(split="train")
    test_dataset = dataset_module.Dataset(split="test")
    train_uids = [x["uid"] for x in train_dataset.items]
    test_uids  = [x["uid"] for x in test_dataset.items]
    intersection = set(train_uids) & set(test_uids)
    assert len(intersection) == 0, f"intersection not zero: {len(intersection)}"

    print(f"len of train_dataset: {len(train_uids)}\nlen of test_dataset: {len(test_uids)}")

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
    #batch = next(iter(dataloaders['test'])) 

    print("Instantiate model.......")
    model = SingleGPT(args)
    print("Beginning train.......")
    do_train(
        args,
        model,
        dataloaders,
        logger,
        accelerator,
    )