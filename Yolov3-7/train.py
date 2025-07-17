from __future__ import division

import argparse
from copy import deepcopy

# ----------------- Torch Components -----------------
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------- Extra Components -----------------
from utils import distributed_utils
from utils.misc import compute_flops

# ----------------- Config Components -----------------
from config import build_dataset_config, build_model_config, build_trans_config

# ----------------- Model Components -----------------
from models import build_model

# ----------------- Train Components -----------------
from engine import build_trainer


def parse_args():
    # 这个命令行参数 可以定义参数名称、类型、默认值、帮助信息
    parser = argparse.ArgumentParser(description='YOLO-Tutorial')
    # Basic
     #action属性中 store_true 用于标记 是否使用前面的 参数，如果使用就标记为 true
    parser.add_argument('--cuda', action='store_true', default=True, help='是否使用CUDA')
    # 这个 “--” 标志，表示是可选参数，可选参数通常有默认值，如果没有在命令行指定他们，就使用默认值
    parser.add_argument('-size', '--img_size', default=640, type=int, help='指定图片输入大小')

    parser.add_argument('--num_workers', default=4, type=int, help='数据加载时使用的线程数')

    parser.add_argument('--tfboard', action='store_true', default=False,help='是否使用 tensorboard')

    parser.add_argument('--save_folder', default='weights/', type=str, help='保存权重文件的路径')

    parser.add_argument('--eval_first', action='store_true', default=False,help='训练前先进行评估')

    parser.add_argument('--fp16', dest="fp16", action="store_true", default=True  ,help="是否使用混合精度训练")

    parser.add_argument('--vis_tgt', action="store_true", default=False,help="可视化训练数据")

    parser.add_argument('--vis_aux_loss', action="store_true", default=False,help="可视化辅助损失")
    
    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='batch size on all the GPUs.')

    # Epoch
    parser.add_argument('--max_epoch', default=400, type=int, 
                        help='max epoch.')
    parser.add_argument('--wp_epoch', default=1, type=int, 
                        help='warmup epoch.')
    parser.add_argument('--eval_epoch', default=1, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--no_aug_epoch', default=50, type=int, 
                        help='cancel strong augmentation.')

    # Model
    parser.add_argument('-m', '--model', default='yolov7', type=str, help='模型类型')
    
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument('-r', '--resume', default="E:/Learning/深度学习/YoLo系列/YOLO书籍-全部源代码/YOLO_Tutorial/weights/VisDrone/yolov7/yolov7_epoch_290.pth", type=str)
    # parser.add_argument('-r', '--resume', default=None, type=str)
    
    # Dataset
    parser.add_argument('--root', default='E:/Learning/Data', help='data root')
    parser.add_argument('-d', '--dataset', default='VisDrone', help='coco, voc, widerface, VisDrone')
    parser.add_argument('--load_cache', action='store_true', default=False, help='load data into memory.')
    
    # Train trick
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='Multi scale')
    parser.add_argument('--ema', action='store_true', default=True,
                        help='Model EMA')
    parser.add_argument('--min_box_size', default=8.0, type=float,
                        help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--grad_accumulate', default=1, type=int,
                        help='gradient accumulation')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # 如果args.distributed为True，则初始化PyTorch框架提供的分布式训练（DDP）
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))
    # 对于单卡，world_size = 1； 对于多卡，world_size = 卡的数量
    world_size = distributed_utils.get_world_size()
    print('World size: {}'.format(world_size))

    # 如果args.cuda为True，则使用GPU来训练，否则使用CPU来训练（强烈不推荐）
    if args.cuda:
        print('use GPU to train')
        device = torch.device("cuda")
    else:
        print('use CPU to train')
        device = torch.device("cpu")

    # 构建训练所用到的 Dataset & Model & Transform相关的config变量
    data_cfg = build_dataset_config(args)   #数据集参数
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # 构建YOLO模型
    #                            通用参数  模型参数   设备            20
    model, criterion = build_model(args, model_cfg, device, data_cfg['num_classes'], True)

    # 如果指定了args.resume，则表明我们要从之前停止的迭代节点继续训练模型
    if distributed_utils.is_main_process and args.resume is not None:
        print('keep training: ', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

    # 将模型切换至train模式
    model = model.to(device).train()

    # 标记单卡模式的model，方便我们做一些其他的处理，省去了DDP模式下的model.module的判断
    model_without_ddp = model

    # 如果args.distributed为True，且args.sybn也为True，表明我们使用SyncBatchNorm层，同步多卡之间的BN统计量
    # 只有在DDP模式下才会考虑SyncBatchNorm层
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 计算模型的参数量和FLOPs
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        compute_flops(model=model_copy,
                      img_size=args.img_size,
                      device=device)
        del model_copy
    if args.distributed:
        dist.barrier()

    # 构建训练所需的Trainer类
    #                    通用参数   数据集参数  模型参数   数据预处理参数 设备    V1模型        计算损失的类     1
    trainer = build_trainer(args, data_cfg, model_cfg, trans_cfg, device, model_without_ddp, criterion, world_size)

    # --------------------------------- Train: Start ---------------------------------
    ## 如果args.eval_first为True，则在训练开始前，先测试模型的性能
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        model_eval = model_without_ddp
        trainer.eval(model_eval)

    ## 开始训练我们的模型
    trainer.train(model)
    # --------------------------------- Train: End ---------------------------------

    # 训练完毕后，清空占用的GPU显存
    del trainer
    if args.cuda:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train()






# from __future__ import division

# import argparse
# from copy import deepcopy

# # ----------------- Torch Components -----------------
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# # ----------------- Extra Components -----------------
# from utils import distributed_utils
# from utils.misc import compute_flops

# # ----------------- Config Components -----------------
# from config import build_dataset_config, build_model_config, build_trans_config

# # ----------------- Model Components -----------------
# from models import build_model

# # ----------------- Train Components -----------------
# from engine import build_trainer


# def parse_args():
#     # 这个命令行参数 可以定义参数名称、类型、默认值、帮助信息
#     parser = argparse.ArgumentParser(description='YOLO-Tutorial')
#     # Basic
#      #action属性中 store_true 用于标记 是否使用前面的 参数，如果使用就标记为 true
#     parser.add_argument('--cuda', action='store_true', default=True, help='是否使用CUDA')
#     # 这个 “--” 标志，表示是可选参数，可选参数通常有默认值，如果没有在命令行指定他们，就使用默认值
#     parser.add_argument('-size', '--img_size', default=448, type=int, help='指定图片输入大小')

#     parser.add_argument('--num_workers', default=4, type=int, help='数据加载时使用的线程数')

#     parser.add_argument('--tfboard', action='store_true', default=False,help='是否使用 tensorboard')

#     parser.add_argument('--save_folder', default='weights/voc/yo_st', type=str, help='保存权重文件的路径')

#     parser.add_argument('--eval_first', action='store_true', default=False,help='训练前先进行评估')

#     parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,help="是否使用混合精度训练")

#     parser.add_argument('--vis_tgt', action="store_true", default=False,help="可视化训练数据")

#     parser.add_argument('--vis_aux_loss', action="store_true", default=False,help="可视化辅助损失")
    
#     # Batchsize
#     parser.add_argument('-bs', '--batch_size', default=16, type=int, help='GPU上处理的每个批次的数量')

#     # Epoch
#     parser.add_argument('--max_epoch', default=100, type=int, help='训练的最大轮数')
#     parser.add_argument('--wp_epoch', default=1, type=int, help='预热轮数')
#     parser.add_argument('--eval_epoch', default=10, type=int, help='每训练多少轮进行一次验证集上的评估，默认是每 10 轮评估一次')
#     parser.add_argument('--no_aug_epoch', default=20, type=int, help='表示从第 no_aug_epoch 轮开始取消强数据增强，默认是从第 20 轮开始取消。')

#     # Model
#     parser.add_argument('-m', '--model', default='yolov1', type=str, help='指定使用的模型类型，默认是 yolov1。可以在命令行中更改为其他模型,如 yolov3 等')
#     parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float, help='设置目标检测模型的置信度阈值，只有置信度大于此阈值的检测结果才会被保留，默认值为 0.005')
#     parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float, help='设置非极大值抑制(NMS)阈值,用于过滤重复框。默认值为 0.6')
#     parser.add_argument('--topk', default=1000, type=int, help='在评估时，选择前 topk 个候选框进行输出，默认是 1000')
#     parser.add_argument('-p', '--pretrained', default=None, type=str, help='指定预训练权重的路径，如果提供，将加载预训练的模型权重进行 fine-tuning')
#     parser.add_argument('-r', '--resume', default=None, type=str, help='指定训练中断后要从哪个权重文件恢复训练')

#     # Dataset
#     parser.add_argument('--root', default='E:/Learning/Data', help='指定数据集的根目录路径')

#     parser.add_argument('-d', '--dataset', default='voc', help='指定使用的数据集类型，默认使用 coco 数据集，可以更改为 voc, widerface 或 crowdhuman 等')

#     parser.add_argument('--load_cache', action='store_true', default=False, help='如果设置了该参数，则会将数据集加载到内存中，以提高数据加载速度')
    
#     # Train trick   
#     parser.add_argument('-ms', '--multi_scale', action='store_true', default=False, help='是否使用多尺度训练。多尺度训练能够提高模型的泛化能力，默认不开启')
#     parser.add_argument('--ema', action='store_true', default=False, help='是否使用模型的指数滑动平均(EMA)。EMA 有助于提高模型的稳定性和性能')
#     parser.add_argument('--min_box_size', default=8.0, type=float, help='目标框的最小大小，用于过滤掉太小的框，默认值是 8.0')
#     parser.add_argument('--mosaic', default=None, type=float, help='是否启用 Mosaic 数据增强。Mosaic 是一种将多个图像拼接成一个新的图像的方法，通常用于数据增强')
#     parser.add_argument('--mixup', default=None, type=float, help='是否启用 Mixup 数据增强。Mixup 是一种数据增强方法，混合两张图像及其标签')
#     parser.add_argument('--grad_accumulate', default=1, type=int, help='梯度累积的步数。在使用较小批量训练时，可能会使用梯度累积来模拟更大的批量')

#     # DDP train
#     parser.add_argument('-dist', '--distributed', action='store_true', default=False, help='是否启用分布式训练')
#     parser.add_argument('--dist_url', default='env://',  help='分布式训练的 URL 地址。通常为 env:// 表示使用环境变量')
#     parser.add_argument('--world_size', default=1, type=int, help='分布式训练时的进程数，表示训练使用多少个 GPU')
#     parser.add_argument('--sybn', action='store_true', default=False,  help='是否启用同步批量归一化(SyncBatchNorm),通常在多卡训练时使用')

#     return parser.parse_args()


# def train():
#     args = parse_args()
#     print("Setting Arguments.. : ", args)
#     print("----------------------------------------------------------")

#     # 如果args.distributed为True，则初始化PyTorch框架提供的分布式训练（DDP）
#     if args.distributed:
#         distributed_utils.init_distributed_mode(args)
#         print("git:\n  {}\n".format(distributed_utils.get_sha()))
#     # 对于单卡，world_size = 1； 对于多卡，world_size = 卡的数量
#     world_size = distributed_utils.get_world_size()
#     print('World size: {}'.format(world_size))    # 这个World size表示参与分布式训练的 进程总数

#     # 如果args.cuda为True，则使用GPU来训练，否则使用CPU来训练（强烈不推荐）
#     if args.cuda:
#         print('use GPU to train')
#         device = torch.device("cuda")
#     else:
#         print('use CPU to train')
#         device = torch.device("cpu")

#     # 构建训练所用到的 Dataset & Model & Transform相关的config变量
#     data_cfg = build_dataset_config(args)   #data_cfg 表示的是 读取的数据集 的 基本信息
#     model_cfg = build_model_config(args)   # 这个是 针对 yolov1网络的 训练类型，上面那些parser参数是总体的，这个是针对特定的网络的 具体的训练配置
#     trans_cfg = build_trans_config(model_cfg['trans_type'])  # 这个表示的是 对数据预处理 的配置参数

#     # 构建YOLO模型
#     model, criterion = build_model(args, model_cfg, device, data_cfg['num_classes'], True)

#     # 如果指定了args.resume，则表明我们要从之前停止的迭代节点继续训练模型
#     if distributed_utils.is_main_process and args.resume is not None:
#         print('keep training: ', args.resume)
#         checkpoint = torch.load(args.resume, map_location='cpu')
#         # checkpoint state dict
#         checkpoint_state_dict = checkpoint.pop("model")
#         model.load_state_dict(checkpoint_state_dict)

#     # 将模型切换至train模式
#     model = model.to(device).train()

#     # 标记单卡模式的model，方便我们做一些其他的处理，省去了DDP模式下的model.module的判断
#     model_without_ddp = model

#     # 如果args.distributed为True，且args.sybn也为True，表明我们使用SyncBatchNorm层，同步多卡之间的BN统计量
#     # 只有在DDP模式下才会考虑SyncBatchNorm层
#     if args.sybn and args.distributed:
#         print('use SyncBatchNorm ...')
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#     if args.distributed:
#         model = DDP(model, device_ids=[args.gpu])
#         model_without_ddp = model.module

#     # 计算模型的参数量和FLOPs
#     if distributed_utils.is_main_process:
#         model_copy = deepcopy(model_without_ddp)
#         model_copy.trainable = False
#         model_copy.eval()
#         compute_flops(model=model_copy,
#                       img_size=args.img_size,
#                       device=device)
#         del model_copy
#     if args.distributed:
#         dist.barrier()   

#     # 构建训练所需的Trainer类
#     trainer = build_trainer(args, data_cfg, model_cfg, trans_cfg, device, model_without_ddp, criterion, world_size)

#     # --------------------------------- Train: Start ---------------------------------
#     ## 如果args.eval_first为True，则在训练开始前，先测试模型的性能
#     if args.eval_first and distributed_utils.is_main_process():
#         # to check whether the evaluator can work
#         model_eval = model_without_ddp
#         trainer.eval(model_eval)

#     ## 开始训练我们的模型
#     trainer.train(model)
#     # --------------------------------- Train: End ---------------------------------

#     # 训练完毕后，清空占用的GPU显存
#     del trainer
#     if args.cuda:
#         torch.cuda.empty_cache()


# if __name__ == '__main__':
#     train()