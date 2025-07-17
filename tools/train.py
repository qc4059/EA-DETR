"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # 因为当前train.py文件所在目录与src所在目录不一致，需要导入src所在的父目录
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    # dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(   # 这个函数负责解析YAML配置文件        这个cfg实例 整合了yaml文件中的所有配置参数、包括
        args.config,
        resume=args.resume,   # 这个参数 用于控制 断点续训   
        use_amp=args.amp,    # amp 是否用 混合精度训练    默认为False
        tuning=args.tuning  # 是否用于 超参数微调
    )
    #      cfg.yaml_cfg['task'] 值是 'detection'       # 此时TASKS[cfg.yaml_cfg['task']]得到是一个类，(cfg)作为初始化类的一个参数
    solver = TASKS[cfg.yaml_cfg['task']](cfg)  # solver是DetSolver的实例  
    
    if args.test_only:   # False 
        solver.val()  # 验证
    else:
        solver.fit()  # 训练


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',default="E:/论文写作/欣赏别人的论文/目标检测论文/RT-DETR/RT-DETR_V7-策略1+2/configs/rtdetr/rtdetr_r18vd_6x_coco.yml", type=str, )
    # parser.add_argument('--resume', '-r',default="E:/论文写作/欣赏别人的论文/目标检测论文/RT-DETR/RT-DETR_V7-策略1+2/output/rtdetr_r18vd_6x_coco/checkpoint0624.pth", type=str, )
    parser.add_argument('--resume', '-r',default="E:/论文写作/欣赏别人的论文/目标检测论文/RT-DETR/RT-DETR_V7-策略1+2/output/HIT-AUV/checkpoint0204.pth", type=str, )
    # parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=True,)
    parser.add_argument('--seed', type=int, default = 42 ,help='seed',)
    args = parser.parse_args()

    main(args)
