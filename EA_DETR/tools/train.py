"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) 
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

    cfg = YAMLConfig(   # 函数负责解析YAML配置文件   cfg实例 整合了yaml文件中的所有配置参数
        args.config,
        resume=args.resume,    
        use_amp=args.amp, 
        tuning=args.tuning  
    )
    solver = TASKS[cfg.yaml_cfg['task']](cfg)  
    
    if args.test_only:  
        solver.val()  
    else:
        solver.fit() 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=True,)
    parser.add_argument('--seed', type=int, default = 42 ,help='seed',)
    args = parser.parse_args()

    main(args)
