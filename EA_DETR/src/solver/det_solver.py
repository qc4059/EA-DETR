'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from copy import deepcopy

class DetSolver(BaseSolver):    # 当子类没有 def __init__（）进行初始化时，会自动调用父类的 初始化
    
    def fit(self, ):
        print("Start training")
        self.train()   # 这里调用父类函数，却什么也不返回的原因是在父类中 进行初始化组件、加载训练数据集，到时候，可以直接self调用父类的

        args = self.cfg  # YAMLConfig 实例
        # 计算模型的总参数量
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)  
        print('number of params:', n_parameters)

        # ========== 新增GFLOPs计算 ========== ↓
        from thop import profile
        model_copy = deepcopy(self.model)
        model_copy.eval()
        # 生成输入样本（根据实际输入尺寸调整）
        input_sample = torch.randn(1, 3, 640, 640).to(self.device)  # 假设输入为800x800
        mask = torch.ones((1, 640, 640), dtype=torch.bool).to(self.device)  # 适配DETR等需要mask的模型

        with torch.no_grad():
            flops, _ = profile(model_copy, 
                            inputs=(input_sample, mask),  # 根据实际模型输入调整
                            verbose=False)
        gflops = flops / 1e9
        print(f'GFLOPs: {gflops:.2f}')
        # ========== 新增GFLOPs计算 ========== ↑

        # 获取coco API 用于目标检测、评估模型
        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }

        # 记录最佳模型
        best_stat = {'epoch': -1, }  

        # 训练循环
        start_time = time.time()                 # 最大训练轮次：72
        for epoch in range(self.last_epoch + 1, args.epoches):  
            # 如果需要多GPU训练
            # if dist.is_dist_available_and_initialized():
            #     self.train_dataloader.sampler.set_epoch(epoch)
            
            # 训练模型1轮
            train_stats = train_one_epoch( # 训练模型、损失函数、训练数据、优化器、设别、轮次、梯度裁剪、多少次打印一次日志、指数移动平均、自动混合精度
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            # 调整学习率
            self.lr_scheduler.step()
            
            # 保存模型checkpoint （模型检查点）
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:  # args.checkpoint_step  表示 每多少轮 保存一下模型
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    # self.state_dict()是本地的一个函数
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)# dist.save_on_master表示：只在主进程保存 checkpoint（防止多 GPU 训练时重复写入）

            #评估模型
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO 记录最佳指标
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat)

            # 记录日志
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        # 记录训练时长
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
        
# #--------------------------------------------------------------------------------------------------------------------- 
#         module = self.ema.module if self.ema else self.model  # 确保使用EMA模型
#         module.eval()
#         input_sample = torch.randn(1, 3, 640, 640).to(self.device)  # 输入尺寸需匹配模型

#         # 预热（避免冷启动误差）
#         with torch.no_grad():
#             for _ in range(10):
#                 _ = module(input_sample)

#         # 正式测量
#         torch.cuda.synchronize()  # 确保CUDA操作同步（GPU设备需要）
#         start_time = time.time()
#         for _ in range(100):  # 重复多次取平均
#             with torch.no_grad():
#                 _ = module(input_sample)
#         torch.cuda.synchronize()  # GPU需要同步
#         total_time = time.time() - start_time
#         fps = 100 / total_time
#         print(f"FPS (bs=1): {fps:.2f}")
#----------------------------------------------------------------------------------------------------------------------

        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
