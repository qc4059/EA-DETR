"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)

    #     #        训练模型、损失函数、训练数据、优化器、设别、轮次、梯度裁剪、多少次打印一次日志、指数移动平均、自动混合精度
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    # 训练模式
    model.train()
    criterion.train()
    
    # 训练日志
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    # 训练过程，遍历数据集
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        # 将数据都移动到GPU上
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # AMP 自动混合精度训练
        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets) # 前向传播
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets)  # 损失计算

            # 计算总损失、反向传播
            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # 更新优化器、清空梯度
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 如果没有启用AMP
        else:
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema EMA更新
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        # 检查loss是否异常
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 记录日志
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # 汇总所有进程的日志
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 返回训练统计信息
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    iou_types = postprocessors.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # with torch.autocast(device_type=str(device)):
        #     outputs = model(samples)

        outputs = model(samples)

        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict
        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
        results = postprocessors(outputs, orig_target_sizes)
        # results = postprocessors(outputs, targets)

        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # for res in results:
        #     res['labels'] += 1  # 0-based → 1-based

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name
        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    
    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
            
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


# import numpy as np
# from collections import defaultdict

# def calculate_fixed_confidence_metrics(coco_eval, cat_ids, confidence_thresh=0.5):
#     """固定置信度阈值下的精确率/召回率统计"""
#     per_cat_stats = {cat_id: {'TP': 0, 'FP': 0, 'FN': 0} for cat_id in cat_ids}
    
#     # 仅处理指定置信度以上的检测
#     for eval_img in coco_eval.evalImgs:
#         if eval_img is None or eval_img['category_id'] not in cat_ids:
#             continue
            
#         cat_id = eval_img['category_id']
#         gt_ids = eval_img['gtIds']
#         dt_ids = eval_img['dtIds']
#         dt_scores = eval_img['dtScores']
#         matches = eval_img['dtMatches'][0]  # IoU=0.5
        
#         # 创建匹配的GT ID集合
#         matched_gt_ids = set()
#         for m in matches:
#             if m > 0:  # 有效的匹配（非背景）
#                 matched_gt_ids.add(m)
        
#         # 统计 TP/FP
#         for dt_idx, (dt_id, score) in enumerate(zip(dt_ids, dt_scores)):
#             if score < confidence_thresh:
#                 continue
                
#             if matches[dt_idx] > 0:  # 正确匹配
#                 per_cat_stats[cat_id]['TP'] += 1
#             else:  # 错误检测
#                 per_cat_stats[cat_id]['FP'] += 1
        
#         # 统计 FN（未被匹配的GT）- 修复错误
#         per_cat_stats[cat_id]['FN'] += len(gt_ids)
    
#     # 计算指标
#     results = {}
#     for cat_id, stats in per_cat_stats.items():
#         tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#         recall = tp / (fn) if (fn) > 0 else 0.0
#         results[cat_id] = {'precision': precision, 'recall': recall}
    
#     return results

# def calculate_per_category_metrics_full(coco_eval, cat_ids):    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     """
#     包含所有指标的完整实现
#     """
#     # 1. 计算精确率和召回率
#     precision_recall = calculate_fixed_confidence_metrics(coco_eval, cat_ids)
    
#     # 2. 计算AP50和AP50-95
#     results = {}
#     iou_thres = coco_eval.params.iouThrs
#     aind = [i for i, aRng in enumerate(coco_eval.params.areaRngLbl) if aRng == "all"]
#     mind = [i for i, maxDets in enumerate(coco_eval.params.maxDets) if maxDets == 100]
    
#     # 创建类别ID到索引的映射
#     cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(coco_eval.params.catIds)}
    
#     for cat_id in cat_ids:
#         if cat_id not in cat_id_to_idx:
#             continue
            
#         idx = cat_id_to_idx[cat_id]
#         metrics = {
#             'precision': precision_recall[cat_id]['precision'],
#             'recall': precision_recall[cat_id]['recall']
#         }
        
#         # 获取0.5 IoU索引
#         iou_50_idx = np.where(iou_thres == 0.5)[0][0] if iou_thres is not None else 0
        
#         # 计算AP50（平均精度）
#         precisions_iou50 = coco_eval.eval['precision'][iou_50_idx, :, idx, aind, mind].squeeze()
#         valid_precisions = precisions_iou50[precisions_iou50 > -1]
#         metrics['AP50'] = np.mean(valid_precisions) if valid_precisions.size > 0 else 0.0
        
#         # 计算AP50-95
#         ap_scores = []
#         for t in range(len(iou_thres)):
#             precisions_t = coco_eval.eval['precision'][t, :, idx, aind, mind].squeeze()
#             valid_pt = precisions_t[precisions_t > -1]
#             ap_score = np.mean(valid_pt) if valid_pt.size > 0 else 0.0
#             ap_scores.append(ap_score)
#         metrics['AP50-95'] = np.mean(ap_scores) if ap_scores else 0.0
        
#         results[cat_id] = metrics
    
#     return results

# @torch.no_grad()
# def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
#     model.eval()
#     criterion.eval()

#     metric_logger = MetricLogger(delimiter="  ")
#     header = 'Test:'
#     iou_types = postprocessors.iou_types
#     coco_evaluator = CocoEvaluator(base_ds, iou_types)

#     panoptic_evaluator = None

#     for samples, targets in metric_logger.log_every(data_loader, 10, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#         outputs = model(samples)

#         orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        
#         results = postprocessors(outputs, orig_target_sizes)

#         res = {target['image_id'].item(): output for target, output in zip(targets, results)}
#         if coco_evaluator is not None:
#             coco_evaluator.update(res)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)

#     if coco_evaluator is not None:
#         coco_evaluator.synchronize_between_processes()
#     if panoptic_evaluator is not None:
#         panoptic_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     if coco_evaluator is not None:
#         coco_evaluator.accumulate()
#         coco_evaluator.summarize()
        
#         # 计算并打印每类别指标
#         for iou_type in iou_types:
#             if iou_type == 'bbox':
#                 # 获取类别信息
#                 cat_ids = base_ds.getCatIds()  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#                 cats = base_ds.loadCats(cat_ids)   # [{'id': 0, 'name': 'pedestrian', 'supercategory': 'mark'}, {'id': 1, 'name': 'people', 'supercategory': 'mark'}, {'id': 2, 'name': 'bicycle', 'supercategory': 'mark'}, {'id': 3, 'name': 'car', 'supercategory': 'mark'}, {'id': 4, 'name': 'van', 'supercategory': 'mark'}, {'id': 5, 'name': 'truck', 'supercategory': 'mark'}, {'id': 6, 'name': 'tricycle', 'supercategory': 'mark'}, {'id': 7, 'name': 'awning-tricycle', 'supercategory': 'mark'}, {'id': 8, 'name': 'bus', 'supercategory': 'mark'}, {'id': 9, 'name': 'motor', 'supercategory': 'mark'}]
#                 cat_id_to_name = {cat['id']: cat['name'] for cat in cats} # {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}
                
#                 # 获取评估器
#                 coco_eval = coco_evaluator.coco_eval['bbox']
#                 results = calculate_per_category_metrics_full(coco_eval, cat_ids)
                
#                 # 打印评估表格
#                 print("\nPer-category metrics:")
#                 print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'AP50':>10} {'AP50-95':>10}")
#                 print("-" * 60)
                
#                 for cat_id, metrics in results.items():
#                     cat_name = cat_id_to_name.get(cat_id, f"Unknown({cat_id})")
#                     print(f"{cat_name:<20} {metrics['precision']*100:10.2f} "
#                           f"{metrics['recall']*100:10.2f} "
#                           f"{metrics['AP50']*100:10.2f} "
#                           f"{metrics['AP50-95']*100:10.2f}")
                
#                 # 计算"All"类别的平均值
#                 all_precision = np.mean([m['precision'] for m in results.values()])
#                 all_recall = np.mean([m['recall'] for m in results.values()])
#                 all_AP50 = np.mean([m['AP50'] for m in results.values()])
#                 all_AP50_95 = np.mean([m['AP50-95'] for m in results.values()])
                
#                 print("-" * 60)
#                 print(f"{'All':<20} {all_precision*100:10.2f} {all_recall*100:10.2f} "
#                       f"{all_AP50 * 100:10.2f} {all_AP50_95 * 100:10.2f}")
                
#                 # 添加全局AP指标（来自COCO官方计算）
#                 print("\nGlobal COCO metrics:")
#                 print(f"{'AP@[0.5:0.95]':<20} {coco_eval.stats[0]*100:10.2f}")
#                 print(f"{'AP50':<20} {coco_eval.stats[1]*100:10.2f}")
#                 print(f"{'AP75':<20} {coco_eval.stats[2]*100:10.2f}")
#                 print(f"{'AP small':<20} {coco_eval.stats[3]*100:10.2f}")
#                 print(f"{'AP medium':<20} {coco_eval.stats[4]*100:10.2f}")
#                 print(f"{'AP large':<20} {coco_eval.stats[5]*100:10.2f}")
    
#     stats = {}
#     if coco_evaluator is not None:
#         if 'bbox' in iou_types:
#             stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
#         if 'segm' in iou_types:
#             stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

#     return stats, coco_evaluator



