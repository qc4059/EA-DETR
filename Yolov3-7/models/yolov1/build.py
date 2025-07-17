#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .loss import build_criterion
from .yolov1 import YOLOv1


# 构建 YOLOv1 网络
def build_yolov1(args, cfg, device, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))
    
    print('==============================')
    print('当前模型配置、Model Configuration: \n', cfg)
    
    # -------------- 构建YOLOv1 --------------
    #              模型配置参数    设备                    640                     20
    model = YOLOv1(cfg = cfg, device = device, img_size = args.img_size, num_classes = num_classes, 
        conf_thresh = args.conf_thresh, #0.005  置信度阈值 nms_thresh = args.nms_thresh,
        trainable = trainable, #True    这个 trainable 参数 表示的是 模型 是否用于训练
        deploy = deploy  # False
        )

    # -------------- 初始化YOLOv1的pred层参数 --------------  这段代码用于初始化模型中不同预测头的权重和偏置项，确保模型从合理的状态开始训练
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
    # obj pred
    b = model.obj_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # cls pred
    b = model.cls_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # reg pred
    b = model.reg_pred.bias.view(-1, )
    b.data.fill_(1.0)
    model.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    w = model.reg_pred.weight
    w.data.fill_(0.)
    model.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    # -------------- 构建用于计算标签分配和计算损失的Criterion类 --------------
    criterion = None
    if trainable:
        # build criterion for training
        #                        模型配置参数 设备   20
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion
