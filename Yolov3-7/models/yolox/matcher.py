# ---------------------------------------------------------------------
# Copyright (c) Megvii Inc. All rights reserved.
# ---------------------------------------------------------------------


import torch
import torch.nn.functional as F
from utils.box_ops import *


class SimOTA(object):
    """
        该代码参考了YOLOX官方项目的源码： https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
    """
     # 'matcher': {'center_sampling_radius': 2.5, 'topk_candicate': 10},
     #                      20                2.5                  10
    def __init__(self, num_classes, center_sampling_radius, topk_candidate ):
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.topk_candidate = topk_candidate


    @torch.no_grad()
    def __call__(self, 
                 fpn_strides,    # [8,16,32]
                 anchors,     
                 pred_obj,     # [8400,1]
                 pred_cls,     # [8400,20]
                 pred_box,     # [8400,4]
                 tgt_labels,   # 每张图片中的 所有目标物的 标签信息 （真实值）
                 tgt_bboxes):  # 每张图片中的 所有目标物的 边界框信息 （真实值）
        # [M,]
        # print(anchors[0])   
        # print(anchors[0].size())  # [6400,2]
        # print('----------------')
        # print(anchors[1])   
        # print(anchors[1].size())  # [1600,2]
        # print('----------------')
        # print(anchors[2])   
        # print(anchors[2].size())  # [400,2]
        strides_tensor = torch.cat([torch.ones_like(anchor_i[:, 0]) * stride_i    #  torch.ones_like 生成的全是1 。 乘上不同尺寸，就变成前6400个8 1600个16 400个32
                                for stride_i, anchor_i in zip(fpn_strides, anchors)], dim=-1)
        # print(strides_tensor.size())    # [8400,1]
        # print(strides_tensor)   #[前6400是下采样尺度8 中间1600是下采样尺度16 最后400是下采样尺度32]

        # List[F, M, 2] -> [M, 2]
        anchors = torch.cat(anchors, dim=0)    # [8400 2]
        num_anchor = anchors.shape[0]     # 8400   锚框的数量
        num_gt = len(tgt_labels)  #  一张图片中 目标物的 数量

        # ----------------------- Find inside points -----------------------
        #                           每张图片中的 所有目标物的 边界框信息 （真实值）  锚框[8400,2]  [8400,1]     8400     x
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info( tgt_bboxes, anchors, strides_tensor, num_anchor, num_gt)
        obj_preds = pred_obj[fg_mask].float()   # [Mp, 1]
        cls_preds = pred_cls[fg_mask].float()   # [Mp, C] # Mp 是指 保留 所有在目标框内的 锚框  这一次维度是[3225，20] 这个3225都是在目标框内的锚框个数
        box_preds = pred_box[fg_mask].float()   # [Mp, 4]  # Mp 俗称 对应着正样本的位置

        # ----------------------- Reg cost -----------------------
        pair_wise_ious, _ = box_iou(tgt_bboxes, box_preds)      # [N, Mp]   计算每一个目标框与所有Mp的预测框 iou N是指 每个真实框
        reg_cost = -torch.log(pair_wise_ious + 1e-8)            # [N, Mp]

        # ----------------------- Cls cost -----------------------
        with torch.cuda.amp.autocast(enabled=False):
            # [Mp, C]
            score_preds = torch.sqrt(obj_preds.sigmoid_()* cls_preds.sigmoid_())
            # [N, Mp, C]
            score_preds = score_preds.unsqueeze(0).repeat(num_gt, 1, 1)
            # prepare cls_target
            cls_targets = F.one_hot(tgt_labels.long(), self.num_classes).float()
            cls_targets = cls_targets.unsqueeze(1).repeat(1, score_preds.size(1), 1)
            # [N, Mp]
            cls_cost = F.binary_cross_entropy(score_preds, cls_targets, reduction="none").sum(-1)
        del score_preds

        #----------------------- Dynamic K-Matching -----------------------
        # a = ~is_in_boxes_and_center
        # print(a)
        cost_matrix = (
            cls_cost
            + 3.0 * reg_cost
            + 100000.0 * (~is_in_boxes_and_center)
        ) # [N, Mp]

        (
            assigned_labels,         # [num_fg,]
            assigned_ious,           # [num_fg,]
            assigned_indexs,         # [num_fg,]
        ) = self.dynamic_k_matching(
            cost_matrix,     # 最终的代价函数
            pair_wise_ious,  # 每一个目标框与所有Mp的预测框 iou
            tgt_labels,  # 每张图片中的 所有目标物的 标签信息
            num_gt,    # 一张图片中 目标物的 数量
            fg_mask   # Mp 在目标框内的 锚框
            )
        del cls_cost, cost_matrix, pair_wise_ious, reg_cost

        return fg_mask, assigned_labels, assigned_ious, assigned_indexs


    def get_in_boxes_info( self,
        gt_bboxes,   # [N, 4]  每张图片中的 所有目标物的 边界框信息 （真实值） 
        anchors,     # [M, 2]   锚框[8400,2] 
        strides,     # [M,]   [8400,1]
        num_anchors, # M     8400
        num_gt,      # N   
        ):
        # anchor center
        x_centers = anchors[:, 0]   # 所有不同尺度的锚框的 中心点的 x坐标
        y_centers = anchors[:, 1]   # 所有不同尺度的锚框的 中心点的 y坐标
        # print(type(x_centers))
        # print(x_centers[:10])
        # [M,] -> [1, M] -> [N, M]
        x_centers = x_centers.unsqueeze(0).repeat(num_gt, 1)    # N 行重复的 锚框x坐标
        y_centers = y_centers.unsqueeze(0).repeat(num_gt, 1)    # N 行重复的 锚框y坐标

        # [N,] -> [N, 1] -> [N, M]
        gt_bboxes_l = gt_bboxes[:, 0].unsqueeze(1).repeat(1, num_anchors) # x1   M 列重复的 x1坐标（真实边框值）
        gt_bboxes_t = gt_bboxes[:, 1].unsqueeze(1).repeat(1, num_anchors) # y1   M 列重复的 y1坐标（真实边框值）
        gt_bboxes_r = gt_bboxes[:, 2].unsqueeze(1).repeat(1, num_anchors) # x2   M 列重复的 x2坐标（真实边框值）
        gt_bboxes_b = gt_bboxes[:, 3].unsqueeze(1).repeat(1, num_anchors) # y2   M 列重复的 y2坐标（真实边框值）

        # 下面这部分 是为了 计算 锚框是否 在目标框内部，计算了锚框的四个边界与目标框的四个边界之间的差值，如果所有四个差值都为正，表示锚框在目标框内，画个图就很明显
        b_l = x_centers - gt_bboxes_l    # 这个得到的维度是 (N,M)  且每一行是 每个目标物 与 所有锚框之间 进行的 横坐标差值（简单来说，将每一个目标物，分别与所有锚框坐标进行比较，为下面筛选锚框是否在目标框内做准备）
        b_r = gt_bboxes_r - x_centers
        b_t = y_centers - gt_bboxes_t
        b_b = gt_bboxes_b - y_centers     
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # (N,M,4) 
        # print(bbox_deltas.size())   # torch.Size([N, 8400, 4])   # 这里的4 指的是 目标框与锚框四个角的 坐标 的差值， 

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0 # [N,8400]  # 判断 如果所有四个差值都为正，表示锚框在目标框内
        # print(type(is_in_boxes))  # tensor类型
        # print(is_in_boxes.size())   # [N,8400]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0   # [8400,]  这行代码的作用是统计每个锚框是否有至少一个目标框将其视为正样本
        # print(is_in_boxes_all.size())   # [8400,]
        # print(is_in_boxes_all)
 

        # in fixed center   这里是规范目标框的中心邻域的
        center_radius = self.center_sampling_radius  # 值为 2.5
        # [N, 2]
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5   # 这个就是通过 目标框坐标 求目标框中心点坐标的  就像这里 xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
        # [1, M]
        center_radius_ = center_radius * strides.unsqueeze(0)

        gt_bboxes_l = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # x1
        gt_bboxes_t = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # y1
        gt_bboxes_r = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # x2
        gt_bboxes_b = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # y2
  
        # 下面是 计算锚框是否在中心邻域内
        c_l = x_centers - gt_bboxes_l
        c_r = gt_bboxes_r - x_centers
        c_t = y_centers - gt_bboxes_t
        c_b = gt_bboxes_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0   #center_deltas :(N,M,4)
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all    # 表示锚框在目标框内  is_in_boxes_anchor: [8400,1] 如果一个锚框满足两个条件之一——它要么在目标框内，要么在目标框的中心邻域内，那么该锚框就被视为“正样本”。

        is_in_boxes_and_center = (              #  表示锚框既在目标框内，又位于目标框的中心区域内  is_in_boxes_and_center: [N ,2833]  确保锚框同时满足“位于目标框内”和“位于目标框的中心邻域内”两个条件
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor] # 这里都是布尔值 is_in_center和is_in_boxes维度是（N,M）
        )                                                            # is_in_boxes_anchor（8400,）如果有一个为True 就直接对应is_in_center的一列 所有在某一次运行过程中 得到的维度是（N,2833）
        # 表示每个锚框是否位于目标框内    表示锚框既在目标框内，又位于目标框的中心区域内
        # print(is_in_boxes_anchor.size())
        # print(is_in_boxes_and_center.size())
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    
    def dynamic_k_matching(
        self, 
        cost,    # 最终的代价函数
        pair_wise_ious,  # 每一个目标框与所有Mp的预测框 iou
        gt_classes,    # 每张图片中的 所有目标物的 标签信息
        num_gt,    # 一张图片中 目标物的 数量
        fg_mask   # Mp 在目标框内的 锚框
        ):
        # Dynamic K
        # ---------------------------------------------------------------
        # (N, Mp)     cost: (N, Mp)
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        ious_in_boxes_matrix = pair_wise_ious   # (N, Mp)
        n_candidate_k = min(self.topk_candidate, ious_in_boxes_matrix.size(1))  # topk_candidata = 10  表示每个目标框最多选择的候选锚框数量
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)   # 对数据进行排序    # 找目标框下 预测框和目标框iou最大的前十个
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()   # tolist 将矩阵或者张量 转换为 列表
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(    #   cost : （N, Mp）  这里实现的是 选择代价最小的 k 个锚框，并返回其值（这里是“_”）和索引（pos_idx）。
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx   # 删除临时变量 释放内存

        anchor_matching_gt = matching_matrix.sum(0)  #  sum(0) 是对第一维进行求和， 这行代码统计每个锚框被分配给了多少个目标框

        # 下面这部分 实现了 锚框与目标框的最终匹配分配 （某些锚框可能被分配给多个目标框，通过代价函数选择代价最小的目标框解决冲突。）
        if (anchor_matching_gt > 1).sum() > 0: # 就是说 是一个目标框可以对应多个正样本锚框，但是一个锚框只能对应一个目标框
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        assigned_indexs = matching_matrix[:, fg_mask_inboxes].argmax(0)
        assigned_labels = gt_classes[assigned_indexs]

        assigned_ious = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return assigned_labels, assigned_ious, assigned_indexs
    