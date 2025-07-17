import numpy as np
import torch


class Yolov3Matcher(object):
    #                      80           3        9个锚框尺寸      0.5
    def __init__(self, num_classes, num_anchors, anchor_size, iou_thresh):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.iou_thresh = iou_thresh
        self.anchor_boxes = np.array(       # anchor_boxes 变为9行4列的 列表
            [[0., 0., anchor[0], anchor[1]]
            for anchor in anchor_size]
            )  # [KA, 4]
    #                      先验框      目标实际框
    def compute_iou(self, anchor_boxes, gt_box):
        """
            anchor_boxes: (numpy.array) -> [KA, 4] (cx, cy, bw, bh).
            gt_box:       (numpy.array) -> [1, 4] (cx, cy, bw, bh).
        """
        # anchors: [KA, 4]
        anchors = np.zeros_like(anchor_boxes)   # 这里的 np.zeros_like 做的事情就是 复制别人的 维度 将其 全部变为0
        anchors[..., :2] = anchor_boxes[..., :2] - anchor_boxes[..., 2:] * 0.5  # x1y1
        anchors[..., 2:] = anchor_boxes[..., :2] + anchor_boxes[..., 2:] * 0.5  # x2y2
        anchors_area = anchor_boxes[..., 2] * anchor_boxes[..., 3]

        # gt_box: [1, 4] -> [KA, 4]        gt_box是经过处理后的，只计算出 目标 真实边界框的 相对自己的 宽高
        gt_box = np.array(gt_box).reshape(-1, 4)  # shape(1,4) [[0, 0, 127, 233]]
        gt_box = np.repeat(gt_box, anchors.shape[0], axis=0)  # anchors.shape[0] = 9    gt_box.shape: (9,4)
        gt_box_ = np.zeros_like(gt_box)
        gt_box_[..., :2] = gt_box[..., :2] - gt_box[..., 2:] * 0.5  # x1y1   # 由于先验框有9个 所以同一个目标框与9个先验框计算时，将目标框也复制9个，方便一一对应 计算
        gt_box_[..., 2:] = gt_box[..., :2] + gt_box[..., 2:] * 0.5  # x2y2
        gt_box_area = np.prod(gt_box[..., 2:] - gt_box[..., :2], axis=1)

        # intersection   # 下面实现的是 np.minimum 算的是 右下角坐标，比较的是 先验框与 目标框 之间 谁w（宽）最小，max 计算左上角 比较谁w最大
        inter_w = np.minimum(anchors[:, 2], gt_box_[:, 2]) - \
                  np.maximum(anchors[:, 0], gt_box_[:, 0])
        # # 下面实现的是 np.minimum 算的是 y下坐标，比较的是 先验框与 目标框 之间 谁h（高）最小，max 计算y上 比较谁h最大
        inter_h = np.minimum(anchors[:, 3], gt_box_[:, 3]) - \
                  np.maximum(anchors[:, 1], gt_box_[:, 1])
        inter_area = inter_w * inter_h   # 这里计算得到的是 先验框与目标框的 相交部分 （计算的方式，是 先验框与目标框是 同一个中心点，都以网格中心(0,0)为中心点）
        
        # union
        union_area = anchors_area + gt_box_area - inter_area  # 先验框面积+目标框面积-相交的面积 = 不相交面积

        # iou
        iou = inter_area / union_area
        iou = np.clip(iou, a_min=1e-10, a_max=1.0)
        
        return iou

    @torch.no_grad()
    #     [[80,80],[40,40],[20,20]]  #[8,16,32]  16
    def __call__(self, fmp_sizes, fpn_strides, targets):
        """
        输入参数的解释:
            fmp_sizes:   (List[List[int, int], ...]) 多尺度特征图的尺寸
            fpn_strides: (List[Int, ...]) 多尺度特征图的输出步长
            targets:     (List[Dict]) 为List类型，包含一批数据的标签，每一个数据标签为Dict类型，其主要的数据结构为：
                             dict{'boxes':  (torch.Tensor) [N, 4], 一张图像中的N个目标边界框坐标
                                  'labels': (torch.Tensor) [N,], 一张图像中的N个目标类别标签
                                  ...}

targets = {   # target 是每张图片中，所有目标物的边界框信息 和 类别索引 和 图片像素大小信息
                "boxes": anno[:, :4],  # 所有目标物 边界框信息
                "labels": anno[:, 4],  # 所有目标物 类别信息
                "orig_size": [height, width] # 该张 图片的大小
            }
        """
        assert len(fmp_sizes) == len(fpn_strides)
        # 准备后续处理会用到的变量
        bs = len(targets)   # 16  targets 数据类型是 list[Dict1,Dict2,...,Dict16] 每个Dict包括那张图片中 所有目标物的 信息

        gt_objectness = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 1])   #[[16,80,80,3,1],[16,40,40,3,1],[16,20,20,3,1]] 初始化每一个层级的置信度标签
            for (fmp_h, fmp_w) in fmp_sizes
            ]
        gt_classes = [          #[[16,80,80,3,20],[],[]]
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, self.num_classes]) 
            for (fmp_h, fmp_w) in fmp_sizes
            ]
        gt_bboxes = [    #[[16,80,80,3,4],[],[]]
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 4]) 
            for (fmp_h, fmp_w) in fmp_sizes
            ]

        # 第一层for循环遍历每一张图像的标签
        for batch_index in range(bs):
            targets_per_image = targets[batch_index]  # Dict1
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()  # (N,1) N 表示 目标物的个数  1表示 每一个目标物的真实类别
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()  # （N,4）同样，4列表示，每一行是该目标物的 边界框信息(边界框 上下左右 4个坐标点)

            # 第二层for循环遍历该张图像的每一个目标的标签
            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                # 获得该目标的边界框坐标
                x1, y1, x2, y2 = gt_box.tolist()  # 目标物的真实边界框大小信息

                # 计算目标框的中心点坐标和宽高
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1
                gt_box = [0, 0, bw, bh]

                # 检查该目标边界框是否有效
                if bw < 1. or bh < 1.:
                    continue    

                # 计算目标框和所有先验框之间的交并比
                #                            先验框     目标物真实框
                iou = self.compute_iou(self.anchor_boxes, gt_box)
                # 返回的iou 类型、维度：
                iou_mask = (iou > self.iou_thresh)

                # 根据IoU结果，确定正样本的标记
                label_assignment_results = []
                if iou_mask.sum() == 0:
                    # 情况1，如果的先验框与目标框的iou值都较低，
                    # 此时，我们将iou最高的先验框标记为正样本
                    iou_ind = np.argmax(iou)

                    # 先验框所对应的特征金字塔的尺度(level)的标记
                    level = iou_ind // self.num_anchors              # pyramid level
                    # 先验框的索引
                    anchor_idx = iou_ind - level * self.num_anchors  # anchor index

                    # 对应尺度的输出步长
                    stride = fpn_strides[level]

                    # 计算网格坐标
                    xc_s = xc / stride
                    yc_s = yc / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    label_assignment_results.append([grid_x, grid_y, level, anchor_idx])
                else:
                    # 情况2&3，至少有一个先验框和目标框的IoU大于给定的阈值
                    for iou_ind, iou_m in enumerate(iou_mask):
                        if iou_m:
                            # 先验框所对应的特征金字塔的尺度(level)的标记
                            level = iou_ind // self.num_anchors              # pyramid level
                            # 先验框的索引
                            anchor_idx = iou_ind - level * self.num_anchors  # anchor index

                            # 对应尺度的输出步长
                            stride = fpn_strides[level]

                            # 计算网格坐标
                            xc_s = xc / stride
                            yc_s = yc / stride
                            grid_x = int(xc_s)
                            grid_y = int(yc_s)

                            label_assignment_results.append([grid_x, grid_y, level, anchor_idx])

                # 依据上述的先验框的标记，开始标记正样本的位置
                for result in label_assignment_results:
                    grid_x, grid_y, level, anchor_idx = result
                    fmp_h, fmp_w = fmp_sizes[level]

                    if grid_x < fmp_w and grid_y < fmp_h:
                        # 标记objectness标签，即此处的网格有物体，对应一个正样本
                        gt_objectness[level][batch_index, grid_y, grid_x, anchor_idx] = 1.0  # 因为gt_objectness是二维数组，有三个元素，分别表示三个层，level就指示了在哪一层
                        # 标记正样本处的类别标签，采用one-hot格式
                        cls_ont_hot = torch.zeros(self.num_classes)
                        cls_ont_hot[int(gt_label)] = 1.0
                        gt_classes[level][batch_index, grid_y, grid_x, anchor_idx] = cls_ont_hot
                        # 标记正样本处的bbox标签
                        gt_bboxes[level][batch_index, grid_y, grid_x, anchor_idx] = torch.as_tensor([x1, y1, x2, y2])

        # 首先，将每个尺度的标签数据的shape从 [B, H, W, A， C] 的形式reshape成 [B, M, C] ，其中M = HWA，以便后续的处理
        # 然后，将所有尺度的预测拼接在一起，方便后续的损失计算
        gt_objectness = torch.cat([gt.view(bs, -1, 1) for gt in gt_objectness], dim=1).float()  # 这里拼接的是 三个尺度上的置信度 即（80x80x3; 40x40x3; 20x20x3) 总和为(16,25200,1)
        print(gt_objectness.size())
        gt_classes = torch.cat([gt.view(bs, -1, self.num_classes) for gt in gt_classes], dim=1).float()  #(16,25200,20)
        gt_bboxes = torch.cat([gt.view(bs, -1, 4) for gt in gt_bboxes], dim=1).float() # (16,25200,4)

        return gt_objectness, gt_classes, gt_bboxes






