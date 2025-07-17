# import json
# import tempfile
# import torch
# from dataset.ourdataset import OurDataset
# from utils.box_ops import rescale_bboxes

# try:
#     from pycocotools.cocoeval import COCOeval
# except:
#     print("It seems that the COCOAPI is not installed.")


# class OurDatasetEvaluator():
#     """
#     COCO AP Evaluation class.
#     All the data in the val2017 dataset are processed \
#     and evaluated by COCO API.
#     """
#     def __init__(self, data_dir, device, image_set='val', transform=None):
#         """
#         Args:
#             data_dir (str): dataset root directory
#             img_size (int): image size after preprocess. images are resized \
#                 to squares whose shape is (img_size, img_size).
#             confthre (float):
#                 confidence threshold ranging from 0 to 1, \
#                 which is defined in the config file.
#             nmsthre (float):
#                 IoU threshold of non-max supression ranging from 0 to 1.
#         """
#         # ----------------- Basic parameters -----------------
#         self.image_set = image_set
#         self.transform = transform
#         self.device = device
#         # ----------------- Metrics -----------------
#         self.map = 0.
#         self.ap50_95 = 0.
#         self.ap50 = 0.
#         # ----------------- Dataset -----------------
#         self.dataset = OurDataset(data_dir=data_dir, image_set=image_set)


#     @torch.no_grad()
#     def evaluate(self, model, epoch=None, result_file="E:/Learning/深度学习/YoLo系列/YOLO书籍-全部源代码/YOLO_Tutorial/outputs/evaluation_results.txt"):
#         """
#         COCO average precision (AP) Evaluation. Iterate inference on the test dataset
#         and the results are evaluated by COCO API.
#         Args:
#             model : model object
#         Returns:
#             ap50_95 (float) : calculated COCO AP for IoU=50:95
#             ap50 (float) : calculated COCO AP for IoU=50
#         """
#         model.eval()
#         ids = []
#         data_dict = []
#         num_images = len(self.dataset)
#         print('total number of images: %d' % (num_images))

#         # start testing
#         for index in range(num_images): # all the data in val2017
#             if index % 500 == 0:
#                 print('[Eval: %d / %d]'%(index, num_images))

#             # load an image
#             img, id_ = self.dataset.pull_image(index)
#             orig_h, orig_w, _ = img.shape

#             # preprocess
#             x, _, deltas = self.transform(img)
#             x = x.unsqueeze(0).to(self.device) / 255.
            
#             id_ = int(id_)
#             ids.append(id_)
#             # inference
#             outputs = model(x)
#             bboxes, scores, cls_inds = outputs

#             # rescale bboxes
#             origin_img_size = [orig_h, orig_w]
#             cur_img_size = [*x.shape[-2:]]
#             bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

#             for i, box in enumerate(bboxes):
#                 x1 = float(box[0])
#                 y1 = float(box[1])
#                 x2 = float(box[2])
#                 y2 = float(box[3])
#                 label = self.dataset.class_ids[int(cls_inds[i])]
                
#                 bbox = [x1, y1, x2 - x1, y2 - y1]
#                 score = float(scores[i]) # object score * class score
#                 A = {"image_id": id_, "category_id": label, "bbox": bbox,
#                      "score": score} # COCO json format
#                 data_dict.append(A)

#         annType = ['segm', 'bbox', 'keypoints']

#         # Evaluate the Dt (detection) json comparing with the ground truth
#         if len(data_dict) > 0:
#             print('evaluating ......')
#             cocoGt = self.dataset.coco
#             _, tmp = tempfile.mkstemp()
#             json.dump(data_dict, open(tmp, 'w'))
#             cocoDt = cocoGt.loadRes(tmp)
#             cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
#             cocoEval.params.imgIds = ids
#             cocoEval.evaluate()
#             cocoEval.accumulate()
            
#             # 保存结果到文件（追加模式）
#             with open(result_file, 'a') as f:  # 改为 'a' 追加模式
#                 # 重定向标准输出到文件
#                 import sys
#                 original_stdout = sys.stdout
#                 sys.stdout = f
                
#                 # 添加epoch标识
#                 if epoch is not None:
#                     print(f"\n\n{'='*50}")
#                     print(f"Evaluation Results for Epoch {epoch}")
#                     print(f"{'='*50}")
                
#                 # 输出评估结果
#                 cocoEval.summarize()
#                 print('\nap50_95 : ', cocoEval.stats[0])
#                 print('ap50 : ', cocoEval.stats[1])
                
#                 # 恢复标准输出
#                 sys.stdout = original_stdout
            
#             # 在控制台打印结果
#             cocoEval.summarize()
#             ap50_95 = cocoEval.stats[0]
#             ap50 = cocoEval.stats[1]
#             print(f'ap50_95 : {ap50_95}')
#             print(f'ap50 : {ap50}')
            
#             if epoch is not None:
#                 print(f'Results for epoch {epoch} saved to {result_file}')
#             else:
#                 print(f'Results saved to {result_file}')
            
#             self.map = ap50_95
#             self.ap50_95 = ap50_95
#             self.ap50 = ap50
#             return ap50, ap50_95
#         else:
#             return 0, 0




import json
import tempfile
import torch
import numpy as np
from dataset.ourdataset import OurDataset
from utils.box_ops import rescale_bboxes

try:
    from pycocotools.cocoeval import COCOeval
except:
    print("It seems that the COCOAPI is not installed.")


def calculate_fixed_confidence_metrics(coco_eval, cat_ids, confidence_thresh=0.5):
    """固定置信度阈值下的精确率/召回率统计"""
    per_cat_stats = {cat_id: {'TP': 0, 'FP': 0, 'FN': 0} for cat_id in cat_ids}
    
    # 仅处理指定置信度以上的检测
    for eval_img in coco_eval.evalImgs:
        if eval_img is None or eval_img['category_id'] not in cat_ids:
            continue
            
        cat_id = eval_img['category_id']
        gt_ids = eval_img['gtIds']
        dt_ids = eval_img['dtIds']
        dt_scores = eval_img['dtScores']
        matches = eval_img['dtMatches'][0]  # IoU=0.5
        gt_ignore = eval_img['gtIgnore']
        
        # 创建匹配的GT ID集合
        matched_gt_ids = set()
        for m in matches:
            if m > 0:  # 有效的匹配（非背景）
                matched_gt_ids.add(m)
        
        # 统计 TP/FP
        for dt_idx, (dt_id, score) in enumerate(zip(dt_ids, dt_scores)):
            if score < confidence_thresh:
                continue
                
            if matches[dt_idx] > 0:  # 正确匹配
                per_cat_stats[cat_id]['TP'] += 1
            else:  # 错误检测
                per_cat_stats[cat_id]['FP'] += 1
        
        # 统计 FN（未被匹配的GT）- 修复错误
        # 统计未被匹配且未被忽略的GT
        # for gt_id, ignore in zip(gt_ids, gt_ignore):
        #     if not ignore and gt_id not in matched_gt_ids:
        #         per_cat_stats[cat_id]['FN'] += 1
        per_cat_stats[cat_id]['FN'] += len(gt_ids)

    # 计算指标
    results = {}
    for cat_id, stats in per_cat_stats.items():
        tp, fp, fn = stats['TP'], stats['FP'], stats['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (fn) if (fn) > 0 else 0.0
        results[cat_id] = {'precision': precision, 'recall': recall}
    
    return results

def calculate_per_category_metrics(coco_eval, cat_ids):
    """计算每类别的详细指标 (AP50, AP50-95, precision, recall)"""
    # 1. 计算精确率和召回率
    precision_recall = calculate_fixed_confidence_metrics(coco_eval, cat_ids)
    
    # 2. 计算AP50和AP50-95
    results = {}
    iou_thres = coco_eval.params.iouThrs
    aind = [i for i, aRng in enumerate(coco_eval.params.areaRngLbl) if aRng == "all"]
    mind = [i for i, maxDets in enumerate(coco_eval.params.maxDets) if maxDets == 100]
    
    # 创建类别ID到索引的映射
    cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(coco_eval.params.catIds)}
    
    for cat_id in cat_ids:
        if cat_id not in cat_id_to_idx:
            continue
            
        idx = cat_id_to_idx[cat_id]
        metrics = {
            'precision': precision_recall[cat_id]['precision'],
            'recall': precision_recall[cat_id]['recall']
        }
        
        # 获取0.5 IoU索引
        if iou_thres is not None:
            iou_50_idx = np.where(iou_thres == 0.5)[0][0] if len(np.where(iou_thres == 0.5)[0]) > 0 else 0
        else:
            iou_50_idx = 0
        
        # 计算AP50（平均精度）
        precisions_iou50 = coco_eval.eval['precision'][iou_50_idx, :, idx, aind, mind].squeeze()
        valid_precisions = precisions_iou50[precisions_iou50 > -1]
        metrics['AP50'] = np.mean(valid_precisions) if valid_precisions.size > 0 else 0.0
        
        # 计算AP50-95
        ap_scores = []
        for t in range(len(iou_thres)):
            precisions_t = coco_eval.eval['precision'][t, :, idx, aind, mind].squeeze()
            valid_pt = precisions_t[precisions_t > -1]
            ap_score = np.mean(valid_pt) if valid_pt.size > 0 else 0.0
            ap_scores.append(ap_score)
        metrics['AP50-95'] = np.mean(ap_scores) if ap_scores else 0.0
        
        results[cat_id] = metrics
    
    return results


class OurDatasetEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, device, image_set='val', transform=None):
        # ----------------- Basic parameters -----------------
        self.image_set = image_set
        self.transform = transform
        self.device = device
        # ----------------- Metrics -----------------
        self.map = 0.
        self.ap50_95 = 0.
        self.ap50 = 0.
        # ----------------- Dataset -----------------
        self.dataset = OurDataset(data_dir=data_dir, image_set=image_set)


    @torch.no_grad()
    def evaluate(self, model, epoch=None, result_file="E:/Learning/深度学习/YoLo系列/YOLO书籍-全部源代码/YOLO_Tutorial/outputs/评估结果.txt"):
        model.eval()
        ids = []
        data_dict = []
        num_images = len(self.dataset)
        print('total number of images: %d' % (num_images))

        # 获取类别ID映射
        # cat_ids = self.dataset.get_cat_ids()
        cat_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # cats = self.dataset.coco.loadCats(cat_ids)
        # cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
        cat_id_to_name = {0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'}

        # start testing
        for index in range(num_images):
            if index % 500 == 0:
                print('[Eval: %d / %d]'%(index, num_images))

            # load an image
            img, id_ = self.dataset.pull_image(index)
            orig_h, orig_w, _ = img.shape

            # preprocess
            x, _, deltas = self.transform(img)
            x = x.unsqueeze(0).to(self.device) / 255.
            
            id_ = int(id_)
            ids.append(id_)
            
            # inference
            outputs = model(x)
            bboxes, scores, cls_inds = outputs

            # rescale bboxes
            origin_img_size = [orig_h, orig_w]
            cur_img_size = [*x.shape[-2:]]
            bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

            for i, box in enumerate(bboxes):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                label = int(cls_inds[i])
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(scores[i])
                A = {"image_id": id_, "category_id": cat_ids[label], "bbox": bbox,
                     "score": score}
                data_dict.append(A)

        annType = ['segm', 'bbox', 'keypoints']

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            print('evaluating ......')
            cocoGt = self.dataset.coco
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            
            # 保存结果到文件（追加模式）
            with open(result_file, 'a') as f:
                # 添加epoch标识
                f.write(f"\n\n{'='*50}\n")
                f.write(f"Evaluation Results for Epoch {epoch}\n")
                f.write(f"{'='*50}\n")
                
                # 1. 输出COCO的标准摘要
                f.write("Global COCO metrics:\n")
                cocoEval.summarize()
                f.write(f"AP@[0.5:0.95]: {cocoEval.stats[0]:.4f}\n")
                f.write(f"AP50: {cocoEval.stats[1]:.4f}\n")
                f.write(f"AP75: {cocoEval.stats[2]:.4f}\n")
                f.write(f"AP small: {cocoEval.stats[3]:.4f}\n")
                f.write(f"AP medium: {cocoEval.stats[4]:.4f}\n")
                f.write(f"AP large: {cocoEval.stats[5]:.4f}\n\n")
                
                # 2. 计算并输出每类别的详细指标
                f.write("\nPer-category metrics:\n")
                f.write(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'AP50':>10} {'AP50-95':>10}\n")
                f.write("-" * 60 + "\n")
                
                # 计算每类别指标
                results = calculate_per_category_metrics(cocoEval, cat_ids)
                
                for cat_id, metrics in results.items():
                    cat_name = cat_id_to_name.get(cat_id, f"Unknown({cat_id})")
                    f.write(f"{cat_name:<20} {metrics['precision']*100:10.2f} "
                            f"{metrics['recall']*100:10.2f} "
                            f"{metrics['AP50']*100:10.2f} "
                            f"{metrics['AP50-95']*100:10.2f}\n")
                
                # 3. 计算"All"类别的平均值
                all_precision = np.mean([m['precision'] for m in results.values()]) if results else 0.0
                all_recall = np.mean([m['recall'] for m in results.values()]) if results else 0.0
                all_AP50 = np.mean([m['AP50'] for m in results.values()]) if results else 0.0
                all_AP50_95 = np.mean([m['AP50-95'] for m in results.values()]) if results else 0.0
                
                f.write("-" * 60 + "\n")
                f.write(f"{'All':<20} {all_precision*100:10.2f} {all_recall*100:10.2f} "
                        f"{all_AP50 * 100:10.2f} {all_AP50_95 * 100:10.2f}\n\n")
            
            # 在控制台打印结果
            print("\nGlobal COCO metrics:")
            cocoEval.summarize()
            print(f"AP@[0.5:0.95]: {cocoEval.stats[0]:.4f}")
            print(f"AP50: {cocoEval.stats[1]:.4f}")
            print(f"AP75: {cocoEval.stats[2]:.4f}")
            print(f"AP small: {cocoEval.stats[3]:.4f}")
            print(f"AP medium: {cocoEval.stats[4]:.4f}")
            print(f"AP large: {cocoEval.stats[5]:.4f}")
            
            print("\nPer-category metrics:")
            print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'AP50':>10} {'AP50-95':>10}")
            print("-" * 60)
            
            for cat_id, metrics in results.items():
                cat_name = cat_id_to_name.get(cat_id, f"Unknown({cat_id})")
                print(f"{cat_name:<20} {metrics['precision']*100:10.2f} "
                      f"{metrics['recall']*100:10.2f} "
                      f"{metrics['AP50']*100:10.2f} "
                      f"{metrics['AP50-95']*100:10.2f}")
            
            print("-" * 60)
            print(f"{'All':<20} {all_precision*100:10.2f} {all_recall*100:10.2f} "
                  f"{all_AP50 * 100:10.2f} {all_AP50_95 * 100:10.2f}")
            
            # 更新类属性
            self.ap50_95 = cocoEval.stats[0]
            self.ap50 = cocoEval.stats[1]
            
            if epoch is not None:
                print(f'\nResults for epoch {epoch} saved to {result_file}')
            else:
                print(f'\nResults saved to {result_file}')
            
            return cocoEval.stats[1], cocoEval.stats[0]
        else:
            return 0, 0