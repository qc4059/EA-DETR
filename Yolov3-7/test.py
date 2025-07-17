import argparse
import cv2
import os
import time
import numpy as np
from copy import deepcopy
import torch

# load transform
from dataset.build import build_dataset, build_transform

# load some utils
from utils.misc import load_weight, compute_flops
from utils.box_ops import rescale_bboxes

from config import build_dataset_config, build_model_config, build_trans_config
from models import build_model

from evaluator.build import build_evluator
from utils import distributed_utils
def measure_fps(model, transform, device, test_image, 
                num_iterations=100, warmup_iterations=10):
    """
    执行专业级的FPS测量，包括预热和各阶段时间统计
    """
    # 记录原始图像尺寸
    orig_h, orig_w = test_image.shape[:2]
    origin_img_size = [orig_h, orig_w]
    
    # 预热GPU
    print(f"🔥 预热GPU ({warmup_iterations}次迭代)...")
    for _ in range(warmup_iterations):
        # 接收所有三个返回值
        x, _, _ = transform(test_image)
        x = x.unsqueeze(0).to(device) / 255.
        model(x)
    
    # 初始化计时器
    total_preprocess = []
    total_inference = []
    total_postprocess = []
    total_fps = []
    
    # 主测量循环
    print(f"📊 开始FPS测量 ({num_iterations}次迭代)...")
    for _ in range(num_iterations):
        # 记录开始时间
        start_time = time.perf_counter()
        
        # ===== 预处理阶段 =====
        t0 = time.time()
        # 获取所有三个返回值
        x, _, deltas = transform(test_image)
        x = x.unsqueeze(0).to(device) / 255.
        t_pre = (time.time() - t0) * 1000  # 毫秒
        
        # ===== 推理阶段 =====
        t1 = time.time()
        bboxes, scores, labels = model(x)
        t_inf = (time.time() - t1) * 1000  # 毫秒
        
        # ===== 后处理阶段 =====
        t2 = time.time()
        # 后处理包括边界框缩放等操作
        cur_img_size = [*x.shape[-2:]]
        bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)
        t_post = (time.time() - t2) * 1000  # 毫秒
        
        # 总时间
        total_time = t_pre + t_inf + t_post
        
        total_preprocess.append(t_pre)
        total_inference.append(t_inf)
        total_postprocess.append(t_post)
        
        # 确保不除以0
        min_valid_time = 0.001  # 1微秒
        actual_time = max(total_time, min_valid_time)
        total_fps.append(1000 / actual_time)
        
        # 防止设备过载
        time.sleep(0.001)
    
    # 计算统计指标
    import numpy as np
    stats = {
        'preprocess_avg': np.mean(total_preprocess) if total_preprocess else 0,
        'preprocess_std': np.std(total_preprocess) if total_preprocess else 0,
        'inference_avg': np.mean(total_inference) if total_inference else 0,
        'inference_std': np.std(total_inference) if total_inference else 0,
        'postprocess_avg': np.mean(total_postprocess) if total_postprocess else 0,
        'postprocess_std': np.std(total_postprocess) if total_postprocess else 0,
        'fps_avg': np.mean(total_fps) if total_fps else 0,
        'fps_std': np.std(total_fps) if total_fps else 0,
        'min_latency': min(total_preprocess) + min(total_inference) + min(total_postprocess),
        'max_latency': max(total_preprocess) + max(total_inference) + max(total_postprocess),
        'p95_latency': np.percentile(
            [pre+inf+post for pre, inf, post in zip(total_preprocess, total_inference, total_postprocess)], 
            95),
        'iterations': num_iterations,
        'warmup': warmup_iterations,
        'preprocess_min': min(total_preprocess) if total_preprocess else 0,
        'preprocess_max': max(total_preprocess) if total_preprocess else 0,
        'inference_min': min(total_inference) if total_inference else 0,
        'inference_max': max(total_inference) if total_inference else 0,
        'postprocess_min': min(total_postprocess) if total_postprocess else 0,
        'postprocess_max': max(total_postprocess) if total_postprocess else 0
    }
    
    # 计算总体延迟
    total_latencies = [p + i + po for p, i, po in zip(total_preprocess, total_inference, total_postprocess)]
    stats['latency_avg'] = np.mean(total_latencies)
    stats['latency_std'] = np.std(total_latencies)
    
    return stats


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-Tutorial')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int, help='输入图像的最大尺寸')

    parser.add_argument('--show', action='store_true', default=False, help='上是否展示可视化结果')

    parser.add_argument('--save', action='store_true', default=False, help='是否将检测结果保存到指定目录中去')

    parser.add_argument('--cuda', action='store_true', default=True,  help='use cuda.')

    parser.add_argument('--save_folder', default='det_results', type=str, help='指定保存检测结果的文件夹路径，一般与上面的--save连用,save为true才会保存')

    parser.add_argument('-vt', '--visual_threshold', default=0.3, type=float, help='置信度阈值')
    parser.add_argument('-ws', '--window_scale', default=1.0, type=float, help='调整可视化窗口的大小')
    parser.add_argument('--resave', action='store_true', default=False, help='保存模型权重时，是否不保存优化器参数')

    # model
    parser.add_argument('-m', '--model', default='yolov5_l', type=str, help='指定YOLO版本')

    parser.add_argument('--weight', default="E:/Learning/深度学习/YoLo系列/YOLO书籍-全部源代码/YOLO_Tutorial/weights/VisDrone/yolov5_l/yolov5_l_best.pth", type=str, help='指定加载预训练模型的路径')
    # parser.add_argument('--weight', default="E:/Learning/深度学习/YoLo系列/YOLO书籍-全部源代码/YOLO_Tutorial/weights/VisDrone/yolox_l/yolox_l_epoch_392.pth", type=str, help='指定加载预训练模型的路径')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float, help='推理时的 置信度阈值')

    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float, help='非极大值抑制的阈值')

    parser.add_argument('--topk', default=100, type=int, help='topk candidates for testing')
    parser.add_argument("--no_decode", action="store_true", default=False, help="not decode in inference or yes")
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False, help='是否将卷积层和批归一化层融合，以提升推理速度')

    # dataset
    parser.add_argument('--root', default='E:/Learning/Data', help='数据集目录')

    parser.add_argument('-d', '--dataset', default='VisDrone', help='coco, voc, widerface, VisDrone')
    
    parser.add_argument('--min_box_size', default=8.0, type=float, help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float, help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float, help='mixup augmentation.')
    parser.add_argument('--load_cache', action='store_true', default=False, help='load data into memory.')
    parser.add_argument('--measure_fps', action='store_true',default=True,
                        help='执行专业FPS测量')
    parser.add_argument('--fps_image', type=str, default="E:/Learning/Data/VisDrone/VisDrone2019-DET-test-dev/images/9999952_00000_d_0000133.jpg",
                        help='用于FPS测量的图像路径')
    parser.add_argument('--fps_iterations', type=int, default=100,
                        help='FPS测量的迭代次数')
    return parser.parse_args()


# 绘制单个的bbox
def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # 绘制bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    # 在bbox上添加类别标签
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img

# 可视化单张图片中的所有bbox
def visualize(img, 
              bboxes, 
              scores, 
              labels, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(labels[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img

# 测试模型的主函数
def eval(self, model):
    # 如果启动了EMA，则使用保存在EMA中的模型参数来进行测试
    # 否则，使用当前的模型参数进行测试
    model_eval = model if self.model_ema is None else self.model_ema.ema

    # 对于分布式训练，只在Rank0线程上进行测试
    if distributed_utils.is_main_process():
        # 如果Evaluator类为None，则只保存模型，不测试（无法测试）
        if self.evaluator is None:
            print('No evaluator ... save model and go on training.')
            print('Saving state, epoch: {}'.format(self.epoch + 1))
            weight_name = '{}_no_eval.pth'.format(self.args.model)
            checkpoint_path = os.path.join(self.path_to_save, weight_name)
            torch.save({'model': model_eval.state_dict(),
                        'mAP': -1.,
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': self.epoch,
                        'args': self.args}, 
                        checkpoint_path)               
        # 如果Evaluator类不是None，则进行测试
        else:
            print('Evaluating model ...')
            # 将模型切换至torch要求的eval模式
            model_eval.eval()
            # 设置模型中的trainable为False，以便模型做前向推理（包括各种后处理）
            model_eval.trainable = False

            # 测试模型的性能
            with torch.no_grad():
                self.evaluator.evaluate(model_eval)

            # 只有当前的性能指标大于上一次的指标，才会保存模型权重
            cur_map = self.evaluator.map
            if cur_map > self.best_map:
                # update best-map
                self.best_map = cur_map
                # save model
                print('Saving state, epoch:', self.epoch + 1)
                weight_name = '{}_best.pth'.format(self.args.model)
                checkpoint_path = os.path.join(self.path_to_save, weight_name)
                torch.save({'model': model_eval.state_dict(),
                            'mAP': round(self.best_map*100, 1),
                            'optimizer': self.optimizer.state_dict(),
                            'epoch': self.epoch,
                            'args': self.args}, 
                            checkpoint_path)                      

            # 将模型切换至torch要求的train模式，以便继续训练
            model_eval.train()
            model_eval.trainable = True

# 测试函数
@torch.no_grad()
def test(args,
         model, 
         device, 
         dataset,
         evaluator=None,  # 新增evaluator参数
         transform=None,
         class_colors=None, 
         class_names=None, 
         class_indexs=None):
    

    if args.measure_fps:
        print("\n🔬 开始专业级FPS测量...")
        
        # 准备测试图像
        test_image = None
        
        # 尝试加载指定图像
        if args.fps_image and os.path.exists(args.fps_image):
            try:
                img = cv2.imread(args.fps_image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_image = img
                print(f"📷 使用指定图像进行FPS测量: {args.fps_image}")
            except Exception as e:
                print(f"⚠️ 无法加载指定图像: {e}")
                test_image = None
        
        # 如果未指定图像或加载失败，尝试使用数据集图像
        if test_image is None:
            if len(dataset) > 0:
                try:
                    test_image, _ = dataset.pull_image(0)
                    print("📷 使用数据集的第一张图像进行FPS测量")
                except:
                    test_image = None
            else:
                print("📷 使用随机生成的图像进行FPS测量")
        
        # 生成一个测试图像
        if test_image is None:
            # 生成随机图像作为测试
            test_image = np.random.randint(0, 255, (args.img_size, args.img_size, 3), dtype=np.uint8)
            print("📷 使用随机生成的图像进行FPS测量")
        
        # 设置预热次数
        warmup = max(10, args.fps_iterations // 10)  # 预热次数为总次数的10%
        
        # 执行测量
        fps_stats = measure_fps(
            model=model,
            transform=transform,
            device=device,
            test_image=test_image,
            num_iterations=args.fps_iterations,
            warmup_iterations=warmup
        )
        
        # 打印结果
        print("\n" + "="*70)
        print(f"{'FPS Measurement Results':^70}")
        print("="*70)
        print(f"{'Stage':<15}{'Avg (ms)':>10}{'Std Dev':>10}{'Range (ms)':>15}")
        print(f"{'-'*70}")
        
        pre_min, pre_max = fps_stats['preprocess_min'], fps_stats['preprocess_max']
        inf_min, inf_max = fps_stats['inference_min'], fps_stats['inference_max']
        post_min, post_max = fps_stats['postprocess_min'], fps_stats['postprocess_max']
        total_min, total_max = fps_stats['min_latency'], fps_stats['max_latency']
        
        print(f"{'Preprocess':<15}{fps_stats['preprocess_avg']:10.2f}{fps_stats['preprocess_std']:10.2f}{pre_min:.2f}-{pre_max:.2f}")
        print(f"{'Inference':<15}{fps_stats['inference_avg']:10.2f}{fps_stats['inference_std']:10.2f}{inf_min:.2f}-{inf_max:.2f}")
        print(f"{'Postprocess':<15}{fps_stats['postprocess_avg']:10.2f}{fps_stats['postprocess_std']:10.2f}{post_min:.2f}-{post_max:.2f}")
        print(f"{'-'*70}")
        print(f"{'Total Latency':<15}{fps_stats['latency_avg']:10.2f}{fps_stats['latency_std']:10.2f}{total_min:.2f}-{total_max:.2f}")
        print(f"{'FPS':<15}{fps_stats['fps_avg']:10.2f}{fps_stats['fps_std']:10.2f}")
        print(f"{'95% Latency':<15}{fps_stats['p95_latency']:10.2f}ms")
        print("="*70)
        print(f"测量迭代次数: {args.fps_iterations} | 预热迭代次数: {warmup}")
        
        # 提前返回，不再执行常规检测
        return
    

    num_images = len(dataset)
    save_path = os.path.join('det_results', args.dataset, args.model)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        orig_h, orig_w, _ = image.shape

        # 数据预处理
        x, _, deltas = transform(image)
        x = x.unsqueeze(0).to(device) / 255.

        # 记录前向推理的耗时，以便计算FPS，默认时间单位为“秒(s)”
        t0 = time.time()
        # 模型前向推理，包括后处理等步骤
        bboxes, scores, labels = model(x)
        # 计算前向推理的耗时
        print("detection time used ", time.time() - t0, "s")
        
        # 依据原始图像的尺寸，调整预测bbox的坐标
        origin_img_size = [orig_h, orig_w]
        cur_img_size = [*x.shape[-2:]]
        bboxes = rescale_bboxes(bboxes, origin_img_size, cur_img_size, deltas)

        # 绘制检测结果
        img_processed = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            labels=labels,
                            vis_thresh=args.visual_threshold,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset_name=args.dataset)
        
        # 如果args.show为True，则可视化上面绘制的检测结果
        if args.show:
            h, w = img_processed.shape[:2]
            sw, sh = int(w*args.window_scale), int(h*args.window_scale)
            cv2.namedWindow('detection', 0)
            cv2.resizeWindow('detection', sw, sh)
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)

        # 如果args.save为True，则保存上面绘制的检测结果
        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)

    # ================= 新增评估逻辑 =================
    if evaluator is not None:
        print('\nEvaluating model...')
        model.eval()            # 切换为评估模式
        model.trainable = False  # 关闭训练模式
        
        with torch.no_grad():
            evaluator.evaluate(model)  # 执行评估
            
        print("mAP: {:.2f}%\n".format(evaluator.map * 100))
        
        model.train()           # 恢复训练模式
        model.trainable = True  # 恢复训练标志
    # ================= 评估结束 =================

if __name__ == '__main__':
    args = parse_args()
    # 如果args.cuda为True，则使用GPU来推理，否则使用CPU来训练（可接受）
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 构建测试所用到的 Dataset & Model & Transform相关的config变量
    data_cfg = build_dataset_config(args)
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # 构建测试所用到的数据预处理Transform类
    val_transform, trans_cfg = build_transform(args, trans_cfg, model_cfg['max_stride'], is_train=False)

    # 构建测试所用到的Dataset类
    dataset, dataset_info = build_dataset(args, data_cfg, trans_cfg, val_transform, is_train=False)
    num_classes = dataset_info['num_classes']

    # 用于标记不同类别的bbox的颜色，更加美观
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    evaluator = build_evluator(args, data_cfg, val_transform, device)

    # 构建YOLO模型
    model = build_model(args, model_cfg, device, num_classes, False)

    # 加载已训练好的模型权重
    model = load_weight(model, args.weight, args.fuse_conv_bn)
    model.to(device).eval()

    # 计算模型的参数量和FLOPs
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    compute_flops(
        model=model_copy,
        img_size=args.img_size, 
        device=device)
    del model_copy

    # 调用测试函数时传入评估器
    test(args=args,
        model=model,
        device=device,
        dataset=dataset,
        evaluator=evaluator,  # 传递评估器实例
        transform=val_transform,
        class_colors=class_colors,
        class_names=dataset_info['class_names'],
        class_indexs=dataset_info['class_indexs'])
    

    # 如果args.resave为True，则重新保存模型的权重，
    # 因为在训练阶段，模型权重文件中还包含了优化器、学习率策略等参数，这会使得权重文件过大
    # 因为，为了缩小文件的大小，可以重新保存一次，只保存模型的参数
    if args.resave:
        print('Resave: {}'.format(args.model.upper()))
        checkpoint = torch.load(args.weight, map_location='cpu')
        checkpoint_path = 'weights/{}/{}/{}_pure.pth'.format(args.dataset, args.model, args.model)
        torch.save({'model': model.state_dict(),
                    'mAP': checkpoint.pop("mAP"),
                    'epoch': checkpoint.pop("epoch")}, 
                    checkpoint_path)
        
    print("================= DETECT =================")
    # 开始在指定的数据集上去测试我们的代码
    # 对于使用VOC数据集训练出来的模型，就使用VOC测试集来做测试
    # 对于使用COCO数据集训练出来的模型，就使用COCO验证机来做测试
    test(args         = args,
         model        = model, 
         device       = device, 
         dataset      = dataset,
         transform    = val_transform,
         class_colors = class_colors,
         class_names  = dataset_info['class_names'],
         class_indexs = dataset_info['class_indexs'],
         )



# python test.py --cuda -d voc -m yolov1 --weight weight/voc/yolov1/yolov1_voc.pth --show-size 416 -vt 0.3