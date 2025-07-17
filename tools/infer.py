import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)



def draw(images, labels, boxes, scores, thrh=0.6, path=""):
    # 定义类别名称映射
    # class_names = {
    #     0: "pedestrian",
    #     1: "people",
    #     2: "bicycle",
    #     3: "car",
    #     4: "van",
    #     5: "truck",
    #     6: "tricycle",
    #     7: "awning-tricycle",
    #     8: "bus",
    #     9: "motor"
    # }
    class_names = {
        0: "Person",
        1: "Car",
        2: "Bicycle",
        3: "OtherVehicle",
        4: "DontCare",
    }
    
    # 定义类别颜色映射
    class_colors = {
        0: (255, 0, 0),    # 红色 - pedestrian
        1: (0, 255, 0),    # 绿色 - people
        2: (0, 0, 255),    # 蓝色 - bicycle
        3: (230, 180, 80),  # 黄色 - car
        4: (255, 0, 255),  # 紫色 - van
        5: (0, 255, 255),  # 青色 - truck
        6: (255, 165, 0),  # 橙色 - tricycle
        7: (128, 0, 128),  # 深紫色 - awning-tricycle
        8: (0, 128, 128),  # 蓝绿色 - bus
        9: (128, 128, 0),  # 橄榄色 - motor
        # 添加更多类别颜色映射...
    }
    
    # 默认颜色（当类别超出定义范围时使用）
    default_color = (200, 200, 200)  # 灰色
    
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        for j, b in enumerate(box):
            # 获取当前类别的ID
            class_id = int(lab[j].item())
            
            # 获取类别名称（如果找不到则显示"Unknown"）
            class_name = class_names.get(class_id, f"Class {class_id}")
            
            # 获取类别颜色
            color = class_colors.get(class_id, default_color)
            
            # 绘制边界框
            draw.rectangle(
                list(b),
                outline=color,
                width=3
            )
            
            # # 准备文本（使用类别名称代替数字）
            # text = f"{class_name}: {scrs[j].item():.2f}"
            
            # # 使用getbbox方法获取文本尺寸
            # text_bbox = font.getbbox(text)
            # text_width = text_bbox[2] - text_bbox[0]  # 计算宽度
            # text_height = text_bbox[3] - text_bbox[1]  # 计算高度
            
            # # 确保文本在图像范围内
            # text_x = max(0, min(b[0], im.width - text_width))
            # text_y = max(0, min(b[1] - text_height, im.height - text_height))
            
            # # 绘制文本背景矩形
            # bg_rect = [
            #     text_x, 
            #     text_y, 
            #     text_x + text_width, 
            #     text_y + text_height
            # ]
            # draw.rectangle(
            #     bg_rect,
            #     fill=color
            # )
            
            # # 绘制文本
            # draw.text(
            #     (text_x, text_y),
            #     text,
            #     font=font,
            #     fill=(255, 255, 255)  # 白色文本
            # )
            
        # 保存结果
        if path == "":
            im.save(f'results_{i}.jpg')
        else:
            im.save(path)
            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)
    
    transforms = T.Compose([
        T.Resize((640, 640)),  
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)
    if args.sliced:
        num_boxes = args.numberofboxes
        
        aspect_ratio = w / h
        num_cols = int(np.sqrt(num_boxes * aspect_ratio)) 
        num_rows = int(num_boxes / num_cols)
        slice_height = h // num_rows
        slice_width = w // num_cols
        overlap_ratio = 0.2
        slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
        predictions = []
        for i, slice_img in enumerate(slices):
            slice_tensor = transforms(slice_img)[None].to(args.device)
            with autocast():  # Use AMP for each slice
                output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
            torch.cuda.empty_cache() 
            labels, boxes, scores = output
            
            labels = labels.cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            predictions.append((labels, boxes, scores))
        
        merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
        labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
    else:
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        
    draw([im_pil], labels, boxes, scores, 0.6)
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',default="E:/论文写作/欣赏别人的论文/目标检测论文/RT-DETR/RT-DETR_V7-策略1+2/configs/rtdetr/rtdetr_r18vd_6x_coco.yml", type=str, )
    # parser.add_argument('--resume', '-r',default="E:/论文写作/欣赏别人的论文/目标检测论文/RT-DETR/RT-DETR_V7-策略1+2/output/rtdetr_r18vd_6x_coco_visdrone/checkpoint0719.pth", type=str, )
    parser.add_argument('--resume', '-r',default="E:/论文写作/欣赏别人的论文/目标检测论文/RT-DETR/RT-DETR_V7-策略1+2/output/HIT-AUV/checkpoint0204.pth", type=str, )
    parser.add_argument('-f', '--im-file',default="C:/Users/MSI/Desktop/HIT-AUV检测结果图/1_70_80_0_07734.jpg", type=str, )
    parser.add_argument('-s', '--sliced', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)











