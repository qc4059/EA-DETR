import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('E:/Learning/深度学习/YoLo系列/v11/ultralytics/cfg/models/v9/yolov9m.yaml')#F:/newyolo11/ultralytics-main/ultralytics/cfg/models/11/yolo11-C3k2-PCCDSConv.yaml
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,ultralytics\cfg\models\v5\yolov5.yamlF:\newyolo11\ultralytics-main\ultralytics\cfg\models\11\yolo11-SCInet-PCCDSConv.yaml
    # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    # model.load("E:/Learning/深度学习/YoLo系列/v11/runs/train/YOLOv11-VisDrone3/weights/best.pt") # 是否加载预训练权重,科研不建议大家加载否则很难提升精度    yolo11-ARE-PCCDSConv.yaml
    #ultralytics\cfg\models\11\yolo11-SCInet-PCCDSConv.yaml  ultralytics\cfg\models\v9\yolov9s.yamlultralytics\cfg\models\v8\yolov8.yaml
    model.train(data=r"ultralytics/cfg/datasets/VisDrone.yaml",
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=400,
                single_cls=False,  # 是否是单类别检测
                batch=16,
                close_mosaic=0, 
                workers=4,   # 0 -> 4
                device='cuda',
                optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
                resume=False, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                amp=True,  # 如果出现训练损失为Nan可以关闭amp
                project='runs/train',
                name='YOLOv11-VisDrone2',
                )
    