# # from ultralytics import YOLO
# # import torch
# #
# # # 加载训练好的模型（仅需传递模型路径）
# # model = YOLO("D:/newyolo11/ultralytics-main/runs/train/exp24/weights/best.pt")
# #
# # model.export(
# #     format="onnx",
# #     imgsz=640,
# #     simplify=True,
# #     opset=16,  # 尝试降低 opset 版本（如 16、15）
# #     dynamic=False
# # )
# # # from ultralytics import YOLO
# # #
# # # # 加载训练好的模型（.pt 文件路径）
# # # model = YOLO("D:/newyolo11/ultralytics-main/runs/train/exp24/weights/best.pt")
# # #
# # # # 导出为 ONNX 并包含 NMS 后处理
# # # model.export(
# # #     format="onnx",        # 导出格式为 ONNX
# # #     imgsz=640,            # 输入图像尺寸（与训练一致）
# # #     simplify=True,        # 简化 ONNX 模型（减少冗余节点）
# # #     opset=16,             # 使用 ONNX opset 16（兼容大多数推理引擎）
# # #     dynamic=False,        # 禁用动态输入尺寸（固定 640x640）
# # #     nms=True,             # 启用 NMS（关键！过滤冗余检测框）
# # #     half=False,           # 禁用 FP16 量化（避免精度损失）
# # #     int8=False,           # 禁用 INT8 量化（避免精度损失）
# # #     batch=1               # 批处理大小为 1（单张图片推理）
# # # )
# # from ultralytics import YOLO
# #
# # model = YOLO("D:/newyolo11/ultralytics-main/runs/train/exp24/weights/best.pt")
# # model.export(
# #     format="onnx",
# #     imgsz=(640, 640),
# #     simplify=True,
# #     opset=16,          # 使用兼容性较好的 opset
# #     dynamic=False,     # 固定输入尺寸
# #     half=False,        # 禁用 FP16
# #     int8=False,        # 禁用 INT8 量化
# #     nms=False          # 确保不自动添加 NMS（手动处理）
# # )
# # import onnx
# #
# # model = onnx.load("D:/newyolo11/ultralytics-main/runs/train/exp24/weights/best.onnx")
# # onnx.checker.check_model(model)  # 如果报错，说明模型导出失败
# 
# from ultralytics import YOLO
# model = YOLO("D:/newyolo11/ultralytics-main/runs/train/exp24/weights/best.pt")
# model.export(
#     format="onnx",
#     imgsz=640,
#     simplify=True,
#     opset=16,
#     dynamic=False,
#     half=False
# )
# 
# import cv2
# import numpy as np
# import onnxruntime as ort
# 
# # 加载 ONNX 模型
# ort_session = ort.InferenceSession("D:/newyolo11/ultralytics-main/runs/train/exp24/weights/best.onnx", providers=["CPUExecutionProvider"])
# 
# # 读取测试图片
# img = cv2.imread("D:/newyolo11/ultralytics-main/VOC2024/images/val/rebar_1_12MM.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV默认是BGR，需转RGB
# img_resized = cv2.resize(img, (640, 640))
# 
# # 预处理：归一化到 [0,1] 并调整维度顺序
# img_tensor = img_resized.astype(np.float32) / 255.0
# img_tensor = np.transpose(img_tensor, (2, 0, 1))[np.newaxis, ...]  # [1,3,640,640]
# 
# # 验证输入张量
# print("输入张量形状:", img_tensor.shape)  # 应为 (1, 3, 640, 640)
# print("输入范围:", np.min(img_tensor), np.max(img_tensor))  # 应为 [0.0, 1.0]
# print("是否有NaN:", np.isnan(img_tensor).any())  # 必须为 False
# print("数据类型:", img_tensor.dtype)  # 应为 float32
# 
# # 推理
# outputs = ort_session.run(None, {"images": img_tensor})
# print("模型输出形状:", [output.shape for output in outputs])

# from ultralytics import YOLO
#
# # Load a model
# # model = YOLO("yolo11n.pt")  # load an official model
# model = YOLO("D:/newyolo11/ultralytics-main/runs/train/exp33/weights/best.pt")  # load a custom trained model
#
# # Export the model
# model.export(format="onnx")
from ultralytics import YOLO

# 加载自定义训练的模型
model = YOLO("D:/newyolo11/ultralytics-main/runs/train/exp33/weights/best.pt")

# 导出为 ONNX 模型（关键参数调整）
model.export(
    format="onnx",          # 指定导出格式为 ONNX
    opset=13,               # 使用 ONNX opset 13（或更高，推荐 13+）
    half=False,             # 禁用半精度（确保所有张量为 float32）
    dynamic=False,          # 固定输入尺寸（可选，若需动态形状设为 True）
    simplify=True,          # 简化模型结构（自动优化冗余节点）
    device="cpu"            # 使用 CPU 导出（避免 GPU 特有的类型问题）
)