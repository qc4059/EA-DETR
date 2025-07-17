# from fastapi import FastAPI, File, UploadFile, HTTPException
# import onnxruntime as ort
# import numpy as np
# from PIL import Image
# import cv2
# import json
#
# # uvicorn onnx_to_API:app --reload 终端运行
# # http://127.0.0.1:8000/docs 然后网页输入网址
#
#
# app = FastAPI(title="钢筋检测API")
#
# # 初始化 ONNX 推理会话（添加输入节点名称打印）
# ort_session = ort.InferenceSession(
#     "D:/newyolo11/ultralytics-main/runs/train/exp24/weights/best.onnx",
#     providers=["CPUExecutionProvider"]
# )
# print(f"模型输入节点名称: {ort_session.get_inputs()[0].name}")  # 打印输入名称（如 "images"）
#
#
# @app.post("/detect")
# async def detect(image: UploadFile = File(...)):
#     try:
#         print("\n===== 开始处理请求 =====")
#
#         # 1. 校验图片格式
#         if image.content_type not in ["image/jpeg", "image/png"]:
#             raise HTTPException(status_code=400, detail="仅支持JPG/PNG格式")
#         print("✅ 图片格式校验通过")
#
#         # 2. 读取并预处理图片
#         img = Image.open(image.file).convert("RGB")
#         print(f"✅ 图片读取成功，原图尺寸: {img.size}")
#         original_width, original_height = img.size
#         img_resized = img.resize((640, 640))
#         print("✅ 图片缩放至 640x640")
#
#         # 均值方差归一化（YOLO 必需预处理）
#         img_array = np.array(img_resized).astype(np.float32)
#         img_array /= 255.0
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         img_array = (img_array - mean) / std
#         print("✅ 均值方差归一化完成")
#
#         img_tensor = np.transpose(img_array, (2, 0, 1))  # [H, W, C] → [C, H, W]
#         img_tensor = img_tensor[np.newaxis, ...]  # [1, 3, 640, 640]
#         print(f"✅ 输入张量形状: {img_tensor.shape}, 数据类型: {img_tensor.dtype}")
#
#         # 3. 模型推理（动态获取输入名称）
#         input_name = ort_session.get_inputs()[0].name
#         outputs = ort_session.run(None, {input_name: img_tensor})
#         predictions = outputs[0]
#         print(f"✅ 模型推理完成，输出形状: {predictions.shape}")
#
#         # 4. 后处理（假设输出为 [1, N, 6]）
#         results = []
#         for box in predictions[0]:  # 遍历批次内的所有框
#             confidence = box[4]
#             cls_id = int(box[5])
#             x1, y1, x2, y2 = box[:4]
#
#             if confidence < 0.3:
#                 continue
#
#             x1_original = x1 * (original_width / 640)
#             y1_original = y1 * (original_height / 640)
#             x2_original = x2 * (original_width / 640)
#             y2_original = y2 * (original_height / 640)
#
#             results.append({
#                 "bbox": [float(x1_original), float(y1_original), float(x2_original), float(y2_original)],
#                 "confidence": float(confidence),
#                 "class": "钢筋" if cls_id == 0 else "其他"
#             })
#
#         print(f"✅ 处理完成，检测到 {len(results)} 个目标")
#         return {"status": "success", "results": results}
#
#     except Exception as e:
#         print(f"❌ 发生异常: {type(e).__name__}, 消息: {str(e)}")
#         return {"status": "error", "message": f"{type(e).__name__}: {str(e)}"}
from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
from ultralytics.utils import ops  # 用于NMS

app = FastAPI(title="钢筋检测API")

# 加载ONNX模型（加固异常处理）
try:
    ort_session = ort.InferenceSession(
        "D:/newyolo11/ultralytics-main/runs/train/exp33/weights/best.onnx",
        providers=["CPUExecutionProvider"]
    )
    print(f"输入节点: {ort_session.get_inputs()[0].name}")
except Exception as e:
    raise RuntimeError(f"模型加载失败: {e}")


def process_output(outputs, original_size, conf_thres=0.3):
    original_width, original_height = original_size
    predictions = outputs[0][0]  # 形状 (5, 8400)
    results = []

    # 遍历所有预测框（8400个）
    for i in range(predictions.shape[1]):
        # 获取当前框的5个参数
        x_center_norm = predictions[0, i]  # 归一化的中心点x坐标
        y_center_norm = predictions[1, i]  # 归一化的中心点y坐标
        width_norm = predictions[2, i]  # 归一化的宽度
        height_norm = predictions[3, i]  # 归一化的高度
        conf = predictions[4, i]  # 置信度

        if conf < conf_thres:
            continue

        # 将归一化坐标转换为原图坐标
        x1 = (x_center_norm - width_norm / 2) * original_width
        y1 = (y_center_norm - height_norm / 2) * original_height
        x2 = (x_center_norm + width_norm / 2) * original_width
        y2 = (y_center_norm + height_norm / 2) * original_height

        # 单类别模型，直接指定类别
        results.append([x1, y1, x2, y2, conf, 0])  # cls_id=0

    # 应用NMS
    if len(results) > 0:
        results = np.array(results)
        nms_results = ops.non_max_suppression(
            results[np.newaxis, ...],  # 输入形状 (1, N, 6)
            conf_thres=0.3,
            iou_thres=0.5,
            max_det=100
        )[0]
        return [
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "class": "钢筋"
            }
            for x1, y1, x2, y2, conf, _ in nms_results
        ]
    return []


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # 校验图片格式
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(400, "仅支持JPG/PNG")

        # 读取并预处理
        img = Image.open(image.file).convert("RGB")
        original_size = img.size
        img_resized = img.resize((640, 640))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]  # [1,3,640,640]

        # 推理
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: img_tensor})

        ########################################################
        # 添加打印语句，验证模型输出格式
        ########################################################
        print(f"模型输出形状: {[output.shape for output in outputs]}")  # 查看所有输出的形状
        print("示例输出数据（第一个输出的前5个预测框前5个元素）:")
        print(outputs[0][0, :5, :5])  # 假设 outputs[0].shape 是 (1, N, 6)
        print("输出数据示例（前10个预测框）:")
        print(outputs[0][0, :, :10].T)  # 输出形状 (1,5,8400) → 取前10个框

        # 检查是否有 NaN
        print("输出是否有 NaN:", np.isnan(outputs[0]).any())  # 必须为 False

        # 后处理
        results = process_output(outputs, original_size)
        return {"status": "success", "results": results}

    except Exception as e:
        raise HTTPException(500, detail=f"处理失败: {str(e)}")