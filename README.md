## 🛠️ 开发环境配置

### 基础环境
- **Python 版本**: 3.8.20
- **CUDA 版本**: 11.8
- **PyTorch 版本**: 2.0.1+cu118

### 安装步骤
```bash
conda create -n myenv python=3.8.20
conda activate myenv

# 安装PyTorch
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 安装项目依赖
pip install -r requirements.txt
```

# EA-DETR: An Enhanced Attention and Multi-Scale Fusion Detector for UAV Images

[![基于 RT-DETR](https://github.com/lyuwenyu/RT-DETR)]

## 项目来源
本项目基于以下开源工作辅助：
 [YOLO](https://github.com/yjh0410/RT-ODLab)
 [YOLO_ultralytics](https://github.com/ultralytics/ultralytics)

## 许可信息
- 原始RT-DETR代码遵循[MIT License](https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE)
- 本项目代码添加遵循[Apache License 2.0](LICENSE)
- 详细第三方声明见[NOTICES](NOTICES/)

> 注意：本项目代码中保留所有原始版权声明，详见各文件头部注释
