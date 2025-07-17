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
