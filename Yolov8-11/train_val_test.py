import os
import shutil
from sklearn.model_selection import train_test_split

# 划分比例
train_size = 0.8
val_size = 0.1
test_size = 0.1

postfix = 'jpg'
imgpath = r'D:/newyolo11/ultralytics-main/VOC2024/JPEGImages'
txtpath = r'D:/newyolo11/ultralytics-main/VOC2024/YOLOLabels'

output_train_img_folder = r'D:/newyolo11/ultralytics-main/VOC2024/images/train'
output_val_img_folder = r'D:/newyolo11/ultralytics-main/VOC2024/images/val'
output_test_img_folder = r'D:/newyolo11/ultralytics-main/VOC2024/images/test'
output_train_txt_folder = r'D:/newyolo11/ultralytics-main/VOC2024/labels/train'
output_val_txt_folder = r'D:/newyolo11/ultralytics-main/VOC2024/labels/val'
output_test_txt_folder = r'D:/newyolo11/ultralytics-main/VOC2024/labels/test'

# 创建输出文件夹
os.makedirs(output_train_img_folder, exist_ok=True)
os.makedirs(output_val_img_folder, exist_ok=True)
os.makedirs(output_test_img_folder, exist_ok=True)
os.makedirs(output_train_txt_folder, exist_ok=True)
os.makedirs(output_val_txt_folder, exist_ok=True)
os.makedirs(output_test_txt_folder, exist_ok=True)

# 获取所有 txt 文件列表
listdir = [i for i in os.listdir(txtpath) if 'txt' in i]

# 先将数据集划分为训练集和临时集（验证集 + 测试集）
train, temp = train_test_split(listdir, test_size=(val_size + test_size), shuffle=True, random_state=0)

# 再将临时集划分为验证集和测试集
val, test = train_test_split(temp, test_size=test_size / (val_size + test_size), shuffle=True, random_state=0)

# 复制训练集数据
for i in train:
    img_source_path = os.path.join(imgpath, '{}.{}'.format(i[:-4], postfix))
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_train_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_train_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)

# 复制验证集数据
for i in val:
    img_source_path = os.path.join(imgpath, '{}.{}'.format(i[:-4], postfix))
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_val_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_val_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)

# 复制测试集数据
for i in test:
    img_source_path = os.path.join(imgpath, '{}.{}'.format(i[:-4], postfix))
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_test_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_test_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)