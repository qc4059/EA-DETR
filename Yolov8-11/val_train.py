import os, shutil
from sklearn.model_selection import train_test_split


val_size = 0.2
#test_size = 0.2
postfix = 'jpg'
imgpath = r'D:/newyolo11/ultralytics-main/VOC2024/JPEGImages'
txtpath =  r'D:/newyolo11/ultralytics-main/VOC2024/YOLOLabels'



output_train_img_folder = r'D:/newyolo11/ultralytics-main/VOC2024/images/train'
output_val_img_folder = r'D:/newyolo11/ultralytics-main/VOC2024/images/val'
output_train_txt_folder = r'D:/newyolo11/ultralytics-main/VOC2024/labels/train'
output_val_txt_folder = r'D:/newyolo11/ultralytics-main/VOC2024/labels/val'

os.makedirs(output_train_img_folder, exist_ok=True)
os.makedirs(output_val_img_folder, exist_ok=True)
os.makedirs(output_train_txt_folder, exist_ok=True)
os.makedirs(output_val_txt_folder, exist_ok=True)


listdir = [i for i in os.listdir(txtpath) if 'txt' in i]
train, val = train_test_split(listdir, test_size=val_size, shuffle=True, random_state=0)

#todo：需要test放开

# train, test = train_test_split(listdir, test_size=test_size, shuffle=True, random_state=0)
# train, val = train_test_split(train, test_size=val_size, shuffle=True, random_state=0)

for i in train:
    img_source_path = os.path.join(imgpath, '{}.{}'.format(i[:-4], postfix))
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_train_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_train_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)

for i in val:
    img_source_path = os.path.join(imgpath, '{}.{}'.format(i[:-4], postfix))
    txt_source_path = os.path.join(txtpath, i)

    img_destination_path = os.path.join(output_val_img_folder, '{}.{}'.format(i[:-4], postfix))
    txt_destination_path = os.path.join(output_val_txt_folder, i)

    shutil.copy(img_source_path, img_destination_path)
    shutil.copy(txt_source_path, txt_destination_path)



