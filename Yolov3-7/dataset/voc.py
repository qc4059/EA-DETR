"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import random
import torch.utils.data as data
import cv2
import numpy as np
import xml.etree.ElementTree as ET

try:
    from .data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment
except:
    from data_augment.yolov5_augment import yolov5_mosaic_augment, yolov5_mixup_augment, yolox_mixup_augment



VOC_CLASSES = (  # always index 0
    'rebar', )

# 这个类，就是读取xml文件，将其中的目标物 边界框位置信息、类别索引  放在一个列表中。
class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        # print(target)
        # print('----------------------------')
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,  img_size=640, data_dir=None, image_sets=[('2007', 'trainval'), ('2012', 'trainval')],  #数据集的划分
                 trans_config=None,
                 transform=None,
                 is_train=False,
                 load_cache=False
                 ):
        self.root = data_dir   # 数据集的路径   
        self.img_size = img_size
        self.image_set = image_sets  # 数据集 划分 列表
        self.target_transform = VOCAnnotationTransform()   # 类的初始化
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.is_train = is_train
        self.load_cache = load_cache
        for (year, name) in image_sets:
            # rootpath = osp.join(self.root, 'VOC' + year)
            rootpath = osp.join(self.root, '' + '')   #  
            rootpath = osp.normpath(rootpath)
            rootpath2 = osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')
            rootpath2 = osp.normpath(rootpath2)
            for line in open(rootpath2):    # ids列表 中 元组作为列表元素 ，每个元组 ('E:\\Learning\\Data\\VOCdevkit\\VOC2007', '000005'),.....
                self.ids.append((rootpath, line.strip()))

        # augmentation
        self.transform = transform  # 数据预处理函数
        self.mosaic_prob = trans_config['mosaic_prob'] if trans_config else 0.0
        self.mixup_prob = trans_config['mixup_prob'] if trans_config else 0.0
        self.trans_config = trans_config
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')

        # load cache data
        if load_cache:
            self._load_cache()


    def __getitem__(self, index):   # 对于你实例化的对象（通常是 序列对象） 想要通过索引 进行访问 内部元素时，就可以定义这个函数
        image, target, deltas = self.pull_item(index)
        return image, target, deltas


    def __len__(self):
        return len(self.ids)


    def _load_cache(self):
        # load image cache
        self.cached_images = []
        self.cached_targets = []
        dataset_size = len(self.ids)

        print('loading data into memory ...')
        for i in range(dataset_size):
            if i % 5000 == 0:
                print("[{} / {}]".format(i, dataset_size))
            # load an image
            image, image_id = self.pull_image(i)
            orig_h, orig_w, _ = image.shape

            # resize image
            r = self.img_size / max(orig_h, orig_w)
            if r != 1: 
                interp = cv2.INTER_LINEAR
                new_size = (int(orig_w * r), int(orig_h * r))
                image = cv2.resize(image, new_size, interpolation=interp)
            img_h, img_w = image.shape[:2]
            self.cached_images.append(image)

            # load target cache
            anno = ET.parse(self._annopath % image_id).getroot()
            anno = self.target_transform(anno)
            anno = np.array(anno).reshape(-1, 5)
            boxes = anno[:, :4]
            labels = anno[:, 4]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / orig_w * img_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / orig_h * img_h
            self.cached_targets.append({"boxes": boxes, "labels": labels})
        

    def load_image_target(self, index):  #读取 图像与标签 函数
        if self.load_cache:
            image = self.cached_images[index]
            target = self.cached_targets[index]
            height, width, channels = image.shape
            target["orig_size"] = [height, width]
        else:
            # load an image
            img_id = self.ids[index]
            # 解释一下下面这段 代码    img_id: ('E:\\Learning\\Data\\VOCdevkit\\VOC2007', '000005')
            # self._imgpath：'%s\\JPEGImages\\%s.jpg' 这里有两个引用字符串变量的地方，后面img_id 中有两个变量，可以引用进去
            image = cv2.imread(self._imgpath % img_id)  # 通过ids列表 读取图片  第一次见用变量引用  牛  得到的结果是一个完整得到图片路径
            # print alrealy loaded photo
            # cv2.imshow("Loaded Image", image)  # 显示图片，窗口标题为 "Loaded Image"
            # cv2.waitKey(0)                     # 等待按键事件
            # cv2.destroyAllWindows()            # 关闭所有窗口

            height, width, channels = image.shape

            # laod an annotation   # 这里跟上面操作一样    _annopath： '%s\\Annotations\\%s.xml'
            anno = ET.parse(self._annopath % img_id).getroot()  # 读取每张图片对应的 xml文件
            if self.target_transform is not None:
                anno = self.target_transform(anno)  # 这里得到的 anno是列表，并且维度为 (n,4+1) n指的是一张图片中的目标数，4：边界框参数，1：类别 

            # guard against no boxes via resizing    同一张图片中，非diffcult的目标物 的 边界框信息 和 类别索引

            # 这里是为了确定 anno的维度 为 (n,5)
            anno = np.array(anno).reshape(-1, 5)  #array([[262, 210, 323, 338,   8], [164, 263, 252, 371,   8], [240, 193, 294, 298,   8]])  8 是chair
            target = {   # target 是每张图片中，所有目标物的边界框信息 和 类别索引 和 图片像素大小信息
                "boxes": anno[:, :4],  # 所有目标物 边界框信息
                "labels": anno[:, 4],  # 所有目标物 类别信息
                "orig_size": [height, width] # 该张 图片的大小
            }
        
        return image, target


    def load_mosaic(self, index):
        # load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]

        # load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # Mosaic
        if self.trans_config['mosaic_type'] == 'yolov5_mosaic':
            image, target = yolov5_mosaic_augment(
                image_list, target_list, self.img_size, self.trans_config, self.is_train)

        return image, target


    def load_mixup(self, origin_image, origin_target):
        # YOLOv5 type Mixup
        if self.trans_config['mixup_type'] == 'yolov5_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_mosaic(new_index)
            image, target = yolov5_mixup_augment(
                origin_image, origin_target, new_image, new_target)
        # YOLOX type Mixup
        elif self.trans_config['mixup_type'] == 'yolox_mixup':
            new_index = np.random.randint(0, len(self.ids))
            new_image, new_target = self.load_image_target(new_index)
            image, target = yolox_mixup_augment(
                origin_image, origin_target, new_image, new_target, self.img_size, self.trans_config['mixup_scale'])

        return image, target
    

    def pull_item(self, index):
        # random.random 随机生成一个 0-1 的浮点数
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)

        # MixUp
        if random.random() < self.mixup_prob:
            image, target = self.load_mixup(image, target)

        # augment
        image, target, deltas = self.transform(image, target, mosaic)
        ## SSD-style Augmentation
                                                                            # class SSDAugmentation(object):
                                                                            #     def __init__(self, img_size=640):
                                                                            #         self.img_size = img_size # 640
                                                                            #         self.augment = Compose([
                                                                            #             ConvertFromInts(),                         # 将int类型转换为float32类型
                                                                            #             PhotometricDistort(),                      # 图像颜色增强
                                                                            #             Expand(),                                  # 扩充增强
                                                                            #             RandomSampleCrop(),                        # 随机剪裁
                                                                            #             RandomHorizontalFlip(),                    # 随机水平翻转
                                                                            #             Resize(self.img_size)                      # resize操作
                                                                            #         ])

        return image, target, deltas


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    import argparse
    from build import build_transform
    
    parser = argparse.ArgumentParser(description='VOC-Dataset')

    # opt
    parser.add_argument('--root', default='E:\Learning\Data\VOCdevkit',
                        help='data root')
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='input image size.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--is_train', action="store_true", default=False,
                        help='mixup augmentation.')
    parser.add_argument('--load_cache', action="store_true", default=False,
                        help='load cached data.')
    
    args = parser.parse_args()

    yolov5_trans_config = {
        'aug_type': 'yolov5',            # 或者改为'ssd'来使用SSD风格的数据增强
        # Basic Augment
        'degrees': 0.0,                  # 可以修改数值来决定旋转图片的程度，如改为YOLOX默认的10.0
        'translate': 0.2,                # 可以修改数值来决定平移图片的程度，
        'scale': [0.1, 2.0],             # 图片尺寸扰动的比例范围
        'shear': 0.0,                    # 可以修改数值来决定旋转图片的程度，如改为YOLOX默认的2.0
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        # Mosaic & Mixup
        'mosaic_prob': 1.0,              # 使用马赛克增强的概率：0～1
        'mixup_prob': 1.0,               # 使用混合增强的概率：0～1
        'mosaic_type': 'yolov5_mosaic',
        'mixup_type': 'yolox_mixup',     # 或者改为'yolov5_mixup'，使用yolov5风格的混合增强
        'mixup_scale': [0.5, 1.5]
    }

    ssd_trans_config = {
        'aug_type': 'ssd',
        'mosaic_prob': '0.0',
        'mixup_prob': '0.0'
    }

    transform, trans_cfg = build_transform(args, ssd_trans_config, 32, args.is_train)

    dataset = VOCDetection(
        img_size=args.img_size,
        data_dir=args.root,
        trans_config=ssd_trans_config,
        transform=transform,
        is_train=args.is_train,
        load_cache=args.load_cache
        )
    # print(len(dataset))  # 这个地方 可以用len 是因为这个类定义了__len__()函数

    # 这里想检验 dataset里面是什么的 下面是检验
    # for i in dataset:
                      # image是数据预处理变化后的维度 并且将其顺序换为(C,H,W)符合pytorch     target数据预处理后的 边界框和类别信息
        # print(i)    # 这里的 i 返回的是 image, target, deltas 


    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))


    for i in range(10):
        image, target, deltas = dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to uint8
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]
        print(type(target))
        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            if x2 - x1 > 1 and y2 - y1 > 1:
                cls_id = int(label)
                color = class_colors[cls_id]
                # class name
                label = VOC_CLASSES[cls_id]
                image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                # put the test on the bbox
                cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)