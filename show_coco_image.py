# coding=UTF-8

#coco数据集标注的显示或者测试文件，如果跑不过不话，一般就是数据集制作的时候标注有问题。

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import os
from skimage.io import imsave
import numpy as np
import mmcv

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

img_path = './data/rubbish/train/'
annFile = './data/rubbish/annotations/rubbish_train2.json'


# img_path = 'data/coco/val2017'
# annFile = 'data/coco/annotations/instances_val2017.json'

if not os.path.exists('test_image_out/'):
    os.makedirs('test_image_out/')

result_path = './test_image_out/'

# 初始化标注数据的 COCO api
coco = COCO(annFile)

#获取并显示coco分类名字和超类名字
cats = coco.loadCats(coco.getCatIds())
catNms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(catNms)))

supNms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(supNms)))

#获得分类的ID
catIds_1 = coco.getCatIds(catNms=catNms, supNms=supNms)
print(catIds_1)
#获得所有图片ID
imgIds_1 = coco.getImgIds()
print(imgIds_1)
#打印图片数量
print(len(imgIds_1))
#循环处理每一张图片
for i in range(len(imgIds_1)):
    img1 = coco.loadImgs(imgIds_1[i])[0]
    image_name = img1['file_name']
    print(image_name)
    #使用mmvc这个库来读取图片是为了后期显示mask。因为我使用的是mmdetection这个库。所以习惯mmcv。当然也可以换成其他的方法。
    img = mmcv.imread(os.path.join(img_path, image_name))
    img = img.copy()

    # catIds=[] 说明展示所有类别的box，也可以指定某个或某些类别
    #获得当前图片里面符合目标catIds的标注的id序号。
    annIds = coco.getAnnIds(imgIds=img1['id'], catIds=[], iscrowd=None)
    print(annIds)
    #加载该id对应的标注
    anns = coco.loadAnns(annIds)
    #显示标注，但是很明显，并没有起作用，还是得自己写显示的部分。
    coco.showAnns(anns)
    print(anns)

    #自己参考mmdetection库里面的显示mask的方式，进行改写的代码。
    masks = []
    showonce = True
    for ann in anns:
        if type(ann['segmentation']) == list and showonce:
            print(ann['segmentation'])
            showonce = False
        if type(ann['segmentation']) != list:
            print(ann['segmentation'])
        mask = coco.annToMask(ann).astype(np.bool)
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        print(mask.shape)
        print(img.shape)
        img[mask] = img[mask] * 0.5 + color_mask * 0.5  # 在图片上修改像素值，画出掩码
        # cv2.imshow('mask_image', img)#显示
        # cv2.waitKey(10) #必须有，不然上一句的显示不起效果。


    #下面是画出标注的方框的代码，如果需要显示方框。
    coordinates = []
    for j in range(len(anns)):
        coordinate = []
        coordinate.append(anns[j]['bbox'][0])
        coordinate.append(anns[j]['bbox'][1] + anns[j]['bbox'][3])
        coordinate.append(anns[j]['bbox'][0] + anns[j]['bbox'][2])
        coordinate.append(anns[j]['bbox'][1])
        # print(coordinate)
        left = np.rint(coordinate[0])
        right = np.rint(coordinate[1])
        top = np.rint(coordinate[2])
        bottom = np.rint(coordinate[3])
        # 左上角坐标, 右下角坐标
        cv2.rectangle(img,
                      (int(left), int(right)),
                      (int(top), int(bottom)),
                      (0, 255, 0),
                      2)
    cv2.imshow('result_image', img)  # 显示
    cv2.imwrite(os.path.join(result_path, image_name),img=img)
    cv2.waitKey(10)  # 必须有，不然上一句的显示不起效果。