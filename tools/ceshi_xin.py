# -*- using coding=utf-8 -*-
# 测试envi subset from ROI的代码
import cv2
from DeepLearn.tools.gdalclip3 import GRID
import numpy as np
import glob
import os

# 最新的产生原始训练样本图片的程序
# 产生512*512的训练样本
# 程序使用：修改输出的尺寸 h_range, w_range
# 修改产生样本所在的路径

dir = r'G:\rs\labelme\data_sub'
path_list = glob.glob(os.path.join(dir, '*_sub'))
data = {}


def statistics_according(pic_data, h_range, w_range):
    global new_pic
    judge = ((pic_data[0] != 0) & (pic_data[1] != 0) & (pic_data[2] != 0))  # fasle代表不在方框中,选择roi的区域,方框外面的背景都是(0,0,0)
    c, h, w = pic_data.shape  # ENVI中的pixel定位，在没有地理参考的情况下，可能不准
    threshold = 0.15

    while True:  # 如果背景的百分比为0
        h_begin = np.random.randint(0, h - h_range)  # [low, high)
        w_begin = np.random.randint(0, w - w_range)  # [low, high)
        panding = judge[h_begin:h_begin + h_range, w_begin:w_begin + w_range]
        num = np.count_nonzero(panding == False)  # 判定符合条件的元素的个数
        percent = num / ((h_range * w_range) * 1.0)
        # panding1=(False in panding)
        if percent > 0.1:  # 不在白色方框中的比例>10%，跳出
            continue
        else:
            new_pic = pic_data[:, h_begin:h_begin + h_range, w_begin:w_begin + w_range]
            break
    return new_pic


num = 30
for item in path_list:
    im_proj, im_geotrans, im_data = GRID().read_img(item)
    pic_name = item.split('\\')[-1]
    for i in range(num):
        new_pic = statistics_according(pic_data=im_data, h_range=256, w_range=256)
        GRID().write_img(
            r'G:\rs\labelme\labelme-master\examples\semantic_segmentation_ric\sub_256_initial1\{}_{}.tif'.format(
                pic_name, i), im_proj=im_proj, im_geotrans=im_geotrans, im_data=new_pic)

# image=cv2.imread(r"E:\semantic_segmentation\labelme-master\examples\semantic_segmentation_rice\pic_fangkuang\1166_pic")
# im_proj, im_geotrans,data=GRID().read_img(filename=r"E:\semantic_segmentation\labelme-master\examples\semantic_segmentation_rice\pic_fangkuang\1207_pic")
# data=np.transpose(data,[1,2,0])
# GRID().write_img(filename=r"E:\semantic_segmentation\labelme-master\examples\semantic_segmentation_rice\pic_fangkuang\1166_pic_new.tif",im_proj=im_proj,im_geotrans=im_geotrans,im_data=data)


# 颜色异常
# cv2.imwrite(filename=r"E:\semantic_segmentation\labelme-master\examples\semantic_segmentation_rice\pic_fangkuang\1166_pic.jpg",img=data)
