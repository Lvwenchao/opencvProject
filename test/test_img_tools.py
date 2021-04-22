# AUTHOR    ：Lv Wenchao
# coding    : utf-8
# @Time     : 2021/4/14 21:41
# @FileName : test_img_tools.py
# @Software : PyCharm
from src.tools import img_tools
import glymur
import os
from PIL import Image
import re
import glymur
import numpy as np

FOLDER = "../resources/image"
FILENAME = "DJI_0021.jp2"
PATH = os.path.join(FOLDER, FILENAME)


class TestImageTools:

    def test_jpeg_2000(self):
        img = Image.open(PATH)
        # img.save("../resources/image/DJI_0021.jp2")
        # jp2k = glymur.Jp2k(PATH)
        # codeStream = jp2k.get_codestream()
        # print(codeStream.segment[1])
        print(np.asarray(img).shape)

