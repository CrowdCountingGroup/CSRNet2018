# -*- coding:utf-8 -*-

from skimage import io,transform
import os

# 读取某个目录下所有训练集图片
g = os.walk("D:\\人群密度识别\\crowdcount-mcnn-master\\data\\formatted_trainval\\shanghaitech_part_B_patches_9\\train")
save_root_path = "D:\\Data\\train_B_patches_9"
for path, d, filelist in g:
    for filename in filelist:
        img = io.imread(os.path.join(path, filename))
        img_new = transform.rescale(img, 8, 1)
        save_path = os.path.join(save_root_path, filename)
        io.imsave(save_path, img_new)