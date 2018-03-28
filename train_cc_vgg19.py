# -*- coding:utf-8 -*-
import tensorflow as tf
import utils
import numpy as np
from skimage import io, transform
import cc_vgg19

train_imgs = list()
import os

# # 读取某个目录下所有训练集图片，比原图放大了8倍
# g = os.walk("D:\\Data\\train_A_patches_9")
# for path,d,filelist in g:
#     for filename in filelist:
#         img = io.imad(reos.path.join(path,filename))
#         train_imgs.append(img)
#         # if filename.endswith('jpeg'):
#         #     print (os.path.join(path, filename))

image_raw_data = io.imread('./test_data/1_4.jpg')
size = list(image_raw_data.shape)
if len(size)==2:
    size = np.concatenate((size, [1]),0)
size = np.concatenate(([1],size),0)
# image_raw_data = tf.gfile.FastGFile('D:\\Data\\train_A_patches_9\\4_6.jpg', 'rb').read()
# img_data = tf.image.decode_jpeg(image_raw_data)
# print(img_data.eval())
img_0 = image_raw_data.reshape(size)
# gt_densitymaps = list()
# gt_d = os.walk("D:\\人群密度识别\\crowdcount-mcnn-master\\data\\formatted_trainval\\shanghaitech_part_A_patches_9\\train_den")
# for path,d,filelist in gt_d:
#     for filename in filelist:
#         densitymap = np.loadtxt(open(os.path.join(path,filename), "rb"), delimiter=",", skiprows=0)
#         # densitymap = io.imread(os.path.join(path,filename))
#         gt_densitymaps.append(densitymap)

with tf.device('/cpu:0'):
    sess = tf.Session()
    # 输入和输出分辨率没有规则化，所以不指定shape
    images = tf.placeholder(tf.float32)
    train_groundtruth = tf.placeholder(tf.float32)
    # 已经训练好的参数文件路径
    vgg19_npy_path = './vgg19.npy'
    # 初始化指定是否为训练模式
    CSRNet = cc_vgg19.cc_vgg19_class(vgg19_npy_path, trainable = True)

    CSRNet.build_network(images)

    # 全局初始化这一行应该在所有变量声明之后
    sess.run(tf.global_variables_initializer())
    print('全局初始化完毕')
    # print(sess.run(CSRNet.pool1, feed_dict={images: img_0}))
    result_0 = sess.run(CSRNet.densitymap_predict, feed_dict={images: img_0})
    result_shape = result_0.shape[1:-1]
    result_0.reshape(result_shape)
    # print(result_0)
    print(result_0.sum())
    # # loss函数的定义
    # loss = tf.reduce_sum((CSRNet.densitymap_predict - train_groundtruth) ** 2)
    # # learning rate 0.0001
    # train = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
    # # 训练模型
    # for num,train_img,gt_densitymap in enumerate(zip(train_imgs,gt_densitymaps)):
    #     sess.run(train,feed_dict={images:train_img, train_groundtruth:gt_densitymap})
    #     if num%10==0:
    #         print('finish %d'%num)
    # # 把新的参数保存到指定目录下
    # CSRNet.SaveModel(sess, "D:\\Data\\npy\\vgg19-new.npy")