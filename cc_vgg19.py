# -*- coding:utf-8 -*-
import tensorflow as tf
import utils
import numpy as np
import skimage

class cc_vgg19_class:
    def __init__(self,vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.datadict = np.load(vgg19_npy_path,encoding='latin1').item()
        else:
            self.datadict = None
        self.var_dict = dict()
        self.trainable = trainable
        self.dropout = dropout
        print('初始化成功')

    def build_network(self, image):
        """
        建立网络，image是输入的图片矩阵
        :param image:
        :return:
        """
        self.conv1_1 = self.conv_layer(bottom=image,in_channels=3,out_channels=64,name='conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pooling(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pooling(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        # self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pooling(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        # self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")

        self.conv5_1 = self.conv_layer(self.conv4_3, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")

        # self.conv6_1 = self.conv_layer(self.conv5_3, 512, 256, "conv6_1")
        # self.conv6_2 = self.conv_layer(self.conv6_1, 256, 128, "conv6_2")
        # self.conv6_3 = self.conv_layer(self.conv6_2, 128, 64, "conv6_3")

        # self.densitymap_predict = self.conv_layer(self.conv6_3, 64, 1, "conv7_1")
        self.densitymap_predict = self.conv_layer(self.conv5_3, 512, 1, "predict")
        # 这一句要更改，暂时用conv1_1代替
        # self.densitymap_predict = self.conv7_1
        print('建立网络成功')

    # 给定name和输入计算卷积和relu结果
    def conv_layer(self, bottom, in_channels, out_channels, name):
        # 管理op变量名，mul_result.name 形如 conv1_1/conv_2d
        with tf.variable_scope(name):
            # filter size = 3*3
            filt, bias = self.get_conv_layer_parameters(3, in_channels, out_channels, name)
            mul_result = tf.nn.conv2d(bottom, filt, [1,1,1,1],padding='SAME')
            conv_result = tf.nn.bias_add(mul_result, bias)
            relu_result = tf.nn.relu(conv_result)
            return relu_result

    # 返回最大池化结果
    def max_pooling(self, bottom, name):
        # 将形如max_pool1的name存到模型中
        return tf.nn.max_pool(bottom,ksize=[1,2,2,1],strides=[1, 2, 2, 1],padding='SAME',name=name)
        # return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 获取给定name的filter和bias值，如果目前模型里没有，则使用随机初始化的值，均值为0，方差为0.001
    def get_conv_layer_parameters(self, filter_size, in_channels, out_channels, layer_name):
        random_initial_values = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filter_parameters = self.GetParameterFromDatadict(layer_name, 0, random_initial_values, layer_name+'_filter')

        random_initial_values = tf.truncated_normal([out_channels], .0, .001)
        bias_parameters = self.GetParameterFromDatadict(layer_name, 1, random_initial_values,layer_name+'_bias')
        return filter_parameters, bias_parameters

    # 从加载的模型字典中获取给定name的参数
    def GetParameterFromDatadict(self, name, idx, random_initial_values,var_name):
        if self.datadict is not None and name in self.datadict:
            value = self.datadict[name][idx]
        else:
            value = random_initial_values
        # 将形如conv1_1_filter存到模型中
        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)
        self.var_dict[(name,idx)] = var
        assert random_initial_values.get_shape() == var.get_shape()
        return var

    # save model to npy file
    def SaveModel(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess,tf.Session)
        model_dict = dict()
        for (name, idx), var in list(self.var_dict.items()):
            # 执行后才有输出
            var_out = sess.run(var)
            if name not in model_dict:
                model_dict[name] = {}
            model_dict[name][idx] = var_out

        np.save(npy_path, model_dict)
        print('model saved to', npy_path)
        return npy_path