#!/usr/bin/env python
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier, modified to display data in TensorBoard.

See extensive documentation for the original model at
http://tensorflow.org/tutorials/mnist/beginners/index.md

See documentation on the TensorBoard specific pieces at
http://tensorflow.org/how_tos/summaries_and_tensorboard/index.md

If you modify this file, please update the exerpt in
how_tos/summaries_and_tensorboard/index.md.

"""
from __future__ import absolute_import#加入绝对引入这个新特性
from __future__ import division  #精确除法
from __future__ import print_function  #即使在python2.X，使用print就得像python3.X那样加括号使用

import tensorflow.python.platform
import input_data
import tensorflow as tf

#执行main函数之前首先进行flags的解析，也就是说TensorFlow通过设置flags来传递tf.app.run()所需要的参数，
#我们可以直接在程序运行前初始化flags，也可以在运行程序的时候设置命令行参数来达到传参的目的。
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')


def main(_):
  # Import data
  mnist = input_data.read_data_sets('Mnist_data/', one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()

  # Create the model   
  # x（图片的特征值）：这里使用了一个28*28=784列的数据来表示一个图片的构成，784个像素点对应784个数  因此输入层是784个神经元 输出层是10个神经元
 # w（特征值对应的权重）：
 # b  （偏置量）：
# 使用 tf.Variable 创建一个变量，然后使用 tf.zeros 将变量 W 和 b 设为值全为0的张量（就是将张量中的向量维度值设定为0）。由于我们在后续的过程中要使用大量的数据来训练 W 和 b 的值，因此他们的初始值是什么并不重要。
# w的形状是一个[784,10]的张量，第一个向量列表表示每个图片都有784个像素点，第二个向量列表表示从“0”到“9”一共有10类图片。所以w用一个784维度的向量表示像素值，用10维度的向量表示分类，而2个向量进行乘法运算（或者说2个向量的笛卡尔集）就表示“某个像素在某个分类下的证据”。
#b的形状是[10]，他仅表示10个分类的偏移值。
  x = tf.placeholder(tf.float32, [None, 784], name='x-input')
  W = tf.Variable(tf.zeros([784, 10]), name='weights')
  b = tf.Variable(tf.zeros([10], name='bias'))

  
  #tf.name_scope主要是给表达式命名，给节点命名
  with tf.name_scope('Wx_b'):
    # y 预测值，softmax相当于sigmoid函数
    y = tf.nn.softmax(tf.matmul(x, W) + b)

  # 在tensorboard中histogram模块中将变量可视化：分布图、直方图
  _ = tf.summary.histogram('weights', W)#可视化观看变量：添加任意shape的Tensor，统计这个Tensor的取值分布。
  _ = tf.summary.histogram('biases', b)#可视化观看变量：添加任意shape的Tenso r，统计这个Tensor的取值分布。
  _ = tf.summary.histogram('y', y)#可视化观看变量：添加任意shape的Tensor，统计这个Tensor的取值分布。

  # 定义偏差率和优化器
  # y_（真实结果）：来自MNIST的训练集，每一个图片所对应的真实值，如果是6，则表示为：[0 0 0 0 0 1 0 0 0]
  y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
  # 更多的名称范围将清理图形表示
  with tf.name_scope('xent'):
    #交叉熵评估代价
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    _ = tf.summary.scalar('cross entropy', cross_entropy)
  with tf.name_scope('train'):
     #梯度下降算法的一种
    train_step = tf.train.GradientDescentOptimizer(  
        FLAGS.learning_rate).minimize(cross_entropy)

  with tf.name_scope('test'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _ = tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs合并所有的摘要并将它们写到/tMP/Mistist日志中
  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter('C:/Users/YS/Documents/course/mnist-master/mnist-master/mnist-master/mnist_logs', sess.graph_def)
  tf.initialize_all_variables().run()

  # Train the model, and feed in test data and record summaries every 10 steps训练模型，并输入试验数据，每10步记录总结 
  #初始化变量，开始迭代执行。每隔10次，使用测试集验证一次，sess.run()的输入，merged合并的Log信息，accuracy计算图，feed数据，  将信息写入test_writer。
  #每隔99步，将运行时间与内存信息，存入Log中，其余步骤正常秩序，添加存储信息。
  for i in range(FLAGS.max_steps):     #训练阶段，迭代最大次数
    if i % 10 == 0:  #每训练100次，测试一次
        #损失函数（交叉熵）和梯度下降算法，通过不断的调整权重和偏置量的值，
        #来逐步减小根据计算的预测结果和提供的真实结果之间的差异，以达到训练模型的目的。
      if FLAGS.fake_data:
        #fill_feed_dict函数会查询给定的DataSet，索要下一批次batch_size的图像和标签，
        #与占位符相匹配的Tensor则会包含下一批次的图像和标签。
        batch_xs, batch_ys = mnist.train.next_batch(
            100, fake_data=FLAGS.fake_data)    #按批次训练，每批100行数据
        feed = {x: batch_xs, y_: batch_ys}  # 用训练数据替代占位符来执行训练
      else:
        feed = {x: mnist.test.images, y_: mnist.test.labels} # 用测试数据替代占位符来执行训练
      result = sess.run([merged, accuracy], feed_dict=feed)  
      summary_str = result[0]
      acc = result[1]
      writer.add_summary(summary_str, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:
      batch_xs, batch_ys = mnist.train.next_batch(
          100, fake_data=FLAGS.fake_data)
      feed = {x: batch_xs, y_: batch_ys}
      sess.run(train_step, feed_dict=feed)

if __name__ == '__main__':
  tf.app.run()

#TensorFlow 基本使用https://blog.csdn.net/yhl_leo/article/details/50619029
#Tensorflow MNIST 数据集测试代码入门https://blog.csdn.net/yhl_leo/article/details/50614444
#Tensorflow之MNIST解析https://blog.csdn.net/qq546542233/article/details/77836328?utm_source=blogxgwz1
#TensorFlow学习笔记（十二）TensorFLow tensorBoard 总结https://blog.csdn.net/qq_36330643/article/details/76709531#commentBox
#Tensorflow的应用（四）https://blog.csdn.net/yyxyyx10/article/details/78695531


