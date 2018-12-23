# CNNlearn
基于经典cnn结构构建的图片分类系统_人工智能 与机器学习课堂作业
主要做了两项工作
1. 利用tensorflow对CIFAR-10 数据集的分类，包含数据集处理，训练，生成模型和测试

(1)配置好python+tensorflow即可
  可以使用下列命令获取源码
  <pre><code>git clone https://github.com/tensorflow/models.git
  
cd models/tutorials/image/cifar10

</code></pre>


cifar10_input.py	读取本地CIFAR-10的二进制文件格式的内容。
cifar10.py	        建立CIFAR-10的模型。
cifar10_train.py	在CPU或GPU上训练CIFAR-10的模型。
cifar10_multi_gpu_train.py	在多GPU上训练CIFAR-10的模型。
cifar10_eval.py	    评估CIFAR-10模型的预测性能。
具体操作 可以参考链接查看中文版的介绍  http://www.tensorfly.cn/tfdoc/tutorials/deep_cnn.html

![cifar10 model](https://github.com/xingyushu/CNNlearn/blob/master/img-folder/cifar_graph.jpg)

(2)下载数据集：
 binary格式：http://www.cs.toronto.edu/~kriz/cifar.html
 图片格式：https://pan.baidu.com/s/1skN4jW5   z6i3
 
2. Resnet网络的构建和使用
![cifar10 model](https://github.com/xingyushu/CNNlearn/blob/master/img-folder/res.jpg)

下面将要实现的是resnet-50。下面是网络模型的整体模型图。其中的CONV表示卷积层，Batch Norm表示Batch 归一化层，ID BLOCK表示Identity块，由多个层构成，具体见第二个图。Conv BLOCK表示卷积块，由多个层构成。为了使得model个结构更加清晰，才提取出了conv block 和id block两个‘块’，分别把它们封装成函数。

![cifar10 model](https://github.com/xingyushu/CNNlearn/blob/master/img-folder/res2.jpg)

上图表示Resnet-50的整体结构图

![cifar10 model](https://github.com/xingyushu/CNNlearn/blob/master/img-folder/res3.jpg)

上图表示ID block

![cifar10 model](https://github.com/xingyushu/CNNlearn/blob/master/img-folder/res4.jpg)

上图表示conv block


下载imagenet2012数据集 

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_test.tar

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar

http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_bbox_train_v2.tar

tensorflow-resnet模型： https://github.com/ry/tensorflow-resnet


其它参考：https://github.com/xvshu/ImageNet-Api  

https://github.com/liangyihuai/my_tensorflow/tree/master/com/huai/converlution/resnets

https://blog.csdn.net/liangyihuai/article/details/79140481

使用：  
<pre><code>(1)python3  train.py  训练生成模型

(2)python3  recognize.py   打开界面

</code></pre>


使用了Tensorflow，可识别物体量从10类增加到1001类，可为：狗熊 椅子 汽车 键盘 箱子 婴儿床 旗杆iPod播放器 轮船 面包车 项链 降落伞 桌子 钱包 球拍 步枪等等接着导入ResNet50网络模型进行处理，主要图像数据处理函数如下：

image.img_to_array：将PIL格式的图像转换为numpy数组。

np.expand_dims：将我们的(3，224，224)大小的图像转换为(1，3，224，224)。因为model.predict函数需要4维数组作为输入，其中第4维为每批预测图像的数量。这也就是说，我们可以一次性分类多个图像。

preprocess_input：使用训练数据集中的平均通道值对图像数据进行零值处理，即使得图像所有点的和为0。这是非常重要的步骤，如果跳过，将大大影响实际预测效果。这个步骤称为数据归一化。

model.predict：对我们的数据分批处理并返回预测值。
decode_predictions：采用与model.predict函数相同的编码标签，并从ImageNet ILSVRC集返回可读的标签。
然后通过调用官方api，得到了这个classfier，可以读取同目录下images文件夹里的指定命名的图片，这个分类器在我的笔记本tensorflow中训练差不多需要两个小时的时间，训练好之后实际识别的速率快达1~2分钟一张图，准确率高达90%以上


3. PyQt的使用

  使用教程 http://code.py40.com/pyqt5/32.html




