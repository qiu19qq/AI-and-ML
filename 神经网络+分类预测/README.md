## 神经网络+分类预测 
- 项目描述
项目名称：神经网络+MNIST，分析神经网络的原理，并利用神经网络识别手写数字。
- 开发环境
开发语言：Python3 
开发工具：PyCharm
所需要的库：numpy,scipy,matplotlib
- 项目结构简介
项目有三个文件夹： 
1.	其中mnist_dataset文件夹中有四个文件，test结尾的表示测试集，train结尾的表示训练集，而另外两个分别是测试集与训练集的子集；
2.	my_own_images文件夹下是手写的数字图片；
3.	weight文件夹下是训练网络得到的参数，他们分别表示输入层与隐藏层，隐藏层与输出层之间的权重矩阵，其中有两个是用子集的到的；
4.	neural.py文件是一个功能完整的神经网络；
5.	read_manuscript_data.py文件是用于识别my_own_images文件夹的手写数字图片的；
6.	test_my_neural_network.py文件是利用weight文件夹下的训练网络得到的权重参数直接测试的，不需要再进行训练，因此执行速度非常快。
