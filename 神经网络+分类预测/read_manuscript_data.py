import scipy.misc
import numpy
import matplotlib.pyplot

#神经网络：由输入得到输出,wih为输入层与隐藏层之间的权重，
def query(inputs_list,wih,who):
    # convert inputs list to 2d array
    inputs = numpy.array(inputs_list, ndmin=2).T
    
    # calculate signals into hidden layer
    hidden_inputs = numpy.dot(wih, inputs)
    # calculate the signals emerging from hidden layer
    hidden_outputs = scipy.special.expit(hidden_inputs)
    
    # calculate signals into final output layer
    final_inputs = numpy.dot(who, hidden_outputs)
    # calculate the signals emerging from final output layer
    final_outputs = scipy.special.expit(final_inputs)#sigmoid函数
    return final_outputs

#加载训练好的模型的权重
wih=numpy.loadtxt('weight/weight_inputs_hiddens.csv',delimiter=",")
who=numpy.loadtxt('weight/weight_hiddens_outputs.csv',delimiter=",")

#设置参数
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
# learning rate
learning_rate = 0.1

#read data
img_array=scipy.misc.imread('my_own_images/2828_my_own_6i.png',flatten=True)#需要Pillow库的支持

#处理data
img_data=255.0-img_array.reshape(784)
img_array=img_data.reshape([28,28])#用于imshow()的参数
img_data=(img_data/255.0*0.99)+0.01#变成MNIST数据集的格式
# print(img_array)
# print(img_data)

#show the manuscript data
matplotlib.pyplot.imshow(img_array, cmap='Greys', interpolation='None')
matplotlib.pyplot.show()

#使用神经网络
outputs = query(img_data,wih,who)
# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)

#print
print('the neural network think it as:',label)


