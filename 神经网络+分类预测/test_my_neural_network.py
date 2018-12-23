#!/usr/bin/env python
# coding: utf-8

# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset
# (c) Tariq Rashid, 2016
# license is GPLv2

import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
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

#开始用数据进行测试
scorecard = []
# load the mnist test data CSV file into a list
with open("mnist_dataset/mnist_test.csv", 'r') as test_data_file:
    test_data_list = test_data_file.readlines()
    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])

        # #show the manuscript data
        # img_array=numpy.asfarray(all_values[1:]).reshape([28,28])#转换为imshow（）的参数，需要是浮点型。
        # matplotlib.pyplot.imshow(img_array, cmap='Greys', interpolation='None')
        # matplotlib.pyplot.show()

        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = query(inputs,wih,who)
        #打印神经网络的输出形式
        # print(outputs)

        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)

        #print
        print('the correct num is:',correct_label,';the neural network think it as:',label)
        # append correct or incorrect to list
        if (label == correct_label):
            # #show the manuscript data
            # img_array=inputs.reshape([28,28])#转换为imshow（）的参数，需要是浮点型。
            # matplotlib.pyplot.imshow(img_array, cmap='Greys', interpolation='None')
            # matplotlib.pyplot.show()

            # print('the correct num is:',correct_label,';the neural network think it as:',label)
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            #show the manuscript data
            # img_array=inputs.reshape([28,28])#转换为imshow（）的参数，需要是浮点型。
            # matplotlib.pyplot.imshow(img_array, cmap='Greys', interpolation='None')
            # matplotlib.pyplot.show()
            
            # print('the correct num is:',correct_label,';the neural network think it as:',label)
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

# calculate the performance score, the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print ("performance = ", scorecard_array.sum() / scorecard_array.size)
