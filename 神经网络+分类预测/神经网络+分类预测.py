#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy
import scipy.special
# import matplotlib.pyplot
# %matpotlib inline

# neural network class definition
class neuralNetwork :
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) :
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # learning rate
        self.lr = learningrate
        
        # link weight matrices, wih and who
        # weight inside the array are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes,self.inodes),
        )
        self.who = numpy.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes,self.hnodes)
        )
        
        #activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    # train the neural network
    def train(self, inputs_list, targets_list) :
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from hidden layer
        final_outputs = self.activation_function(final_inputs)
        
        # error is the (targets - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weight,
        # recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs)
        )
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs)
        )
        pass
    
    # query the neural network
    def query(self, inputs_list) :
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate signals emerging from hidden layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
    
# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

#learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# load the mnist training data csv file into a list
training_data_file = open("MNIST/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# go through all records in the training data set
for record in training_data_list:
    train_all_values = record.split(",")
    #transfrom the inputs
    inputs = (numpy.asfarray(train_all_values[1:]) / 255 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(train_all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

# test the neural network
scorecard = []
test_data_file = open("MNIST/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()
for record in test_data_list:
    test_all_values = record.split(",")
    # correct answer
    correct_label = int(test_all_values[0])
    # network's answer
    inputs = (numpy.asfarray(test_all_values[1:]) / 255 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# calculate the performance score
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum()/scorecard_array.size)


# In[41]:


test_data_file = open("MNIST/mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()
all_values = test_data_list[0].split(",")
# print(all_values[0])
image_array = numpy.asfarray(all_value[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
n.query((numpy.asfarray(all_value[1:]) / 255 * 0.99) + 0.01)


# In[31]:


import numpy
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

data_file = open("MNIST/mnist_train_100.csv", 'r')
data_list = data_file.readlines()
data_file.close()

all_value = data_list[4].split(",")
image_array = numpy.asfarray(all_value[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
targets = numpy.zeros(10)+0.01
targets[int(all_value[0])] = 0.99

