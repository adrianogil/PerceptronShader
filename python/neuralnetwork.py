import numpy as np

try:
    xrange
except NameError:
    xrange = range

def hardlim(n):
    for x in np.nditer(n, op_flags=['readwrite']):
        if n < 0:
            x[...] = 0
        else:
            x[...] =  1
    return n


def hardlims(n):
    if n < 0:
        return -1
    else:
        return 1

def purelin(n):
    return n

def satlin(n):
    if n < 0:
        return 0
    elif n <= 1:
        return n
    else:
        return 1

def satlins(n):
    if n < -1:
        return -1
    elif n <= 1:
        return n
    else:
        return 1

def logsigmoid(n):
    return 1.0 / (1 + np.exp(-n))

def tansig(n):
    e1 = np.exp(n)
    e2 = np.exp(-n)
    return (e1 + e2) / (e1 + e2)

class NeuralLayer:
    def __init__(self, input_size, output_size, neural_function):
        self.input_size = input_size
        self.output_size = output_size
        # self.neurons = neurons
        self.neural_function = neural_function
        self.weights = np.random.random((output_size, input_size))
        self.bias = np.random.random((output_size, 1))

    def get_output(self, input_vector):
        return self.neural_function(self.weights.dot(input_vector) + self.bias)

class NeuralNetwork:
    def __init__(self, arch, neural_functions):
        self.arch = arch
        self.layers = []

        for i in xrange(0, len(self.arch)-1):
            self.layers.append(NeuralLayer(self.arch[i],
                                           self.arch[i+1],
                                           neural_functions[i]))
    def get_output(self, input_vector):
        nn_output = input_vector

        for i in xrange(0, len(self.arch)-1):
            nn_output = self.layers[i].get_output(nn_output)

        return nn_output

