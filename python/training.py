from neuralnetwork import NeuralNetwork
import neuralnetwork as nnet

import perceptron
import sys
import cv2

target_image = sys.argv[1]

image = cv2.imread(target_image)
image = cv2.resize(image, (200, 200))

training_data = []
targets = []

rows, cols, _ = image.shape  # Size of background Image
for i in range(rows):
    for j in range(cols):
        input_data = [[i * 1.0 / rows], [j * 1.0 / cols]]
        target_data = [[image[i, j, 0] * 1.0 / 255],
                       [image[i, j, 1] * 1.0 / 255],
                       [image[i, j, 2] * 1.0 / 255]]

        training_data.append(input_data)
        targets.append(target_data)

nn = NeuralNetwork([2, 3], [nnet.tansig])

# input_samples = [[[0], [0]], [[0], [2]], [[2], [1]], [[3], [2]]]
# targets = [[0], [0], [1], [1]]

perceptron.learn(nn, training_data, targets)
import pdb; pdb.set_trace() # Start debugger