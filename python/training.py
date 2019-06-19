from neuralnetwork import NeuralNetwork
import neuralnetwork as nnet

import numpy as np
import perceptron
import sys
import cv2

target_image = sys.argv[1]

image = cv2.imread(target_image)
image = cv2.resize(image, (250, 250))

training_data = []
targets = []


def int_to_bin(i):
    s = [0, 0, 0, 0, 0, 0, 0, 0]
    if i == 0:
        return s
    index = 0
    while i:
        if i & 1 == 1:
            s[index] = 1
        else:
            s[index] = 0
        index += 1
        i //= 2
    return s

rows, cols, _ = image.shape  # Size of background Image
for i in range(rows):
    for j in range(cols):
        input_data = [[i * 1.0 / rows],
            [j * 1.0 / cols],
            [np.sin(20 * 3.14 * i * 1.0 / rows)],
            [np.sin(20 * 3.14 * j * 1.0 / cols)],
            [np.sin(20 * 3.14 * (i + j) * 1.0 / (cols + rows))],
            [np.sqrt((i - 0.5) * (i - 0.5) + (j - 0.5) * (j - 0.5))]]
        r = image[i, j, 0] * 1.0 / 255
        g = image[i, j, 1] * 1.0 / 255
        b = image[i, j, 2] * 1.0 / 255
        m = (r + g + b) / 3
        # target_data = [[2 * (r - m) - 1],
        #                [2 * (g - m) - 1],
        #                [2 * (b - m) - 1],
        #                [2 * m - 1]]
        red_bin = int_to_bin(image[i, j, 0])
        green_bin = int_to_bin(image[i, j, 1])
        blue_bin = int_to_bin(image[i, j, 2])
        # target_data = [[r],
        #                [g],
        #                [b]]
        target_data = [[red_bin[0] * 1.0],
                       [red_bin[1] * 1.0],
                       [red_bin[2] * 1.0],
                       [red_bin[3] * 1.0],
                       [red_bin[4] * 1.0],
                       [red_bin[5] * 1.0],
                       [red_bin[6] * 1.0],
                       [red_bin[7] * 1.0],
                       [green_bin[0] * 1.0],
                       [green_bin[1] * 1.0],
                       [green_bin[2] * 1.0],
                       [green_bin[3] * 1.0],
                       [green_bin[4] * 1.0],
                       [green_bin[5] * 1.0],
                       [green_bin[6] * 1.0],
                       [green_bin[7] * 1.0],
                       [blue_bin[0] * 1.0],
                       [blue_bin[1] * 1.0],
                       [blue_bin[2] * 1.0],
                       [blue_bin[3] * 1.0],
                       [blue_bin[4] * 1.0],
                       [blue_bin[5] * 1.0],
                       [blue_bin[6] * 1.0],
                       [blue_bin[7] * 1.0]]

        training_data.append(input_data)
        targets.append(target_data)

nn = NeuralNetwork([6, 24], [nnet.purelin])

# input_samples = [[[0], [0]], [[0], [2]], [[2], [1]], [[3], [2]]]
# targets = [[0], [0], [1], [1]]

perceptron.learn(nn, training_data, targets, epoches=2)

print(str(nn.layers[0].weights))
print(str(nn.layers[0].bias))

mse = 0

image = cv2.imread(target_image)
rows, cols, _ = image.shape  # Size of background Image

new_image = np.zeros((rows, cols, 3))

for i in range(rows):
    for j in range(cols):
        input_data = [[i * 1.0 / rows],
                      [j * 1.0 / cols],
                      [np.sin(20 * 3.14 * i * 1.0 / rows)],
                      [np.sin(20 * 3.14 * j * 1.0 / cols)],
                      [np.sin(20 * 3.14 * (i + j) * 1.0 / cols)]]

        r = image[i, j, 0] * 1.0 / 255
        g = image[i, j, 1] * 1.0 / 255
        b = image[i, j, 2] * 1.0 / 255
        m = (r + g + b) / 3
        # target_data = [[r],
        #                [g],
        #                [b]]
        red_bin = int_to_bin(image[i, j, 0])
        green_bin = int_to_bin(image[i, j, 1])
        blue_bin = int_to_bin(image[i, j, 2])
        # target_data = [[r],
        #                [g],
        #                [b]]
        target_data = [[red_bin[0] * 1.0],
                       [red_bin[1] * 1.0],
                       [red_bin[2] * 1.0],
                       [red_bin[3] * 1.0],
                       [red_bin[4] * 1.0],
                       [red_bin[5] * 1.0],
                       [red_bin[6] * 1.0],
                       [red_bin[7] * 1.0],
                       [green_bin[0] * 1.0],
                       [green_bin[1] * 1.0],
                       [green_bin[2] * 1.0],
                       [green_bin[3] * 1.0],
                       [green_bin[4] * 1.0],
                       [green_bin[5] * 1.0],
                       [green_bin[6] * 1.0],
                       [green_bin[7] * 1.0],
                       [blue_bin[0] * 1.0],
                       [blue_bin[1] * 1.0],
                       [blue_bin[2] * 1.0],
                       [blue_bin[3] * 1.0],
                       [blue_bin[4] * 1.0],
                       [blue_bin[5] * 1.0],
                       [blue_bin[6] * 1.0],
                       [blue_bin[7] * 1.0]]

        nn_output = nn.get_output(input_data)
        if np.isnan(np.sum(nn_output)):
            continue
        # print('For epoch %s and input %s got output %s given target %s' \
        #     % (e, i, nn_output, targets[i]))
        # print("Current weights: ")
        # print(str(nn.layers[0].weights))
        # print("Current bias: ")
        # print(str(nn.layers[0].bias))
        nn_error = target_data - nn_output
        # print("Current error: ")
        # print(str(nn_error))
        mse = mse + np.inner(np.transpose(nn_error), np.transpose(nn_error))

        def norm_data(x):
            if x > 0.5:
                return 1
            else:
                return 0

        red = norm_data(nn_output[0]) + norm_data(nn_output[1]) * 2 + norm_data(nn_output[2]) * 4 + \
            norm_data(nn_output[3]) * 8 + norm_data(nn_output[4]) * 16 + norm_data(nn_output[5]) * 32 + \
            norm_data(nn_output[6]) * 64 + norm_data(nn_output[7]) * 128

        blue = norm_data(nn_output[8]) + norm_data(nn_output[9]) * 2 + norm_data(nn_output[10]) * 4 + \
            norm_data(nn_output[11]) * 8 + norm_data(nn_output[12]) * 16 + norm_data(nn_output[13]) * 32 + \
            norm_data(nn_output[14]) * 64 + norm_data(nn_output[15]) * 128

        green = norm_data(nn_output[16]) + norm_data(nn_output[17]) * 2 + norm_data(nn_output[18]) * 4 + \
            norm_data(nn_output[19]) * 8 + norm_data(nn_output[20]) * 16 + norm_data(nn_output[21]) * 32 + \
            norm_data(nn_output[22]) * 64 + norm_data(nn_output[23]) * 128

        new_image[i, j, 0] = int(min(255, max(0, 255 * red)))
        new_image[i, j, 1] = int(min(255, max(0, 255 * blue)))
        new_image[i, j, 2] = int(min(255, max(0, 255 * green)))

print("Error on testing dataset: " + str(mse))
cv2.imwrite("generated_image_nn.png", new_image)
