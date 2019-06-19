import numpy as np
import math
import random

try:
    xrange
except NameError:
    xrange = range

def learn(nn, input_samples, targets, epoches=10, min_mse=1e-8, max_mse=1e+10):

    if len(nn.layers) > 1:
        print("Error: " + \
            "perceptron learning rule can be applied only in one layer architectures!")
        return

    if len(input_samples) != len(targets):
        print("Error: " + \
            "Input and Targets list should have the same size!")
        return

    alpha = 0.15

    for e in xrange(0, epoches):

        mse = 0
        for i in xrange(0, len(input_samples)):
            if random.randint(0, 100) > 35:
                continue

            nn_output = nn.get_output(input_samples[i])
            if np.isnan(np.sum(nn_output)):
                continue
            print('For epoch %s and input %s got output %s given target %s' \
                % (e, i, nn_output, targets[i]))
            print("Current weights: ")
            print(str(nn.layers[0].weights))
            print("Current bias: ")
            print(str(nn.layers[0].bias))
            nn_error = targets[i] - nn_output
            print("Current error: ")
            print(str(nn_error))
            mse = mse + np.inner(np.transpose(nn_error), np.transpose(nn_error))

            new_weights = nn.layers[0].weights + \
                alpha * np.dot(nn_error, np.transpose(input_samples[i]))
            if not np.isnan(np.sum(new_weights)):
                nn.layers[0].weights = new_weights
            new_bias = nn.layers[0].bias + alpha * nn_error
            if not np.isnan(np.sum(new_bias)):
                nn.layers[0].bias = new_bias

        print('For epoch %s got MSE %s' % (e, mse))
        if np.isnan(np.sum(mse)):
            print('At epoch %s training achieved MSE as NaN' % (e,))
            return
        if mse < min_mse:
            print('At epoch %s training achieved MSE min limit: %s' % (e, mse))
            return
        if mse > max_mse:
            print('At epoch %s training achieved MSE max limit: %s' %(e, mse))
            return



