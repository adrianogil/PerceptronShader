import matplotlib
matplotlib.use('PS')

import os
import sys
import random
import numpy
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import cv2
from neuralnetwork import NeuralNetwork
import neuralnetwork as nnet

target_image = sys.argv[1]

image = cv2.imread(target_image)
image = cv2.resize(image, (100, 100))

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
        input_data = [[i * 1.0 / rows], [j * 1.0 / cols], [np.sin(20 * 3.14 * (i + j) * 1.0 / cols)]]
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

nn = NeuralNetwork([3, 8, 24], [nnet.logsigmoid, nnet.purelin])

chromo_size = 0

for i in range(0, len(nn.layers)):
    chromo_size += nn.layers[i].weights.size
    chromo_size += nn.layers[i].bias.size

# INIT
toolbox = base.Toolbox()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Definindo a estrutura do indivíduo
IND_SIZE = chromo_size  # Individual size
INT_MIN, INT_MAX = 1, 25500
FLOAT_MIN, FLOAT_MAX = -2, 2

creator.create("Individual", list, fitness=creator.FitnessMin)

# funcao para gerar o gene com alelos 0 ou 1 randomicamente uniforme
# toolbox.register("attr_int", random.randint, INT_MIN, INT_MAX)
toolbox.register("attr_flt", random.uniform, FLOAT_MIN, FLOAT_MAX)

# funcao para gerar o indivíduo (nome, forma de gerar, Estrutura, funcao geradora, tamanho)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_flt, n=IND_SIZE)

# funcao para gerar a populacao
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# funcao de fitness
def evaluate(individual):
    index = 0
    for i in range(0, len(nn.layers)):
        sx, sy = nn.layers[i].weights.shape
        for x in range(0, sx):
            for y in range(0, sy):
                nn.layers[i].weights[x, y] = individual[index]
                index += 1
        sx, sy = nn.layers[i].bias.shape
        for x in range(0, sx):
            for y in range(0, sy):
                nn.layers[i].bias[x, y] = individual[index]
                index += 1

    mse = 0

    colors = []

    for i in range(0, len(training_data)):

        nn_output = nn.get_output(training_data[i])
        if np.isnan(np.sum(nn_output)):
            continue
        # print('For epoch %s and input %s got output %s given target %s' \
        #     % (e, i, nn_output, targets[i]))
        # print("Current weights: ")
        # print(str(nn.layers[0].weights))
        # print("Current bias: ")
        # print(str(nn.layers[0].bias))
        nn_error = targets[i] - nn_output
        colors.append(nn_output)
        # print("Current error: ")
        # print(str(nn_error))
        mse = mse + np.inner(np.transpose(nn_error), np.transpose(nn_error))

    # unique_colors = numpy.unique(colors)
    # print(len(unique_colors))

    return mse

# registra funcao de fitness
toolbox.register("evaluate", evaluate)

# registra crossOver
toolbox.register("mate", tools.cxTwoPoint)

# registra mutacao com probabilidade default de mudar cada gene de 5%
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# registra o metodo de selecao como torneio de tamanho 3
toolbox.register("select", tools.selTournament, tournsize=3)

def print_ind(individual):
    print(str(individual))

#Plotar Gráfico
def plot_log(logbook):
    gen = logbook.select("gen")
    min = logbook.select("min")
    avg = logbook.select("avg")
    max = logbook.select("max")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, min, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, avg, "g-", label="Average Fitness")
    for tl in ax2.get_yticklabels():
        tl.set_color("g")

    ax3 = ax1.twinx()
    line3 = ax3.plot(gen, max, "y-", label="Maximum Fitness")
    ax3.set_ylabel("Size")
    for tl in ax3.get_yticklabels():
        tl.set_color("y")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()

def main():
    random.seed(24)

    # cria populacao inicial
    pop = toolbox.population(n=30)

    # CXPB - probabilidade de crossover
    # MUTPB - probabilidade de mutacao
    # NGEN - numero de geracoes
    CXPB, MUTPB, NGEN =0.8, 0.05, 10

    #stats a serem guardados
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)

    #Roda o algoritmo
    pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NGEN, stats=stats)

    #Seleciona o melhor individuo da populacao resultante
    best_ind = tools.selSPEA2(pop, 1)

    #Imprime as infromações do melhor individuo
    print_ind(best_ind[0])

    # Plota o Gráfico
    # plot_log(logbook)

    mse = 0

    image = cv2.imread(target_image)
    rows, cols, _ = image.shape  # Size of background Image

    new_image = numpy.zeros((rows, cols, 3))

    for i in range(rows):
        for j in range(cols):
            input_data = [[i * 1.0 / rows], [j * 1.0 / cols], [np.sin(20 * 3.14 * (i + j) * 1.0 / cols)]]
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

            # new_image[i, j, 0] = min(255, max(0, int(255 * nn_output[0])))
            # new_image[i, j, 1] = min(255, max(0, int(255 * nn_output[1])))
            # new_image[i, j, 2] = min(255, max(0, int(255 * nn_output[2])))
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
    cv2.imwrite("generated_image.png", new_image)


if __name__ == "__main__":
    main()
