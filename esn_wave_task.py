import random
import sys
# Third-party libraries
import numpy as np
import esn as esn
import matplotlib.pyplot as plt
from tqdm import tqdm


def rand_gen(num, u_bar, r):
    test_data = []
    for i in tqdm(range(num)):
        if np.random.rand() < r:
            wave = [[u_bar+1]]
        else:
            wave = [[u_bar-1]]
        test_data = test_data + wave
    return test_data


def main():
    num = 10000
    reservoir_num = 50
    s = np.sqrt(1)
    u_bar = 0.5
    r = 0.2

    """
    test_data =[]
    for i in tqdm(range(num/8)):
        wave = [[0], [1], [0], [0], [1], [1], [0], [1]]
        test_data = test_data + wave
    """
    wave = [[0], [1], [0], [0], [1], [1], [0], [1]]  # fake
    test_data = rand_gen(num*len(wave), u_bar, r)
    input_data = np.array(test_data)

    del(test_data[0])
    test_data = test_data + [[0]]
    teach_signal = test_data
    teach_signal = np.array(teach_signal)

    net = esn.ESN(1, reservoir_num, 1, s)

    x = np.array(range(num*len(wave)))
    out = np.zeros(num*len(wave))
    cost = np.zeros(num*len(wave))
    reservoir_plot = np.zeros((len(wave)*10, reservoir_num))

    for i in tqdm(range(num*len(wave))):
        #print("epoch:"+str(i))
        #print(net.forward(input_data[i]))
        out[i] = net.forward(input_data[i])
        cost[i] = net.cost(input_data[i], teach_signal[i])
        if i < len(wave)*10:reservoir_plot[i] = net.reservoir_neuron
        #print(cost[i])
        net.update(input_data[i], teach_signal[i], eta=0.3)

    plt.figure()
    plt.plot(x[-len(wave)*10:], out[-len(wave)*10:])
    plt.plot(x[-len(wave)*10:], teach_signal[-len(wave)*10:])

    plt.figure()
    plt.plot(x, cost)

    plt.figure()
    plt.imshow(reservoir_plot.T, cmap='gray')
    plt.colorbar()

    plt.show()

main()