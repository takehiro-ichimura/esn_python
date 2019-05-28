#!/usr/bin/env python3
# coding:utf-8
# ---------------------------------------------------------------------------
# __author__ = 'takehiro ichimura'
# __version__ = '1.0.0'
# ---------------------------------------------------------------------------

# Third-party libraries
import numpy as np

# if you deal with more than 10000 reservoir neuron, you should use esn_gpu.py
# reference: Real-Time Computation at the Edge of Chaos in Recurrent Neural Networks / Nils Bertschinger (2004)


class ESN():
    def __init__(self, input_num, reservoir_num, output_num, s):
        #self.input_weights = np.random.randn(input_num, reservoir_num)  # fixed
        self.input_weights = np.ones((input_num, reservoir_num))  # fixed
        self.reservoir_neuron = np.random.randn(reservoir_num)  # fixed
        self.reservoir_weights = np.random.normal(0, s, (reservoir_num, reservoir_num))  # fixed
        #self.reservoir_biases = np.random.normal(0, 1, (reservoir_num))  # fixed
        #self.reservoir_biases = np.ones(reservoir_num)  # fixed
        self.reservoir_biases = np.zeros(reservoir_num)  # fixed
        self.output_weights = np.random.randn(reservoir_num, output_num)
        self.output_biases = np.random.randn(output_num)

    def output(self, input_data):
        reservoir_neuron = step_tanh_vec(
            np.dot(input_data, self.input_weights) + np.dot(self.reservoir_neuron, self.reservoir_weights))
        return sigmoid_vec(np.dot(reservoir_neuron, self.output_weights) + self.output_biases)

    def forward(self, input_data):
        self.reservoir_neuron = step_tanh_vec(np.dot(input_data, self.input_weights) + np.dot(self.reservoir_neuron, self.reservoir_weights))
        return sigmoid_vec(np.dot(self.reservoir_neuron, self.output_weights) + self.output_biases)

    def update(self, input_data, teach_signal, eta):
        delta_b = -eta*(self.output(input_data) - teach_signal)*sigmoid_prime_vec(np.dot(self.reservoir_neuron, self.output_weights) + self.output_biases)
        delta_w = delta_b * np.reshape(self.reservoir_neuron, (self.output_weights.shape[0], self.output_weights.shape[1]))
        self.output_biases = self.output_biases + delta_b
        self.output_weights = self.output_weights + delta_w

    def cost(self, input_data, teach_signal):
        dif = self.forward(input_data) - teach_signal
        cost = (1/2)*np.dot(dif, dif.T)
        return cost


def sigmoid(z, a=1):
    return 1.0/(1.0+np.exp(-a*z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

def tanh(z):
    return np.tanh(z)

tanh_vec = np.vectorize(tanh)

def step_tanh(z):
    if z >= 0:
        return 1
    else:
        return -1

step_tanh_vec = np.vectorize(step_tanh)