#!/usr/bin/env python3
# coding:utf-8
# ---------------------------------------------------------------------------
# __author__ = 'takehiro ichimura'
# __version__ = '1.0.0'
# ---------------------------------------------------------------------------

# Third-party libraries
import numpy as np
import cupy as cp

class ESN():
    def __init__(self, input_num, reservoir_num, output_num):
        self.input_weights = cp.asarray(np.random.randn(input_num, reservoir_num)) # fixed
        self.reservoir_neuron = cp.asarray(np.random.randn(reservoir_num)) # fixed
        self.reservoir_weights = cp.asarray(np.random.randn(reservoir_num, reservoir_num)) # fixed
        self.reservoir_biases = cp.asarray(np.random.randn(reservoir_num)) # fixed
        self.output_weights = cp.asarray(np.random.randn(reservoir_num, output_num))
        self.output_biases = cp.asarray(np.random.randn(output_num))

    def output(self, input_data):
        reservoir_neuron = sigmoid(
            cp.dot(input_data, self.input_weights) + cp.dot(self.reservoir_neuron, self.reservoir_weights))
        return sigmoid(cp.dot(reservoir_neuron, self.output_weights) + self.output_biases)

    def forward(self, input_data):
        self.reservoir_neuron = sigmoid(cp.dot(input_data, self.input_weights) + cp.dot(self.reservoir_neuron, self.reservoir_weights))
        return sigmoid(cp.dot(self.reservoir_neuron, self.output_weights) + self.output_biases)

    def update(self, input_data, teach_signal, eta):
        delta_b = -eta*(self.output(cp.asarray(input_data)) - teach_signal)*sigmoid_prime(cp.dot(self.reservoir_neuron, self.output_weights) + self.output_biases)
        delta_w = delta_b * cp.reshape(self.reservoir_neuron, (self.output_weights.shape[0], self.output_weights.shape[1]))
        self.output_biases = self.output_biases + delta_b
        self.output_weights = self.output_weights + delta_w

    def cost(self, input_data, teach_signal):
        dif = self.forward(input_data) - teach_signal
        cost = (1/2)*cp.dot(dif, dif.T)
        return cost


def sigmoid(z, a=1):
    return 1.0/(1.0+cp.exp(-a*z))

#sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#sigmoid_prime_vec = np.vectorize(sigmoid_prime)