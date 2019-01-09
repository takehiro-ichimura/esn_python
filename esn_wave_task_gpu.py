import random
import sys
# Third-party libraries
import numpy as np
import cupy as cp
import esn_gpu as esn
import matplotlib.pyplot as plt
from tqdm import tqdm

test_data = []
num = 1000

for i in tqdm(range(num)):
    wave = [[1], [0], [1], [0], [1], [1], [1], [1]]
    test_data = test_data + wave
input_data = cp.array(test_data)

del(test_data[0])
test_data = test_data + [[0]]
teach_signal = test_data
teach_signal = cp.asarray(teach_signal)

net = esn.ESN(1, 8, 1)

x = np.array(range(num*len(wave)))
out = cp.zeros(num*len(wave))
cost = cp.zeros(num*len(wave))

for i in tqdm(range(num*len(wave))):
    #print("epoch:"+str(i))
    #print(net.forward(input_data[i]))
    out[i] = net.forward(input_data[i])[0]
    cost[i] = net.cost(input_data[i], teach_signal[i])
    #print(cost[i])
    net.update(input_data[i], teach_signal[i], eta=0.8)

out = cp.asnumpy(out)
cost = cp.asnumpy(cost)
teach_signal = cp.asnumpy(teach_signal)

plt.figure()
plt.plot(x, out)
plt.plot(x, teach_signal)

plt.figure()
plt.plot(x, cost)

plt.show()