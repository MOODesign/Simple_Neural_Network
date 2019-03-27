import numpy as np
import pandas as pd

data = {'Inputs' : ['0 0 1','1 1 1','1 0 1','0 1 1','1 0 0'], 'Output' : [0,1,1,0,'?']}
df = pd.DataFrame(data, index = ['x 0', 'x 1', 'x 2', 'x 3', 'x new'])
df

import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

def sigmoid_derivative(x):
    b = []
    for item in x:
        b.append(item*(1-item))
    return b

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)
derivsig = sigmoid_derivative(sig)
plt.plot(x,sig)
plt.show()

plt.plot(x,derivsig)
plt.grid(True)
plt.show()

#sigmoid function.
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input dataset.
X = np.array([  [0,0,1],[0,1,1],[1,0,1],[1,1,1]])

#output dataset.
y = np.array([[0,0,1,1]]).T

#seeding random numbers to make calculation.
np.random.seed(1)

#initializing weights randomly with mean 0
synaptic_weights = 2*np.random.random((3,1)) - 1

for iter in range(10000):

    #forward propagation.
    l0 = X
    l0_dot_weights = np.dot(l0,synaptic_weights)
    l1 = nonlin(l0_dot_weights)

    #loss?
    l1_error = y - l1

    #delta error.
    l1_delta = l1_error * nonlin(l1,True)

    #updating weights.
    synaptic_weights = synaptic_weights + np.dot(l0.T,l1_delta)

    if iter%2000 == 0:
        print('Iteration Number ', iter)
        print('Sigmoid Output ', l1)
        print('Error = ',l1_error)
        print('Derivative =', nonlin(l1,True))
        print('l1_delta =' , l1_delta)
        print('Weights_delta =', np.dot(l0.T, l1_delta))

print("Output After Training:")
print(l1)


#testing the model with new data.
new_x = np.array([1,0,0])
l0_dot_weights = np.dot(new_x,synaptic_weights)
y = nonlin(l0_dot_weights)
print('Y is ', y)
