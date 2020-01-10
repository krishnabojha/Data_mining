import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) + np.exp(-x))/(np.exp(x) - np.exp(-x))

def relu(x):
    if x >= 0:
        return x
    return 0

def leaky(x):
    if x >=0:
        return x
    return 0.02*x

def derivative(fun, a, h):
    return (fun(a+h) - fun(a-h))/(2*h)

print(derivative(sigmoid,1, 0.1))
n = np.arange(-5, 5, 0.1)


plt.plot(n,sigmoid(n))
plt.plot(n, derivative(sigmoid, n, 0.1))
plt.show()


plt.plot(n,tanh(n))
plt.plot(n, derivative(tanh, n, 0.1))
plt.show()

a = []
b = []
for i in n:
    a.append(relu(i))
    b.append(derivative(relu, i, 0.1))
    
plt.plot(n, a)
plt.plot(n, b)
plt.show()

c = []
d = []
for i in n:
    c.append(leaky(i))
    d.append(derivative(leaky, i, 0.5))
    
plt.plot(n, c)
plt.plot(n, d)
plt.show()


class network:
    def __init__(self,a,b,c,d):
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        
    
        





































