from numpy import tanh
from numpy.random import uniform
from autograd import *

class Neuron:
    def __init__(self,n):
        self.weights=[]
        for i in range(n):
            self.weights.append(Value(uniform(low=-1,high=1)))
        self.bias = Value(uniform(low=-1,high=1))
    def forward(self,inputs):
        if len(self.weights)!=len(inputs):
            print("No of inputs not correct.")
            raise ValueError
        for i in range(len(inputs)):
            if not isinstance(inputs[i],Value):
                inputs[i]=Value(inputs[i])
        sum_param=0
        for i in range(len(inputs)):
            sum_param=(self.weights[i]*inputs[i])+sum_param
        sum_param+=self.bias
        result = sum_param.tanh()
        return result
    def zero_grad(self):
        for i in self.weights:
            i.grad=0
        self.bias.grad=0
    def update_params(self,lr):
        for i in self.weights:
            i.data-=lr*i.grad
        self.bias.data-=lr*self.bias.grad

# x=Neuron(3)
# res = x.forward([1,2,3])

# topo = res.build_topo()

# for i in x.weights:
#     print(i.grad,end=" ")

# print("gradient before:",res.grad)
# res.backward()
# print("data after:",res.grad)
# for i in x.weights:
#     print(i.grad,end=" ")