from neurons import *

class Layer:
    def __init__(self,neurons,inputs):
        self.neurons=[]
        for i in range(neurons):
            self.neurons.append(Neuron(inputs))
    def forward(self,inputs):
        if len(self.neurons[0].weights)!=len(inputs):
                    print("No of inputs not correct.")
                    raise ValueError
        result=[]
        for i in self.neurons:
            result.append(i.forward(inputs.copy()))
        return result
    def zero_grad(self):
        for i in self.neurons:
            i.zero_grad()
    def update_params(self,lr):
        for i in self.neurons:
            i.update_params(lr)
             
# layer = Layer(3, 4)
# outputs = layer.forward([1, 2, 3, 4])
# print([_.data for _ in outputs])

# cumulative=0
# for i in outputs:
#     cumulative=i+cumulative
# cumulative.backward()
# neurons=[_.weights for _ in layer.neurons]
# for i in neurons:
#     print(i)
# print([_.bias.data for _ in layer.neurons])