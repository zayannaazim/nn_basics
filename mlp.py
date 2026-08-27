from numpy import tanh
from numpy.random import uniform

class Value:
    def __init__(self,d,prev=()):
        self.data=d
        self.grad=0
        self._backward=lambda:None
        self._prev=prev

    def __add__(self,other):
        if not isinstance(other,Value):
            other = Value(other)
        out = Value(self.data+other.data,prev=(self,other))
        def backwards():
            for i in out._prev:
                i.grad+=out.grad
        out._backward=backwards
        return out

    def __mul__(self,other):
        if not isinstance(other,Value):
           other = Value(other)
        out=Value(self.data*other.data,prev=(self,other))
        def backwards():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward=backwards        
        return out

    def tanh(self):
        res= float(tanh(self.data))
        out = Value(res,prev=(self,))
        def backward():
            diff = 1-out.data**2
            self.grad+=diff*out.grad
        out._backward=backward
        return out
    
    def build_topo(self):
        visited_nodes=set()
        node_order=[]
        def recurse_nodes(node):
            if node not in visited_nodes:
                visited_nodes.add(node)
                for parent in node._prev:
                    recurse_nodes(parent)
                node_order.append(node)
        recurse_nodes(self)
        return node_order
    
    def backward(self):
        self.grad=1
        nodes=self.build_topo()[::-1]
        for i in nodes:
            i._backward()
    def zero_grad(self):
        nodes =self.build_topo()
        for i in nodes:
            i.grad=0


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


layer = Layer(3, 4)
outputs = layer.forward([1, 2, 3, 4])
print([_.data for _ in outputs])

cumulative=0
for i in outputs:
    cumulative=i+cumulative
cumulative.backward()
neurons=[_.weights for _ in layer.neurons]
for i in neurons:
    print(i)
print([_.bias.data for _ in layer.neurons])