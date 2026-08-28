from layers import *
class MLP:
    def __init__(self,neurons:list,inputs):
        self.layers=[]
        for i in neurons:
            self.layers.append(Layer(i,inputs))
            inputs=i

    def forward(self,inputs):
        result=[]
        for i in self.layers:
            result=i.forward(inputs.copy())
            inputs=result.copy()
        return result
    def zero_grad(self):
        for i in self.layers:
            i.zero_grad()

    def update_params(self,lr):
        for i in self.layers:
            i.update_params(lr)

# mlp = MLP([4,4,1],3)
# res=mlp.forward([1,2,3])
# res[0].backward()
# print('''

# Gradients:

# ''')
# for i in res[0].build_topo():
#     print(i.grad, end=" ")
# print('''

# Weights

# ''')
# for i in range(len(mlp.layers)):
#     print(f'''

# Layer {i+1}

# ''')
#     for j in range(len(mlp.layers[i].neurons)):
#         print(f"Neuron {j+1}")
#         for k in mlp.layers[i].neurons[j].weights:
#             print(k.grad)
# print('''

# Biases

# ''')
# for i in range(len(mlp.layers)):
#     print(f'''

# Layer {i+1}

# ''')
#     for j in range(len(mlp.layers[i].neurons)):
#         print(f"Neuron {j+1}")
#         print(mlp.layers[i].neurons[j].bias.grad)