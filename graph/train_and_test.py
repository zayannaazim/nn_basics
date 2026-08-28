from mlp import *
import matplotlib.pyplot as plt
epochs=10000
lr=0.1

xor_mlp=MLP([4,1],2)

inputs=[[0,0],[0,1],[1,0],[1,1]]
y=[0,1,1,0]
losses=[]

def loss_fn(y,y_i):
    N=len(y)
    loss=0
    for i in range(N):
         loss=((y_i[i]+(-y[i]))*(y_i[i]+(-y[i])))+loss
    return loss * (1/N)

for epoch in range(epochs):
    xor_mlp.zero_grad()
    y_i=[]
    for i in inputs:
        y_i.extend(xor_mlp.forward(i))
    loss = loss_fn(y,y_i)
    loss.backward()
    xor_mlp.update_params(lr)
    losses.append(loss.data)
    print(epoch)
print([_.data for _ in y_i])

plt.plot(losses)
plt.show()