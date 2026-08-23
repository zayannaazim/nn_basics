W=0.41
B=0.7
epochs=1000000

w_train=0.5
b_train=0.5

lr=0.00001

def f(x,w=W,b=B):
    l=[]
    for i in range(len(x)):
        l.append(w*x[i]+b)
    return l

def loss_fn(x,y,w=W,b=B):
    N=len(x)
    loss=[]
    for i in range(N):
         loss.append((w*x[i]+b-y[i])**2)
    return 1/N * sum(loss)

def gradient(x,y,w=W,b=B):
    N=len(x)
    w_ret=0
    b_ret=0
    for i in range(N):
        w_ret += 2*(w*x[i]+b-y[i])*x[i]
        b_ret += 2*(w*x[i]+b-y[i])
    return 1/N*w_ret,1/N*b_ret

y_i=[]
x=[1,2,3,4,5,6,7,8,9,10,11,12,1,4,1,1234,34,1,134,6,234,6,8,28,27,5,234]
y=f(x)
for epoch in range(epochs):
    w_grad,b_grad=gradient(x,y,w=w_train,b=b_train)
    w_train-=lr*w_grad
    b_train-=lr*b_grad
    if epoch%10==0:
        print("W: ",w_train,", B: ",b_train)