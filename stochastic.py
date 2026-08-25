
import matplotlib.pyplot as plt
W=0.41
B=0.7
epochs=10000
lr=0.001

def f(x,w=W,b=B):
    res=[]
    for i in x:
        res.append(w*i+b)
    return res

def loss_fn(x_i,y_i,w=W,b=B):
    return ((w*x_i+b-y_i)**2)

def grad(x_i, y_i,w,b):
    return 2 * (w*x_i+b-y_i)*x_i, 2 * (w*x_i+b-y_i)

x=[1,2,3,4,5,6,7,8,9,10,11,12,1,4,1,34,1,6,6,8,28,27,5]
y=f(x)
w_train=0.5
b_train=0.5
N=len(x)
losses=[]
e=[]
k=10
n=0
w_cumulative=0
b_cumulative=0
for epoch in range(epochs):
    for i in range(N):
        w_res,b_res=grad(x[i],y[i],w_train,b_train)
        w_cumulative+=w_res
        b_cumulative+=b_res
        if n == k or (n!=0 and epoch == epochs-1 and i==N-1):
            w_train -= (1/n) * lr * w_cumulative
            b_train -= (1/n) * lr * b_cumulative
            w_cumulative=0
            b_cumulative=0
            n=0
            e.append(epoch)
            losses.append(loss_fn(x[i],y[i],w=w_train,b=b_train))
        n+=1
        
    x=x[::2]+x[1::2]
    y=y[::2]+y[1::2]
    if epoch%10==0:
        print("W: ",w_train,", B: ",b_train)


plt.plot(losses)
plt.show()