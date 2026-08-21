import numpy as np
W=0.5
B=1

def f(x,w=W,b=B):
    return w*x+b

def loss_fn(x,y):
    N=len(y)
    return (1/N)*sum()