from numpy import tanh
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
        res= tanh(self.data)
        out = Value(res,prev=(self,))
        def backward():
            diff = 1-out.data**2
            self.grad+=diff*out.grad
        out._backward=backward
        return out


x=Value(2)
y=Value(3)
z=x*y
w=x.tanh()
w.grad=3
z.grad=3
z._backward()
w._backward()
print(x.grad,y.grad)