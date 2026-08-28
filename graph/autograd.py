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

    def __sub__(self,other):
        if not isinstance(other,Value):
            other = Value(other)
        out = Value(self.data-other.data,prev=(self,other))
        def backwards():
            k=1
            for i in out._prev:
                i.grad+=k*out.grad
                k*=-1
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


# x=Value(2)
# y=Value(13)
# z=x-y
# z.backward()
# print(z.data)
# print(x.grad)
# print(y.grad)