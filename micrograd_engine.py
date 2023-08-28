class Value:
    # stores a single scalar value and its gradient

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self.backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

    
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward


    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for one"
        out = Value(self.data ** other , (self,), f"**{other}")


        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward


        return out


    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLu")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out


    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(V):
            if v is not visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)


        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    
    def __rmul__(self, other):
        return self + (-other)

    
    def __truediv__(self, other):
        return self * other ** -1


    def __rtruediv__(self, other):
        return other * self ** -1

    
    def __repr__(self):
        return f"Value(data = {self.data}, grad = {self.grad}"


    


