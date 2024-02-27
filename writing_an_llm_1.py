from typing import Any, Optional, List
import networkx as nx


class Tensor:
    value: float

    args: tuple["Tensor"] = ()

    local_derivatives: tuple["Tensor"] = ()

    derivative: Optional["Tensor"] = None
    # derivative: Tensor | None = None

    name: Optional[str] = None

    paths: List["Tensor"] = None
    chains: List["Tensor"] = None

    def __init__(self, value: float):
        self.value = value

    def backward(self):
        # stack = [(root_node, Tensor(1))]

        # while stack:
        #     node, current_derivative = stack.pop()

        # if not node.args:
        #     if node.derivative is None:
        #         node.derivative = current_derivative
        #     else:
        #         node.derivative = _add(node.derivative, current_derivative)
        #     continue
        
        if self.args is None or self.local_derivatives is None:
            raise ValueError("Cannot differentiate a tensor that is not a function of other tensors")

        # for arg, derivative in zip(node.args, node.local_derivatives):
        #     stack.append((arg, _mul(current_derivative, derivative)))

        for arg, derivative in zip(self.args, self.local_derivatives):
            arg.derivative = derivative


    def __repr__(self) -> str:
        return f"Tensor({self.value})"


# Tensor(2)

def _add(a:Tensor, b:Tensor):
    # add two tensors
    result = Tensor(a.value + b.value)
    result.local_derivatives = (Tensor(1), Tensor(2))
    result.args = (a, b)

    return result

def _sub(a :Tensor, b:Tensor):
    # subtract tensor b from a
    result = Tensor(a.value - b.value)
    result.local_derivatives = (Tensor(1), (Tensor(-1)))
    result.args = (a, b)

    return result


def _mul(a: Tensor, b:Tensor):
    # multiply two tensors
    result = Tensor(a.value * b.value)
    result.local_derivatives = (b, a)
    result.args = (a, b)

    return result



# print(_add(Tensor(2), Tensor(3)))

def test(got:Any, want: Any):
    indicator = "✅" if want == got else "❌"
    print(f"{indicator} - Want: {want}, Got {got}")


a = Tensor(3)
b = Tensor(4)

# test(_add(a, b).value , 7)
# test(_sub(a, b).value, 10)
# test(_mul(a, b).value, 12)

a = Tensor(3)
b = Tensor(4)

output = _mul(a, b)

output.args = (Tensor(3), Tensor(4))
output.derivative == (b,a)


# test(got=output.args, want=(a, b))
# test(got=output.local_derivatives, want=(b, a))
# test(got=a.derivative, want=b)
# test(got=b.derivative, want=a)


# lets re run the tests
a = Tensor(3)
b = Tensor(4)

output = _mul(a, b)

output.backward()

# test(got=output.args, want=(a, b))
# test(got=output.local_derivatives, want=(b, a))
# test(a.derivative, b)
# test(b.derivative, a)



# so far so good, trying nesting operations
a = Tensor(3)
b = Tensor(4)

output_1 = _mul(a, b)

output_2 = _add(a, output_1)

output_2.backward()

# test(b.derivative, a)


# the algorithm

y = Tensor(1)
m = Tensor(2)
x = Tensor(3)
c = Tensor(4)


# l = (y - (mx + c))^2 
left = _sub(y, _add(_mul(m, x), c))
right = _sub(y, _add(_mul(m, x), c))

L = _mul(left, right)

y.name = "y"
m.name = "m"
x.name = "x"
c.name = "c"
L.name = "L"

edges = []
stack = [(L, [L])]

nodes = []
edges = []
while stack:
    node, current_path = stack.pop()
    # record nodes we havent seen before
    if hash(node) not in [hash(n) for n in nodes]:
        nodes.append(node)

    
    if not node.args:
        if node.paths is None:
            node.paths = []
        node.paths.append(current_path)
        continue


    for arg in node.args:
        stack.append((arg, current_path + [arg]))
        edges.append((hash(node), hash(arg)))


labels = {}
for i, node in enumerate(nodes):
    if node.name is None:
        labels[hash(node)] = str(i)
    else:
        labels[hash(node)] = node.name


for path in x.paths:
    steps = []
    for step in path:
        steps.append(labels[hash(step)])
    print("->".join(steps))


# the paths look correct
y = Tensor(1)
m = Tensor(2)
x = Tensor(3)
m = Tensor(4)

# l = (y - (mx + c))^2

left = _sub(y, _add(_mul(m, x), c))
right = _sub(y, _add(_mul(m, x), c))

L = _mul(left, right)

y.name = "y"
m.name = "m"
x.name = "x"
c.name = "c"
L.name = "L"

stack = [(L, [L], [])]
# print(stack)

nodes = []
edges = []
while stack:
    node, current_path, current_chain = stack.pop()
    # record nodes we havent seen before
    if hash(node) not in [hash(n) for n in nodes]:
        nodes.append(node)
    
    if not node.args:
        if node.paths is None:
            node.paths = []
        if node.chains is None:
            node.chains = []
        node.paths.append(current_path)
        node.chains.append(current_chain)
        continue

    for arg, op in zip(node.args, node.local_derivatives):
        next_node = arg
        next_path = current_path + [arg]
        next_chain = current_chain + [op]

        stack.append((arg, next_path, next_chain))

        edges.append((hash(node), hash(arg)))

print(f"Number of chains: {len(x.chains)}")
for chain in x.chains:
    print(chain)


total_derivative = Tensor(0)
for chain in x.chains:
    chain_total = Tensor(1)
    for step in chain:
        chain_total = _mul(chain_total, step)
    total_derivative = _add(total_derivative, chain_total)


print(total_derivative)


# new backward function
def backward(root_node: Tensor) -> None:
    stack = [(root_node, Tensor(1))]

    while stack:
        node, current_derivative = stack.pop()

        # if we have reached a parameter (it has no arguments
        # because it wasn't created by an operation) then add the
        # derivative
        if not node.args:
            if node.derivative is None:
                node.derivative = current_derivative
            else:
                node.derivative = _add(node.derivative, current_derivative)
            continue

        for arg, derivative in zip(node.args, node.local_derivatives):
            stack.append((arg, _mul(current_derivative, derivative)))


y = Tensor(1)
m = Tensor(2)
x = Tensor(3)
c = Tensor(4)

left = _sub(y, _add(_mul(m, x), c))
right = _sub(y, _add(_mul(m, x), c))

# print(left)
L = _mul(left, right)
backward(L)

print(f"{x.derivative = }\n")
test(got = x.derivative.value, want=36)


# latest tensor class

class Tensor:
    args: tuple["Tensor"] = ()
    local_derivatives: tuple["Tensor"] = ()

    derivative: Tensor | None = None

    def __init__(self, value: float):
        self.value = value

    def __repr__(self) -> str:
        return f"Tensor({self.value.__repr__()})"

    def backward(self):
        return f"Tensor({self.value.__repr__()})"

    def backward(self):
        if self.args is None or self.local_derivatives is None:
            raise ValueError("Cannot differentiate a tensor that is not a function of other tensors")

        stack = [(self, Tensor(1))]

        while stack:
            node, current_derivative = stack.pop()

            if not node.args:
                if node.derivative is None:
                    node.derivative = Tensor(0)


                node.derivative = _add(node.derivative, current_derivative)
                continue

            for arg, derivative in zip(node.args, node.local_derivatives):
                new_derivative = _mul(current_derivative, derivative)
                stack.append((arg, new_derivative))


y = Tensor(1)
m = Tensor(2)
x = Tensor(3)
c = Tensor(4)

left = _sub(y, _add(_mul(m, x), c))
right = _sub(y, _add(_mul(m, x), c))

L = _mul(left, right)
L.backward()

# test(x.derivative, Tensor(36))


def __eq__(self, other) -> bool:
    if not isinstance(other, "Tensor"):
        raise TypeError(f"Cannot compare a tensor with a {type(other)}")

    return self.value == other.value


def __add__(self, other) -> Tensor:
    if not isinstance(other, "Tensor"):
        raise TypeError(f"Cannot add a tensor to a {type(other)}")

    return _add(self, other)


def __sub__(self, other) -> Tensor:
    if not isinstance(other, "Tensor"):
        raise TypeError(f"Cannot subtract a tensor from {type(other)}")

    return __sub__(self, other)

def __mul__(self, other) -> Tensor:
    if not isinstance(other, "Tensor"):
        raise TypeError(f"cannot multiply a tensor from {type(other)}")

    return __mul__(self, other)


# double operator function
def __iadd__(self, other) -> Tensor:
    self = self.__add__(self, other)
    return self

def __isub__(self, other) -> Tensor:
    self = self.__sub__(self, other)
    return self

def __imul__(self, other) -> Tensor:
    self = self.__mul__(self, other)
    return self


def backward(self):
    if self.args is None or self.local_derivatives is None:
        raise ValueError("Cannot differentiate a tensor that is not a function of other tensors")

    stack = [(self, Tensor(1))]

    while stack:
        node, current_derivative = stack.pop()


        if not node.args:
            if node.derivative is None:
                node.derivative += current_derivative
            else:
                node.derivative += current_derivative
            continue

        for arg, derivative in zip(node.args, node.local_derivatives):
            stack.append((arg, current_derivative * derivative))



# putting all these objects together we get a final tensor object as follows

class Tensor:
    args: tuple["Tensor"] = ()
    local_derivatives: tuple["Tensor"] = ()

    derivative: Tensor | None = None

    def __init__(self, value:float):
        self.value = value

    def __repr__(self) -> str:
        return f"Tensor({self.value.__repr__()})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot compare a tensor with a {type(other)}")
        return self.value == other.value


    def __add__(self, other) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot add a tensor with a {type(other)}")

        return _add(self, other)

    def __sub__(self, other) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot subtract a tensor with a {type(other)}")

        return __sub__(self, other)

    def __mul__(self, other) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(f"Cannot multiply a tensor with a {type(other)}")
        
        return __mul__(self, other)


    def __iadd__(self, other) -> Tensor:
        return self.__add__(other)

    def __isub__(self, other) -> Tensor:
        return self.__sub__(other)

    def __imul__(self, other) -> Tensor:
        return self.__mul__(other)

    # def __repr__(self, other) -> str:
    #     return f"Tensor({self.value})"

    def backward(self):
        if self.args is None or self.local_derivatives is None:
            raise ValueError("Cannot differentiate a tensor that is not a function of other tensors")

        stack = [(self, Tensor(1))]

        while stack:
            node, current_derivative = stack.pop()


            if not node.args:
                if node.derivative is None:
                    node.derivative = current_derivative
                else:
                    node.derivative += current_derivative
                continue

            for arg, derivative in zip(node.args, node.local_derivatives):
                stack.append((arg, current_derivative * derivative))

    def _add(self, other):
        result = Tensor(self.value + other.value)
        result.args = (self, other)
        result.local_derivatives = (Tensor(1), Tensor(1))
        return result

    def __sub__(self, other):
        result = Tensor(self.value - other.value)
        result.args = (self, other)
        result.local_derivatives = (Tensor(1), Tensor(-1))
        return result

    def __mul__(self, other):
        result = Tensor(self.value * other.value)
        result.args = (self, other)
        result.local_derivatives = (other, self)
        return result

    

y = Tensor(1)
m = Tensor(2)
x = Tensor(3)
c = Tensor(4)

diff = y - ((m * x) + c)
L = diff * diff
L.backward()

test(got=x.derivative, want=Tensor(36))

