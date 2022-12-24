import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

def symmetric(x):
    return x.triu() + x.triu(1).transpose(-1, -2)


x = torch.rand(3, 3)
a = symmetric(x)
assert torch.allclose(a, a.T)

print(a)


class LinearSymmetric(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(n_features, n_features))


    def forward(self, x):
        a = symmetric(self.weight)
        print(a)
        return x @ a



layer = LinearSymmetric(3)
print(layer)
out = layer(torch.rand(8, 3))


print()

class Symmetric(nn.Module):
    def forward(self, x):
        return x.triu() + x.triu(1).transpose(-1, -2)


layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Symmetric())


a = layer.weight
assert torch.allclose(a, a.T)
print(a)

print()

class Skew(nn.Module):
    def forward(self, x):
        a = x.triu(1)
        return a - a.transpose(-1, -2)


cnn = nn.Conv2d(in_channels = 5, out_channels = 8 , kernel_size = 3)
parametrize.register_parametrization(cnn, "weight", Skew())

print()
print(cnn.weight[0, 1])
print(cnn.weight[2, 2])

print()

layer = nn.Linear(3, 3)
print(f"Unparametrized: \n{layer}")
parametrize.register_parametrization(layer, "weight", Symmetric())
print(f"\nParametrized:\b{layer}")


print()


print(layer.parametrizations)
print()
print(layer.parametrizations.weight)
print()

#indexing weights here

print(layer.parametrizations.weight[0])
print()

print(dict(layer.named_parameters()))
print()

print(layer.parametrizations.weight.original)
print()


 
symmetric = Symmetric()
print(symmetric)
print()
weight_orig = layer.parametrizations.weight.original
print()
print(weight_orig)

print(torch.dist(layer.weight, symmetric(weight_orig)))
print()

class NoisyParametrization(nn.Module):
    def forward(self, x):
        print("Computing the parametrizations")
        return x


layer = nn.Linear(4, 4)
parametrize.register_parametrization(layer, "weight", NoisyParametrization())
print()
print("Here layer.weight is recomputed every time we call it")
foo = layer.weight + layer.weight.T
bar = layer.weight.sum()

with parametrize.cached():
    print()
    print("Here, it is computed just the first time layer.weight is called")
    foo = layer.weight + layer.weight.T
    bar = layer.weight.sum()


#concatenating parametrizations

class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        print()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, x):
        # (I+X) (I - X)^{-1}
        print()
        return torch.linalg.solve(self.Id + x, self.Id - x)


layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
parametrize.register_parametrization(layer, "weight", CayleyMap(3))
x = layer.weight
print(torch.dist(x.T @ x, torch.eye(3))) #x is orthogonal



class MatrixExponential(nn.Module):
    def forward(self, x):
        return torch.matrix_exp(x)


layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", MatrixExponential())

x = layer_orthogonal.weight
print(x)
print(torch.dist(x.T @ x , torch.eye(3)))

layer_spd = nn.Linear(3, 3)
parametrize.register_parametrization(layer_spd, "weight", Symmetric())
parametrize.register_parametrization(layer_spd, "weight", MatrixExponential())

x = layer_spd.weight
print(torch.dist(x, x.T))  # symmetricity is seen here with tho output
print()
print((torch.symeig(x).eigenvalues > 0.).all()) # 3 is positive definite
print()

class Skew(nn.Module):
    def forward(self, x):
        x = x.triu(1)
        return a - a.transpose(-1, -2)


    def right_inverse(self, a):
        return a.triu(1)


layer = nn.Linear(3, 3)
parametrize.register_parametrization(layer, "weight", Skew())
x = torch.rand(3,3 )
x = x - x.T
layer.weight = x
print(layer.weight)
print()
print(torch.dist(layer.weight, x))
print()



class CayleyMap(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("Id", torch.eye(n))

    def forward(self, x):
        #  assume x is self symmetric
        return torch.linalg.solve(self.Id + x , self.Id - x)

    def right_inverse(self, a):
        return torch.linalg.solve(x - self.Id, self.Id + x)

    

layer_orthogonal = nn.Linear(3, 3)
parametrize.register_parametrization(layer_orthogonal, "weight", Skew())
parametrize.register_parametrization(layer_orthogonal, "weight", CayleyMap(3))

#sample an orthogonal matrix with positive  determinant
x = torch.empty(3, 3)
nn.init.orthogonal_(x)
if x.det() < 0:
    x[0].neg_()

layer_orthogonal.weight = x
print()
print(torch.dist(layer_orthogonal.weight, x))

layer_orthogonal.weight = nn.init.orthogonal_(layer_orthogonal.weight)
print()
print(layer_orthogonal.weight)



class PruningParametrization(nn.Module):
    def __init__(self, x, p_drop= 0.2):
        super().__init__()
        # sample zero with probability p_drop
        mask = torch.full_like(x, 1.0 - p_drop)
        self.mask = torch.bernoulli(mask)

    def forward(self, x):
        return x * self.mask

    def right_inverse(self, a):
        return a


    
layer = nn.Linear(3, 4)
print(layer)
print()
x = torch.rand_like(layer.weight)
print(f"initialization matrix:\n{x}")
parametrize.register_parametrization(layer, "weight", PruningParametrization(layer.weight))
layer.weight = x
print(layer.weight)
print()
print(f"\n Initialized weight:\n{layer.weight}")
print()


#removing parametrizations
layer = nn.Linear(3, 3)
print("Before: ")
print(layer)
print(parametrize.register_parametrization(layer, "weight", Skew()))
print("Parametrized:")
print(layer)
print(layer.weight)
parametrize.remove_parametrizations(layer, "weight")
print("\nAfter Weight has skew -symmetric values but it is unconstrained")
print(layer)
print(layer.weight)

print()
linear = nn.Linear(3, 3)
print("Before: ")
print(layer)
print(layer.weight)
parametrize.register_parametrization(layer, "weight", Skew())
print()
print(layer)
print(layer.weight)
parametrize.remove_parametrizations(layer, "weight", leave_parametrized= False)
print(layer)
print(layer.weight)
