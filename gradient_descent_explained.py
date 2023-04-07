# define the function
def f(x):
    return x ** 2


# get the derivative here
def df(x):
    return 2 * x

# model parameter initalized here

x = 2.0

# set the learning rate and the num of iterations -> aka epochs

learning_rate = 0.1
num_iterations = 50


# get the for loop to the run the algorithm here
# differ learning rate to see the changes
for i in range(num_iterations):
    dx = df(x)

    # update x in the direction of steepest descent

    x = x - learning_rate * dx

    # print the current value of x and the corresponding value of y

    print("Iteration: ", i+ 1, "x: ", x, "y: ", f(x))