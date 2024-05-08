#include <torch/torch.h>
#include <iostream>

struct SimpleModel : torch::nn::Module {
    SimpleModel() {
        fc1 = register_module("fc1", torch::nn::Linear(10, 50));
        fc2 = register_module("fc2", torch::nn::Linear(50, 2));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1 -> forward(x));
        x = fc2-> forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};





int main() {
    torch::manual_seed(42);

    auto model = std::make_shared<SimpleModel>();
    torch::optim::SGD optimizer(model -> parameters(), 0.01);
    torch::nn::CrossEntropyLoss criterion;

    auto input = torch::randn({16, 10});
    auto target = torch::randint(0, 2, {16});

    model -> train();
    optimizer.zero_grad();

    auto output = model -> forward(input);
    auto loss = criterion(output, target);

    loss.backward();
    optimizer.step();

    std::cout << "finished training step on cpu" << std::endl;
    return 0;
}
