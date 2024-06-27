#include <torch/torch.h>
#include <iostream>

// define a simple cnn model
struct SimpleCNN : torch::nn::Module {
    SimpleCNN() {
        // convolutional layers
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 3).padding(1)));




        // fully connected layer
        fc1 = register_module("fc1", torch::nn::Linear(100, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        // convolution and activation
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));

        // apply the second convolution
        x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));

        x = x.view({-1, 180});

        x = fc1-> forward(x);
        return x;
    }


    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr};
};

int main() {
    torch::manual_seed(42);

    // instantiate the model
    auto model = std::make_shared<SimpleCNN>();
    torch::optim::SGD optimizer(model-> parameters(), 0.01);
    torch::nn::MSELoss criterion;

    // input numbers as 1x1x5x5 tensors
    auto input1 = torch::randn({1, 1, 5, 5});
    auto input2 = torch::randn({1, 1, 5, 5});

    auto target = input1.sum() + input2.sum();

    model-> train();
    optimizer.zero_grad();

    // forward pass through the model
    auto output = model-> forward(torch::cat({input1, input2}, 1));
    auto loss = criterion(output, target);

    loss.backward();
    optimizer.step();

    std::cout << "Finished training step on cpu: " << std::endl;
    std::cout << "Predicted sum: " << output.item<float>() <<std::endl;
    std::cout << "Actual sum: " << target.item<float>() << std::endl;



    return 0;
}