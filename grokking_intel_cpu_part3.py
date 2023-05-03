import torch
import numpy as np
import pandas as pd
import torchvision.models as models
import time
import torch.nn as nn
# import intel_extension_for_pytorch as ipex
import time

# load resnet-50 model and profile its inference time
def profile_resnet50_inference_time():
    model = models.resnet50(pretrained=False)
    model.eval()

    batch_size = 32

    data = torch.rand(batch_size, 3, 224, 224)

    print(data)

    with torch.autograd.profiler.emit_itt():
        start = time.time()
        for i in range(100):
            torch.profiler.itt.range_push("step{}".format(i))
            model(data)
            torch.profiler.itt.range_pop()
        end = time.time()

    # time.sleep(1000)
    print("Inference took {:.2f} ms on average".format((end - start) / 100 * 1000))



# create custom convolutional neural network model
def create_custom_model():
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv = torch.nn.Conv2d(16, 33, 3, stride = 2)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x


    model = Model()
    model.eval()

    data = torch.rand(20, 16, 50, 100)

    print(data)

    with torch.no_grad():
        model = torch.jit.trace(model, data)
        model = torch.jit.freeze(model)


    return model




if __name__ == "__main__":
    # profile resnet-50 models' inference time
    profile_resnet50_inference_time()

    # create custom model and print its structure
    custom_model = create_custom_model()

    print(custom_model)