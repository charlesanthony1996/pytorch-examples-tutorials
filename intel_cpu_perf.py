import torch
import torchvision.models as models
import time

model = models.resnet50(pretrained=True)
model.eval()
data = torch.rand(1, 3, 224, 224)

#warm up
for _ in range(100):
    model(data)


start = time.time()

for _ in range(100):
    model(data)

end = time.time()
print("Inference took {:.2f} ms in average".format((end-start)/ 100 * 1000))