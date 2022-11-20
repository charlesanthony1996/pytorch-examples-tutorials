from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
pretrained_model= "/users/charles/dekstop/pytorch-tutorials/data/lenet_mnist_model.pth"
use_cuda = True


# LeNet Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# Load the pretrained model
# model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()




# fsgm attack

def fsgm_attack(image, epsilon, data_grad):
    #collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    #create the perturbed image by adjusting each pixel level of the input image
    perturbed_image = image + epsilon * sign_data_grad
    #adding clipping to maintain [0, 1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    #return the perturbed image
    return perturbed_image



#testing function here -> central part


def test(model, device, test_loader, epsilon):
    #accuracy counter
    correct = 0
    adv_examples = []

    #loop all over the examples in the test set
    for data, target in test_loader:
        #send the data and label to the device
        data, target = data.to(device) , target.to(device)

        #set requires_grad attribute of tensor. Important for attack
        data.requires_grad = True

        #forward pass the data through the model
        output = model(data)
        #get the index of the max log probability
        init_pred = output.max(1, keepdim =True)[1]

        #if the initial prediction is wrong , dont bother attacking, just move on 
        if init_pred.item() != target.item():
            continue

        #calculate the loss
        loss = F.nll_loss(output, target)

        #zero all existing gradients
        model.zero_grad()

        #calculate gradients of model in backward pass
        loss.backward()

        #collect datagrad
        data_grad = data.grad.data

        #call fsgm attack
        perturbed_data = fsgm_attack(data, epsilon, data_grad)

        #reclassify the perturbed image
        output = model(perturbed_data)

        #check for success
        #get the index of the max log probability
        final_pred = output.max(1, keepdim=True)[1]
        
        if final_pred.item() == target.item():
            correct += 1
            #Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            #save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    #correct the final accuracy for this epsilon
    final_acc = correct/ float(len(test_loader))
    print("Epsilon: {}\t Test Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    #return the final accuracy and an advesarial example
    return final_acc , adv_examples





#run attack

accuracies = []
examples = []

#run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)


#accuracy vs epsilon


plt.figure(figsize=(5, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.35, step=0.05))
plt.title("Accuracy vs epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")

plt.show()



#sample adversarial examples

#plot several examples of adversarial examples at each epsilon
cnt = 0
plt.figure(figsize=(8, 10))
for i in range(len(epsilons)):
    for j in range(examples[i]):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig , adv, ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")

plt.tight_layout()
plt.show()



