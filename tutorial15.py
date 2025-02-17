import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import time

plt.ion()

import torch
from torchvision import transforms, datasets


data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = "data/hymenoptera_data"

image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
for x in ["train", "val"]}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size= 16, shuffle=True, num_workers = 0) 
for x in ["train", "val"]}


dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

class_names = image_datasets["train"].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import torchvision


def imshow(inp, title =None, ax= None, figsize=(5, 5)):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    if ax is None:
        fig, ax = plt.subplots(1, figsize= figsize)
    ax.imshow(inp)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)



    #get a batch of training data
    inputs, classes = next(iter(dataloaders["train"]))


    # make a grid from batch
    out = torchvision.utils.make_grid(inputs, nrow= 4)

    fig , ax = plt.subplots(1, figsize =(10, 10))
    imshow(out, title=[class_names[x] for x in classes], ax= ax)




# support function for model training

def train_model(model,criterion, optimizer, scheduler, num_epochs = 25, device ="cpu"):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # each epoch has a training and a validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0.0

            # iterator over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss= criterion(outputs, labels)


                    if phase == "train":
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]


            print("{} loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))



            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print("training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60 , time_elapsed % 60))
    print("Best val Acc : {:4f}".format(best_acc))


    model.load_state_dict(best_model_wts)
    return model






            



def visualize_model(model, rows=3, cols=3):
  was_training = model.training
  model.eval()
  current_row = current_col = 0
  fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

  with torch.no_grad():
    for idx, (imgs, lbls) in enumerate(dataloaders['val']):
      imgs = imgs.cpu()
      lbls = lbls.cpu()

      outputs = model(imgs)
      _, preds = torch.max(outputs, 1)

      for jdx in range(imgs.size()[0]):
        imshow(imgs.data[jdx], ax=ax[current_row, current_col])
        ax[current_row, current_col].axis('off')
        ax[current_row, current_col].set_title('predicted: {}'.format(class_names[preds[jdx]]))

        current_col += 1
        if current_col >= cols:
          current_row += 1
          current_col = 0
        if current_row >= rows:
          model.train(mode=was_training)
          return
    model.train(mode=was_training)


# training a custom classifier based on a quantized feature extractor
import torchvision.models.quantization as models

model_fe = models.resnet18(pretrained =True, progress=True, quantize =True)
num_ftrs = model_fe.fc.in_features



from torch import nn

def create_combined_model(model_fe):
    model_fe_features = nn.Sequential(
        model_fe.quant,
        model_fe.conv1,
        model_fe.bn1,
        model_fe.relu,
        model_fe.maxpool,
        model_fe.layer1,
        model_fe.layer2,
        model_fe.layer3,
        model_fe.layer4,
        model_fe.avgpool,
        model_fe.dequant
    )

    # step 2
    new_head = nn.Sequential(
        nn.Dropout(p =0.5),
        nn.Linear(num_ftrs, 2),
    )

    # step 3 Combine and dont forget the quant stubs
    new_model = nn.Sequential(model_fe_features, nn.Flatten(1), new_head,)
    return new_model

import torch.optim as optim
new_model = create_combined_model(model_fe)
new_model = new_model.to("cpu")


criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(new_model.parameters(), lr=0.01, momentum = 0.9)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size= 7, gamma=0.1)



# train and evaluate

new_model = train_model(new_model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=25, device='cpu')

visualize_model(new_model)
plt.tight_layout()

# fine tuning the quantizable model
