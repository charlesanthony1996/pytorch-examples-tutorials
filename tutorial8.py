from __future__ import print_function
#%matplotlib inline

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

#set seed for reproducibility
manualSeed = 999

#manualSeed = random.randint(1, 1000)
#use if you want new results

print("Random seed: ", manualSeed)

random.seed(manualSeed)
torch.manual_seed(manualSeed)


#root directory for dataset
dataroot = "data/celeba"

#number of workers for dataloader
workers = 0

#batch size during training
batch_size = 128

#spatial size of training images.All images will be resized to this
#size using a transformer
image_size = 64

#number of channels in the training images.For color images this is 3
nc = 3

#size of z latent vector (i.e  size of generator input)
nz = 100

#size of feature maps in generator
ngf = 64

#size of feature maps in generator
ndf = 64

#number of training epochs
num_epochs = 5


#learning rate of optimizers
lr = 0.0002


#beta1 hyperparam for adam optimizers
beta1 = 0.5

#number of gpus available. use 0 for cpu mode
ngpu = 1



#download the data here.create a directory called celeba and extract it there





#we can use the image folder dataset the way we have it setup
#create the dataset

dataset = dset.ImageFolder(root= dataroot, 
    transform = transforms.Compose([
    transforms.Resize(image_size), 
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]))


#create the dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size , shuffle=True, num_workers = 0)


#decide which device we want to turn on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0 ) else 'cpu')

#plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding= 2, normalize=True).cpu(), (1, 2, 0)))



#weight initialization

#custom weights initialization called on netG and netD

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0 , 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


    


#generator code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias =False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #state size . (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #state size (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias= False),
            nn.BatchNorm2d(ngf* 2),
            nn.ReLU(True),
            #state size (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #state size (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)



#implementing the generator and apply weights_init function and check out how the printed model to
#see how the generator object is structured


netG = Generator(ngpu).to(device)


#handle mutlple gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG,list(range(ngpu)))


#apply the weights_init function to randomly initialize all weights
#to mean = 0, stdev = 0.02
netG.apply(weights_init)

print(netG)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)




#discriminator model eval here man!
netD = Discriminator(ngpu).to(device)

#handle mutli gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

#apply the weights init function to randomly initialize all weights
#to mean = 0 stdev = 0.2


netD.apply(weights_init)

print(netD)


#loss function and optimizers

#initialize BCEloss function
criterion = nn.BCELoss()


#create batch of latent vectors that we will use to visualize
#the progression of the generator

fixed_noise = torch.randn(64, nz, 1, 1, device =device)

#establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

#setup adam optimizers for both G and D
optimizerD = optim.Adam(netG.parameters(), lr=lr, betas =(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas = (beta1, 0.999))


#training

#training loop

#lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training loop: ")
#for each epoch
for epoch in range(num_epochs):
    #for each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        #update d network: maximize log(D(x)) + log(1 - D(G(z)))
        #train with all-real batch
        netD.zero_grad()
        #format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label , dtype=torch.float, device =device)
        #forward pass real batch through D
        output = netD(real_cpu).view(-1)
        #calculate loss on all-real batch
        errD_real = criterion(output, label)
        #calculate gradients for d in backward pass
        errD_real.backward()
        D_x = output.mean().item()


        #train with all fake batch
        #Generate bacth of latent vectors
        noise = torch.randn(b_size, nz, 1 , 1, device = device)
        #generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        #classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        #calculate D's loss on the all fake batch
        errD_fake = criterion(output, label)
        #calculate the gradients for this batch , accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        #compute error of D as sum over the fake nd the real batches
        errD = errD_real + errD_fake
        #update D
        optimizerD.step()

        #update G network: maximize log(D(G(z)))
        netG.zero_grad()
        label.fill_(real_label)
        #since we mnust updated D , perform another forward pass pf all fake batch through D
        output = netD(fake.detach()).view(-1)
        #calculate G's loss based on this output
        errG = criterion(output, label)
        #calculat gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        #update G
        optimizerG.step()


        #output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % 
            (epoch, num_epochs, i, len(dataloader),
            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            #save losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())


            #check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1



plt.figure(figsize=(10, 5))
plt.title("Generator and discriminator loss during training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

