import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# exercise 1

imagefoldertrain = "/users/charles/downloads/segmentation_data/train/images/"
maskfoldertrain = "/users/charles/downloads/segmentation_data/train/masks/"
imagefoldertest = "/users/charles/downloads/segmentation_data/test/images/"
maskfoldertest = "/users/charles/downloads/segmentation_data/test/masks/"

samplefilename = "SIMCEPImages_A21_C87_F1_s05_w2.tif"

sampleimage = cv2.imread(os.path.join(imagefoldertrain, samplefilename),cv2.IMREAD_UNCHANGED)
print(sampleimage.shape)
plt.imshow(sampleimage)
plt.figure()
# plt.show()


samplemask = cv2.imread(os.path.join(maskfoldertrain, samplefilename), cv2.IMREAD_UNCHANGED)
print("Mask shape:", samplemask.shape)
plt.imshow(samplemask, cmap='gray')
plt.title("Corresponding Mask")
# plt.show()

# let us create a function to load the images into a list. 
def loadimages(folder):
    outputlist = []
    for file in os.listdir(folder):
        img = cv2.imread(folder + file,cv2.IMREAD_UNCHANGED)
        outputlist.append(img)
    return outputlist

rawImagesTrain = loadimages(imagefoldertrain)

# Print the number of images loaded
print("Number of images loaded:", len(rawImagesTrain))


maskfoldertrain = "/users/charles/downloads/segmentation_data/train/masks/"

# Use the function to load in the raw masks from the training mask folder
rawMasksTrain = loadimages(maskfoldertrain)

# Print the number of masks loaded
print("Number of masks loaded:", len(rawMasksTrain))

# You can get the shape of one image as follows
img = rawImagesTrain[1]
print(img.shape)

# collect all heights and widths of all images
allWidths = []
allHeights = []
for img in rawImagesTrain:
    if img is not None:
        # Append the height and width
        allHeights.append(img.shape[0])  # height is the first element of the shape tuple
        allWidths.append(img.shape[1])  # width is the second element of the shape tuple
    else:
        print("An image was None, skipping...")

# Print some sample heights and widths
print("Sample heights:", allHeights[:5])
print("Sample widths:", allWidths[:5])

print("Minimal width: " + str(min(allWidths)))
print("Average width: " + str(sum(allWidths)/len(allWidths)))
print("Maximum width: " + str(max(allWidths)))
print("Minimal height: " + str(min(allHeights)))
print("Average height: " + str(sum(allHeights)/len(allHeights)))
print("Maximum height: " + str(max(allHeights)))

# exercise 2

img = rawImagesTrain[100]
img.shape

# Here we define the size of the slices we want to create
slicesize = 232

# This is the original image, and the top left slice
plt.imshow(img)
plt.figure()
plt.imshow(img[0:slicesize, 0:slicesize])
plt.figure()


heightslicenr = int(np.floor(img.shape[0]/slicesize))
print('We can inlude ' + str(heightslicenr) + ' slices on top of eachother.')
widthslicenr = int(np.floor(img.shape[1]/slicesize))
print('We can inlude ' + str(widthslicenr) + ' slices next to eachother.')
print('So in total, we can create ' + str(heightslicenr*widthslicenr) + ' slices from this image.')

x = 0 # first row is index 0
y = 2 # third column is index 2
plt.imshow(img[x*slicesize:(x+1)*slicesize, y*slicesize:(y+1)*slicesize])
plt.figure()

mask = rawMasksTrain[100]

x = 0 # first row index 0
y = 2 # third column is index 2
plt.imshow(mask[x*slicesize:(x+1)*slicesize, y*slicesize:(y+1)*slicesize])
plt.figure()

# Assumption: rawImagesTrain and rawMasksTrain contain the images and masks, and they have the same lenght. 
# This is necessary for the algorithm to work
print(len(rawImagesTrain))
print(len(rawMasksTrain))

def returnslices(imagelist, slicesize):
    outputlist = []
    for img in imagelist:
        if img is not None:
            heightslicenr = img.shape[0] // slicesize  # Number of slices in the vertical direction
            widthslicenr = img.shape[1] // slicesize  # Number of slices in the horizontal direction
            for y in range(heightslicenr):
                for x in range(widthslicenr):
                    imgslice = img[y * slicesize: (y + 1) * slicesize, x * slicesize: (x + 1) * slicesize]
                    outputlist.append(imgslice)
        else:
            print("An image was None, skipping...")
    return outputlist

slicesize = 232

imgslicestrain = returnslices(rawImagesTrain,slicesize) 
maskslicestrain = returnslices(rawMasksTrain,slicesize) 

# Visually test a few slices and masks
for i in range(9):
    plt.imshow(imgslicestrain[i])
    plt.figure()
    plt.imshow(maskslicestrain[i])
    plt.figure()
    # plt.show()


# exercise 3

# Importing the necessary Keras and Tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.config.run_functions_eagerly(True)

# Initialization, setting of hyperparameters
# We assume the training data is in the lists imgslices and maskslices
input_shape = (232,232,1)
opt = 'adam'
loss = 'binary_crossentropy'

# TODO: Fill in the blanks. General advice: Reuse code from the layers provided. 

# In this syntax, we chain functions. x is the neural net we are building; it represents a function. 

# Input layer
inputs = keras.Input(shape=input_shape)

# First downwards stride (232,232,32)
x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),padding='same',activation="relu")(inputs)
x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),padding='same',activation="relu")(x)
copy1 = x # we store this convolutional state in a temporary var, since we want to concatenate it later in the upwards stride
x =  tf.keras.layers.MaxPooling2D((2, 2))(x)

# Second downwards stride (116,116,64)
# Add two convolutional layers, with kernel_size 3, padding = 'same' and activation relu. We want 64 filters per layer
x = ...(x)
x = ...(x)
copy2 = x
x =  ...(x) # add a maxpooling layer, like above

# Third downwards stride (58,58,128)
# Add two convolutional layers, with kernel_size 3, padding = 'same' and activation relu. We want 128 filters per layer
x = ...(x)
x = ...(x)
copy3 = x
x =  tf.keras.layers.MaxPooling2D((2, 2))(x)

# Bottom layer stride, just one convolutions (29,29,256)
x = tf.keras.layers.Conv2D(filters=256,kernel_size=(3, 3),padding='same',activation="relu")(x)

# First upward stride, uses copy 3. (58,58,128), after first Conv2DTranspose
x = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same',activation="relu")(x)
x = tf.keras.layers.concatenate([x,copy3]) # here we concatenate the feature maps from the downward stride. 
x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3, 3),padding='same',activation="relu")(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size=(3, 3),padding='same',activation="relu")(x)

# Second upward stride, uses copy 2. (116,116,64)
x = ...(x) # add a Conv2DTranspose layer, like before. We will use 64 filters. 
x = ... # here we concatenate the correct feature map from the downward stride. 
x = ...(x) # add a regular convolution layer, like before, 64 filters. 
x = ...(x) # add a regular convolution layer, like before, 64 filters. 

# Final upward stride, uses copy 1. (232,232,32)
x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same',activation="relu")(x)
x = ... # here we concatenate the correct feature maps from the downward stride. 
x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),padding='same',activation="relu")(x)
x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),padding='same',activation="relu")(x)

# Output layer has 1 dimension, like the mask. (232,232,1)
# Add a final convolutional layer, with kernel size 1 and using a sigmoid activation function
# You should be able to deduct the nr of filters and whether you need padding yourself. 
outputs = ...(x) 

# Now, we can create the model as follows: 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])