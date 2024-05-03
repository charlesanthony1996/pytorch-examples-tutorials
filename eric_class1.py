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
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
copy2 = x
x =  tf.keras.layers.MaxPooling2D((2, 2))(x) # add a maxpooling layer, like above

# Third downwards stride (58,58,128)
# Add two convolutional layers, with kernel_size 3, padding = 'same' and activation relu. We want 128 filters per layer
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation="relu")(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation="relu")(x)
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
x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same', activation="relu")(x) # add a Conv2DTranspose layer, like before. We will use 64 filters. 
x = tf.keras.layers.concatenate([x, copy2]) # here we concatenate the correct feature map from the downward stride. 
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation="relu")(x) # add a regular convolution layer, like before, 64 filters. 
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(x) # add a regular convolution layer, like before, 64 filters. 

# Final upward stride, uses copy 1. (232,232,32)
x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same',activation="relu")(x)
x = tf.keras.layers.concatenate([x, copy1]) # here we concatenate the correct feature maps from the downward stride. 
x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),padding='same',activation="relu")(x)
x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3, 3),padding='same',activation="relu")(x)

# Output layer has 1 dimension, like the mask. (232,232,1)
# Add a final convolutional layer, with kernel size 1 and using a sigmoid activation function
# You should be able to deduct the nr of filters and whether you need padding yourself. 
# Output layer for binary segmentation should have only 1 filter
outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding="same", activation="sigmoid")(x)

# Now, we can create the model as follows: 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])


# summary of the model
model.summary()

# compile the model
model.compile(optimizer=opt, loss=loss)

# training the model
# TODO: Pick a batchsize and a number of epochs for you to train on. 
batchsize = 32
epochs = 10 # on my machine, 1 epoch took 7 minutes

X = np.array(imgslicestrain,dtype=np.uint8)
Y = np.array(maskslicestrain,dtype=np.bool_)

print(X.shape)
print(Y.shape)

X = np.expand_dims(X,axis=-1)
Y = np.expand_dims(Y,axis=-1)
print(X.shape)
print(Y.shape)

# TODO: Google how to use the train_test_split from sklearn. We want 20 percent of the values in the validation set. 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# model.fit(x=X_train,y=Y_train,validation_data=(X_val,Y_val),epochs=epochs, batch_size=batchsize)``

# Use this cell to store your model to file. It will be stored as a folder in your notebooks directory
model.save('mymodel')

# Use this cell to load your model back into this notebook, from a version saved on file:
model = tf.keras.models.load_model('mymodel')



# exercise 4

len(X_test)

testimg = X_test[0]
testimg.shape

plt.imshow(X_test[0])
plt.figure()
plt.imshow(Y_test[0])
plt.figure()

 
prediction = model(np.expand_dims(testimg, axis=0) )

plt.imshow(prediction[0])
plt.figure()

# TODO: Fill the blanks
for i in range(10):
    # Display the ground truth mask
    plt.imshow(Y_test[i, :, :, 0], cmap='gray')  # Assuming masks are single-channel and in the last dimension
    plt.title("Ground Truth Mask")
    plt.colorbar()
    # plt.show()

    # Pick the corresponding image from the test set
    test_img = X_test[i]
    
    # Predict the mask using the trained model
    prediction = model.predict(np.expand_dims(test_img, axis=0))
    
    # Display the predicted mask
    plt.imshow(prediction[0, :, :, 0], cmap='gray', interpolation='nearest')
    plt.title("Predicted Mask")
    plt.colorbar()
    # plt.show()



# **QUESTION:** Can you explain why in the predicted masks, the cells seem to have "green edges" (assuming you used the default visualisation scheme from Matplotlib).

# First, we read in the images from the test folder, using the functions defined before
rawImagesTest = loadimages(imagefoldertest)
rawMasksTest = loadimages(maskfoldertest)

# Print the number of images and masks loaded in the test set
print("Number of test images loaded:", len(rawImagesTest))
print("Number of test masks loaded:", len(rawMasksTest))

imgslicestest = returnslices(rawImagesTest,slicesize) 
maskslicestest = returnslices(rawMasksTest,slicesize)

# TODO: Validate that the imgslicestest and maskslicestest lists have the same amount of slices in them. 
# You can either print the lenghts of both lists, or validate it via a boolean expression
# Validate that the number of image slices matches the number of mask slices
if len(imgslicestest) == len(maskslicestest):
    print("The number of image slices and mask slices are the same.")
else:
    print("Mismatch in the number of image slices and mask slices.")


predictionlist = []
for img in imgslicestest:
    predictedmask = model(np.expand_dims(img, axis=0))
    predictionlist.append(predictedmask)
print(len(predictionlist))

predictionlist[0]

roundedpredictions = []
for prediction in predictionlist: 
    roundedprediction = np.rint(prediction)
    roundedpredictions.append(roundedprediction)
print(len(roundedpredictions))

np.max(roundedpredictions)

np.max(maskslicestest)

roundedmaskslicestest = np.array(maskslicestest)/255

np.max(roundedmaskslicestest)


# exercise 5

# We pick one of the slices in the test set to analyse first. Pick a slice that contains something.
index = 8

image = imgslicestest[index]
groundtruth = roundedmaskslicestest[index]
prediction = roundedpredictions[index][0][:,:,0] # This is needed to get the prediction and groundtruth in the same shape
print(groundtruth.shape)
print(prediction.shape)

# Calculate Intersection
intersection = np.logical_and(prediction, groundtruth)

# Calculate Union
union = np.logical_or(prediction, groundtruth)

# Calculate Difference - symmetric difference: elements that are in either of the sets, but not in their intersection.
difference = np.logical_xor(prediction, groundtruth)

# Optionally, calculate the Intersection over Union (IoU), also known as the Jaccard Index, which is a common evaluation metric.
iou = np.sum(intersection) / np.sum(union)

# Plot the six images from above in a grid

# Plot the image
plt.figure(figsize=(16,10))
plt.subplot(2,3,1)
plt.imshow(image)
plt.title('Test Image', fontsize=14)
plt.axis('off')

# Plot the ground truth mask
plt.subplot(2,3,2)
plt.imshow(groundtruth)
plt.title('Ground truth mask', fontsize=12)
plt.axis('off')

# Plot the predicted mask
plt.subplot(2,3,3)
plt.imshow(prediction)
plt.title('Predicted Mask', fontsize=14)
plt.axis('off')

# Plot the Union
plt.subplot(2,3,4)
plt.imshow(union)
plt.title('Union', fontsize=14)
plt.axis('off')

# Plot the Intersect
plt.subplot(2,3,5)
plt.imshow(intersection)
plt.title('Intersect', fontsize=14)
plt.axis('off')

# Plot the difference:
plt.subplot(2,3,6)
plt.imshow(difference)
plt.title('Difference', fontsize=14)
plt.axis('off')


np.sum(intersection) / np.sum(union)

# TODO: Fill in the blanks. Add the union and the intersection as before. 
ioulist = []
for index in range(len(roundedmaskslicestest)):
    groundtruth = roundedmaskslicestest[index]
    prediction = roundedpredictions[index][0][:,:,0]  # Ensure predictions are properly indexed.
    
    if np.sum(groundtruth) > 0:  # Only calculate IoU where there are cells in the ground truth
        union = np.logical_or(prediction, groundtruth)
        intersection = np.logical_and(prediction, groundtruth)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0  # Prevent division by zero
        ioulist.append(iou)
    # If no cells are in the image, IoU is not calculated; it could be treated differently depending on context.

print("Total computed IoU scores:", len(ioulist))
print("Average IoU score:", np.mean(ioulist) if ioulist else 0)

print(np.mean(ioulist))











