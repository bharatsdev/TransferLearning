# import the necessary packages
import os
import random

import numpy as np
import progressbar
from imutils import paths
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder

from dogs_vs_cats.config import dog_vs_cat_config as config
from utils.hdf5datasetwriter import HDF5DatasetWriter

# grab the list of images that we will be describing then randomly shuffle them to allow for easy training  and testing
# splits via array slicing during  training time
print('[INFO] : Loading images...!')
imagePaths = list(paths.list_images(config.IMAGES_PATH))
random.shuffle(imagePaths)
print(imagePaths)


print('[INFO] : Extract the class labels from the image path then encode the labels...!')
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
print(labels)

le = LabelEncoder()
labels = le.fit_transform(labels)

print('[INFO] : Load the RestNet....!')
model = ResNet50(weights='imagenet', include_top=False)

# initialize the HDF5 dataset writer, then store the class label names in the dataset
dataset = HDF5DatasetWriter((len(imagePaths), 100352), config.EXTRACT_FEATURES_HDF5, dataKey="features", bufSize=10000)
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

# loop over the images in batches
for i in np.arange(0, len(imagePaths), config.BATCH_SIZE):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    batchPaths = imagePaths[i:i + config.BATCH_SIZE]
    batchLabels = labels[i:i + config.BATCH_SIZE]
    batchImages = []

    # loop over the images and labels in the current batch
    for (j, imagePath) in enumerate(batchPaths):
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as our actual features
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=config.BATCH_SIZE)
    print(features.shape)

    # reshape the features so that each image is represented by
    # a flattened feature vector of the `MaxPooling2D` outputs
    features = features.reshape((features.shape[0], 100352))

    # add the features and labels to our HDF5 dataset
    dataset.add(features, batchLabels)
    pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()
