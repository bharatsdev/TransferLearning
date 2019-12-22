import json

import numpy as np
import progressbar
from keras.models import load_model

from dogs_vs_cats.config import dog_vs_cat_config as config
from preprocess.croppreprocessor import CropPreProcessor
from preprocess.imagetoarraypreprocessor import ImageToArrayPreprocess
from preprocess.meanpreprocessor import MeanPreprocessor
from preprocess.simplepreprocessor import SimplePreprocessor
from utils.hdf5datasetgenerator import HDF5DatasetGenerator
from utils.ranked import rank5_accuracy

print('[INFO] : Load the RGB means for the training set....!')
means = json.loads(open(config.DATASET_MEAN).read())

print('[INFO] : Initialize the image preprocessor...!')
sp = SimplePreprocessor(config.IMG_WIDTH, config.IMG_HEIGHT)
mp = MeanPreprocessor(means.R, means.G, means.B)
cp = CropPreProcessor(config.IMG_HEIGHT, config.IMG_WIDTH)
iap = ImageToArrayPreprocess()

print('[INFO] : Load the pre-train model')
model = load_model(config.MODEL_PATH)
print('[INOF] : Initialize the  testing dataset generator, then make predictions  on the testing data')
print('[INFO] : Predicting on test data (no cropping)....!')

testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[sp, mp, iap], classes=2)
predictions = model.predict_generator(testGen.generator(), steps=testGen.numImages // 64, max_queue_size=10)

# Compute the rank-1 & rank-5 accuracies
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print('[INFO] : rank-1 : {:.2f}%'.format(rank1))
testGen.close()

print('[INFO] : Re-initialize the testing set generator, this time excluding the SimplePreprocessor..!')
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[mp], classes=2)

predictions = []

print('[INFO] : Initialize the Progress bar...!')
widgets = ['Evaluating: ', progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages // 64, widgets=widgets).start()

# loop over a single pass of the test data
for (idx, (images, labels)) in enumerate(testGen.generator(passes=1)):
    for image in images:
        # Apply the crop  preprocessor to the images to generate 10 separates crops
        # and then convert them image to arrays
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype='float32')

        # Make the predictions on the crop and then average  them together to obtain the final predictions
        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))
    # Update the progressbar
    pbar.update(idx)

# compute the rank-1 accuracy
pbar.finish()
print('[INFO] : Predicting on test data (with crops)....!')
(rank1, _) = rank5_accuracy(predictions, testGen.db['labels'])
print('[INFO] : rank-1 : {:.2f}%'.format(rank1 * 100))
testGen.close()
