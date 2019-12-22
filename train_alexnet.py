import json
import os

import matplotlib
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from dogs_vs_cats.config import dog_vs_cat_config as config
from callbacks.trainingmonitor import TrainingMonitor
from nn.conv.alexnet import AlexNet
from preprocess.imagetoarraypreprocessor import ImageToArrayPreprocess
from preprocess.meanpreprocessor import MeanPreprocessor
from preprocess.patchpreprocessor import PatchPreprocessor
from preprocess.simplepreprocessor import SimplePreprocessor
from utils.hdf5datasetgenerator import HDF5DatasetGenerator

matplotlib.use('Agg')

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=0.20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
# load the RGB means for thr training set
means = json.loads(open(config.DATASET_MEAN).read())
print(means)

# Initialize the pre-processor

sp = SimplePreprocessor(config.IMG_WIDTH, config.IMG_HEIGHT)
pp = PatchPreprocessor(config.IMG_WIDTH, config.IMG_HEIGHT)
mp = MeanPreprocessor(means['R'], means['G'], means['B'])
iap = ImageToArrayPreprocess()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=aug, preprocessors=[pp, mp, iap], classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=aug, preprocessors=[sp, mp, iap], classes=2)

print('[INFO] : Compiling model....!')
opt = Adam(lr=1e-3)
model = AlexNet.build(width=227, height=227, depth=3, classes=2, reg=0.0002)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

print('[INFO] : Construct the set of callbacks...!')
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
print(path)
from slacknotify.slacknotify import SendNotification

slackNoticaiton = SendNotification()
callbacks = [TrainingMonitor(path, notificaiton=slackNoticaiton)]

try:
    slackNoticaiton.train_start()

    model.fit_generator(trainGen.generator(),
                        steps_per_epoch=trainGen.numImages / config.BATCH_SIZE,
                        validation_data=valGen.generator(),
                        validation_steps=valGen.numImages / config.BATCH_SIZE,

                        epochs=config.EPOCHS,
                        max_queue_size=10,
                        callbacks=callbacks,
                        verbose=1)
except Exception as ex:
    slackNoticaiton.trainCrash(ex)

# save the model to file
print("[INFO] : Serializing model...")
model.save(config.MODEL_PATH, overwrite=True)

# close the HDF5 database
trainGen.close()
valGen.close()
