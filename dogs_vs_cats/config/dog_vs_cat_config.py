# define the paths to the images directory
IMAGES_PATH = "datasets/kaggle_dogs_vs_cats/train"

# Image Dim
IMG_WIDTH = 227
IMG_HEIGHT = 227
IMG_CHANNEL = 3

# since we do not have validation data or access to the testing labels
# we need to take a number of images from the training data and use them instead

NUM_CLASSES = 2
NUM_VAL_IMAGES = 1250 * NUM_CLASSES
NUM_TEST_IMAGES = 1250 * NUM_CLASSES
BATCH_SIZE = 128
EPOCHS = 75

# define the path to the output training, validation, and testing HDF5 files
TRAIN_HDF5 = "datasets/kaggle_dogs_vs_cats/hdf5/train.hdf5"
EXTRACT_FEATURES_HDF5 = "datasets/kaggle_dogs_vs_cats/hdf5/extract.hdf5"
VAL_HDF5 = "datasets/kaggle_dogs_vs_cats/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/kaggle_dogs_vs_cats/hdf5/test.hdf5"

# path to the output model file
MODEL_PATH = "../datasets/output/alexnet_dogs_vs_cats.model"
MODEL_PATH = "../datasets/output/dogs_vs_cats.model"

# define the path to the dataset mean
# DATASET_MEAN = "../datasets/output/dogs_vs_cats_mean.json"
DATASET_MEAN = "datasets/output/dogs_vs_cats_mean.json"

# define the path to the output directory used for storing plots, classification reports, etc.
OUTPUT_PATH = "datasets/output"
