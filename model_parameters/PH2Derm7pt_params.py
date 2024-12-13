from datetime import datetime

x = datetime.now()

# Dataset Information
DATASET = "PH2Derm7pt"
IMAGES_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/PH2Derm7pt/images"
MASKS_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/PH2Derm7pt/PH2Derm7pt_GT_Masks"
#IMAGES_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/ISIC 2018/train_images"
#MASKS_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/ISIC 2018/ISIC_2018_Masks"
FILE_EXTENSION = ""
NUM_WORKERS = 12
NUM_CLASSES = 2
IMG_SIZE = (512, 512)
TRAIN_FE = False
TRAIN_FILENAME = "data/PH2Derm7pt_train.csv"
VALIDATION_FILENAME = "data/PH2Derm7pt_validation.csv"
TEST_FILENAME = "data/PH2Derm7pt_test.csv"
#TEST_FILENAME = "data/ISIC_2018_test.csv"
IMG_TYPE = "Raw"

# Model Parameters
MODEL_NAME = "DenseNet-121"
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 16

# Results Information
FIGURE_NAME = f"figures/Fig_Loss_{MODEL_NAME}_{DATASET}_{x.strftime('%Y%m%d%H%M%S')}"
MODEL_SUB_DIR = f"Model_{x.strftime('%Y%m%d%H%M%S')}_{MODEL_NAME}_{DATASET}"
MODEL_TO_SAVE_NAME = f"model_{DATASET}"