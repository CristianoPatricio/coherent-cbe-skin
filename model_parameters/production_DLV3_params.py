from datetime import datetime

x = datetime.now()

# Dataset Information
DATASET = "production_dlv3"
IMAGES_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/PH2Derm7pt/images_DeepLab_FT_HAM10000"
MASKS_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/PH2Derm7pt/PH2Derm7pt_Masks_HAM10000"
FILE_EXTENSION = "png"
NUM_WORKERS = 12
NUM_CLASSES = 2
IMG_SIZE = (512, 512)
TRAIN_FE = False
TRAIN_FILENAME = "data/train_split2.csv"
VALIDATION_FILENAME = "data/val_split2.csv"
TEST_FILENAME = "data/PH2_test.csv"
IMG_TYPE = "Segmented (DeepLabV3)"

# Model Parameters
MODEL_NAME = "DenseNet-121"
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 16

# Results Information
FIGURE_NAME = f"figures/Fig_Loss_{MODEL_NAME}_{DATASET}_{x.strftime('%Y%m%d%H%M%S')}"
MODEL_SUB_DIR = f"Model_{x.strftime('%Y%m%d%H%M%S')}_{MODEL_NAME}_{DATASET}"
MODEL_TO_SAVE_NAME = f"model_{DATASET}"