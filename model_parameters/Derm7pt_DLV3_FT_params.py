from datetime import datetime

x = datetime.now()

# Dataset Information
DATASET = "Derm7pt_DeepLabV3_FT"
IMAGES_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/Derm7pt/Derm7pt_Segmented_Images_DLAB_Trained_on_HAM10000"
MASKS_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/Derm7pt/Derm7pt_Masks_HAM10000"
#IMAGES_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/ISIC 2018/ISIC_2018_DLV3"
#MASKS_DIR = "/home/cristianopatricio/Desktop/PhD/Datasets/Skin/ISIC 2018/ISIC_2018_DLV3_Masks"
FILE_EXTENSION = "png"
NUM_WORKERS = 12
NUM_CLASSES = 2
IMG_SIZE = (512, 512)
TRAIN_FE = False
TRAIN_FILENAME = "data/derm7pt_train_seg.csv"
VALIDATION_FILENAME = "data/derm7pt_validation_seg.csv"
TEST_FILENAME = "data/derm7pt_test_seg.csv"
#TEST_FILENAME = "data/ISIC_2018_test.csv"
IMG_TYPE = "Segmented (DeepLabV3 FT on HAM10000)"

# Model Parameters
MODEL_NAME = "DenseNet-121"
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 16

# Results Information
FIGURE_NAME = f"figures/Fig_Loss_{MODEL_NAME}_{DATASET}_{x.strftime('%Y%m%d%H%M%S')}"
MODEL_SUB_DIR = f"Model_{x.strftime('%Y%m%d%H%M%S')}_{MODEL_NAME}_{DATASET}"
MODEL_TO_SAVE_NAME = f"model_{DATASET}"