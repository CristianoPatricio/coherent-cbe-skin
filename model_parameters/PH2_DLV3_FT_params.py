from datetime import datetime

x = datetime.now()

# Dataset Information
DATASET = "PH2_DeepLabV3_FT"
IMAGES_DIR = "/home/user/Datasets/PH2Dataset/PH2_Segmented_Images_DLab_Trained_on_HAM10000"
MASKS_DIR = "/home/user/Datasets/PH2Dataset/PH2_Masks_HAM10000"
FILE_EXTENSION = "png"
NUM_WORKERS = 12
NUM_CLASSES = 2
IMG_SIZE = (512, 512)
TRAIN_FE = False
TRAIN_FILENAME = "data/PH2_train.csv"
VALIDATION_FILENAME = "data/PH2_validation.csv"
TEST_FILENAME = "data/PH2_test.csv"
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