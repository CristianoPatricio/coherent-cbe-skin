INDICATOR_VECTORS = "data/image_indicator_vectors_8_concepts.npy"
CLASS_INDICATOR_VECTORS = "data/class_indicator_vectors_8_concepts.npy"
CONCEPT_WORD_PHRASE_VECTORS = "data/concept_word_phrase_vectors_8_concepts.npy"
CONCEPT_TEXT_FILE = "data/concepts.txt"
CLASS_CONCEPTS = "data/skin_concepts_8.npy"
NOUNS_ADJECTIVES_FILE = "data/image_wise_noun_adjectives.npy"
GLOVE_FILE_NAME = "data/glove.6B.50d.txt"
IMG_SIZE = (512, 512)

NUM_CLASSES = 2
CLASS_NAME_FILE = "data/classes.txt"

# -------- CCBM parameters --------
LAMBDA_VALUE = 0.4
GAMMA_VALUE = 0.4
BETA = 0.5
ALPHA = 1
OUT_DIM = 196
EMBED_SPACE_DIM = 24
TEXT_SPACE_DIM = 50
NUM_CONCEPTS = 8
TOP_R = 4
BASELINE = False
CALCULATE_FILTER_DISTRIBUTION = False
PLOT_HISTOGRAM_FILTERS = False
OPTIMIZER = 'Adam'  # 'Adam'
SAVE_LAST = False

# -------- Training parameters --------
INPUT_SHAPE = 256
BATCH_SIZE = 16
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16
NUM_WORKERS = 12
MAX_NUM_EPOCHS_1 = 100  # 100
MAX_NUM_EPOCHS_2 = 30   # 30
MAX_NUM_EPOCHS_3 = 10   # 10
LEARNING_RATE_1 = 1e-3
LEARNING_RATE_2 = 1e-4
LEARNING_RATE_3 = 1e-4
DROP_OUT_KEEP_PROB = 0.5
WEIGHT_DECAY = 5e-4
VGG_MEAN = [123.68, 116.78, 103.94]
