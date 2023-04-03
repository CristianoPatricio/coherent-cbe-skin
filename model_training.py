import torch
import torch.nn as nn
import torchvision
from timeit import default_timer as timer

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.tensorboard import SummaryWriter

import numpy as np

import model_params
import model_params as params
from modules.model_builder import get_ccbm_model

from losses import SemanticLoss, CounterLoss, UniquenessLoss, ConceptLoss, CoherenceLoss

from model_parameters import PH2Derm7pt_params, PH2Derm7pt_DLV3_FT_params, PH2Derm7pt_DLV3_params, \
    PH2Derm7pt_Manually_params, \
    PH2_params, PH2_DLV3_FT_params, PH2_DLV3_params, PH2_Manually_params, Derm7pt_params, \
    Derm7pt_DLV3_FT_params, Derm7pt_DLV3_params, Derm7pt_Manually_params, production_raw_params, production_DLV3_params, \
    production_manually_params

from modules import data_setup, engine, utils, model_builder
from modules.utils import set_seeds, create_writer, print_train_time, view_examples_dataloader

from torchinfo import summary

# To run with updated APIs, we need torch 1.12+ and torchvision 0.13+
assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# Setup device-agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using {device} device.")

##################################################
# Define training and val transforms
##################################################

train_transform = A.Compose([
    A.PadIfNeeded(512, 512),
    A.CenterCrop(width=512, height=512),
    A.Resize(width=224, height=224),  # (299, 299) for inception; (224,224) for others
    A.RandomRotate90(),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.PadIfNeeded(512, 512),
    A.CenterCrop(width=512, height=512),
    A.Resize(width=224, height=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

#################################################
# Create training and val dataloaders
#################################################

# PH2Derm7pt (Original)
train_dataloader_ph2derm7pt, val_dataloader_ph2derm7pt, class_names = data_setup.create_dataloaders(
    params=PH2Derm7pt_params,
    train_transform=train_transform,
    val_transform=val_transform)

# PH2Derm7pt (DeepLabV3)
train_dataloader_ph2derm7pt_dlv3, val_dataloader_ph2derm7pt_dlv3, _ = data_setup.create_dataloaders(
    params=PH2Derm7pt_DLV3_params,
    train_transform=train_transform,
    val_transform=val_transform)

# PH2Derm7pt (DeepLabV3 FT on HAM10000)
train_dataloader_ph2derm7pt_dlv3_ft, val_dataloader_ph2derm7pt_dlv3_ft, _ = data_setup.create_dataloaders(
    params=PH2Derm7pt_DLV3_FT_params,
    train_transform=train_transform,
    val_transform=val_transform)

# PH2Derm7pt (Manually)
train_dataloader_ph2derm7pt_manually, val_dataloader_ph2derm7pt_manually, _ = data_setup.create_dataloaders(
    params=PH2Derm7pt_Manually_params,
    train_transform=train_transform,
    val_transform=val_transform)

# PH2 (Original)
train_dataloader_ph2, val_dataloader_ph2, _ = data_setup.create_dataloaders(params=PH2_params,
                                                                            train_transform=train_transform,
                                                                            val_transform=val_transform)

# PH2 (DeepLabV3)
train_dataloader_ph2_dlv3, val_dataloader_ph2_dlv3, _ = data_setup.create_dataloaders(params=PH2_DLV3_params,
                                                                                      train_transform=train_transform,
                                                                                      val_transform=val_transform)

# PH2 (DeepLabV3 FT on HAM10000)
train_dataloader_ph2_dlv3_ft, val_dataloader_ph2_dlv3_ft, _ = data_setup.create_dataloaders(params=PH2_DLV3_FT_params,
                                                                                            train_transform=train_transform,
                                                                                            val_transform=val_transform)

# PH2 (Manually)
train_dataloader_ph2_manually, val_dataloader_ph2_manually, _ = data_setup.create_dataloaders(
    params=PH2_Manually_params,
    train_transform=train_transform,
    val_transform=val_transform)

# Derm7pt (Original)
train_dataloader_derm7pt, val_dataloader_derm7pt, _ = data_setup.create_dataloaders(params=Derm7pt_params,
                                                                                    train_transform=train_transform,
                                                                                    val_transform=val_transform)

# Derm7pt (DeepLabV3)
train_dataloader_derm7pt_dlv3, val_dataloader_derm7pt_dlv3, _ = data_setup.create_dataloaders(
    params=Derm7pt_DLV3_params,
    train_transform=train_transform,
    val_transform=val_transform)

# Derm7pt (DeepLabV3 FT on HAM10000)
train_dataloader_derm7pt_dlv3_ft, val_dataloader_derm7pt_dlv3_ft, _ = data_setup.create_dataloaders(
    params=Derm7pt_DLV3_FT_params,
    train_transform=train_transform,
    val_transform=val_transform)

# Derm7pt (Manually)
train_dataloader_derm7pt_manually, val_dataloader_derm7pt_manually, _ = data_setup.create_dataloaders(
    params=Derm7pt_Manually_params,
    train_transform=train_transform,
    val_transform=val_transform)

# Production (Original)
train_dataloader_production_raw, val_dataloader_production_raw, _ = data_setup.create_dataloaders(
    params=production_raw_params,
    train_transform=train_transform,
    val_transform=val_transform)

# Production (DeepLabV3 FT on HAM10000)
train_dataloader_production_dlv3, val_dataloader_production_dlv3, _ = data_setup.create_dataloaders(
    params=production_DLV3_params,
    train_transform=train_transform,
    val_transform=val_transform)

# Production (Manually)
train_dataloader_production_manually, val_dataloader_production_manually, _ = data_setup.create_dataloaders(
    params=production_manually_params,
    train_transform=train_transform,
    val_transform=val_transform)

# view_examples_dataloader(train_dataloader_ph2derm7pt, class_names)

dtype = torch.FloatTensor

# Load class indicator vectors
class_indicator_vectors = np.load(params.CLASS_INDICATOR_VECTORS).T
print(f"[INFO] Shape of class_indicator_vectors: {class_indicator_vectors.shape}")

# load class word phrase vectors
word_phrase_vector_tensor = torch.from_numpy(np.load(params.CONCEPT_WORD_PHRASE_VECTORS)).type(dtype)
print(f"[INFO] Shape of concept_word_phrase_vector: {word_phrase_vector_tensor.shape}")


#models = ["resnet18", "resnet50", "resnet101", "vgg16", "mobilenetv2", "densenet201", "seresnext"]
models = ["resnet101"]

# Create dataloaders dictionary for the various dataloaders
dataloaders = {#"ph2": [train_dataloader_ph2, val_dataloader_ph2, PH2_params],
               #"ph2derm7pt": [train_dataloader_ph2derm7pt, val_dataloader_ph2derm7pt, PH2Derm7pt_params],
               #"derm7pt": [train_dataloader_derm7pt, val_dataloader_derm7pt, Derm7pt_params],
               "ph2_dlv3_ft": [train_dataloader_ph2_dlv3_ft, val_dataloader_ph2_dlv3_ft, PH2_DLV3_FT_params],
               "ph2derm7pt_dlv3_ft": [train_dataloader_ph2derm7pt_dlv3_ft, val_dataloader_ph2derm7pt_dlv3_ft,
                                   PH2Derm7pt_DLV3_FT_params],
               "derm7pt_dlv3_ft": [train_dataloader_derm7pt_dlv3_ft, val_dataloader_derm7pt_dlv3_ft,
                                 Derm7pt_DLV3_FT_params],
               #"ph2_manually": [train_dataloader_ph2_manually, val_dataloader_ph2_manually, PH2_Manually_params],
               #"ph2derm7pt_manually": [train_dataloader_ph2derm7pt_manually, val_dataloader_ph2derm7pt_manually,
               #                        PH2Derm7pt_Manually_params],
               #"derm7pt_manually": [train_dataloader_derm7pt_manually, val_dataloader_derm7pt_manually,
               #                    Derm7pt_Manually_params]
               }

# OURS
# gammas = {"ph2": [0.6, 0.6, 0.6],
#           "ph2_dlv3_ft": [0.6, 0.6, 0.6],
#           "ph2_manually": [0.6, 0.6, 0.6],
#           "derm7pt": [0.3, 0.7, 0.3],
#           "derm7pt_dlv3_ft": [0.6, 0.5, 0.6],
#           "derm7pt_manually": [0.6, 0.5, 0.5],
#           "ph2derm7pt": [0.4, 0.9, 0.6],
#           "ph2derm7pt_dlv3_ft": [0.4, 0.7, 0.6],
#           "ph2derm7pt_manually": [0.4, 0.7, 0.6]}

# BASELINE
gammas = {"ph2": [None, None, None],
          "ph2_dlv3_ft": [None, None, None],
          "ph2_manually": [None, None, None],
          "derm7pt": [None, None, None],
          "derm7pt_dlv3_ft": [None, None, None],
          "derm7pt_manually": [None, None, None],
          "ph2derm7pt": [None, None, None],
          "ph2derm7pt_dlv3_ft": [None, None, None],
          "ph2derm7pt_manually": [None, None, None]}

comment = "relu_tanh_"

# Set the random seeds
set_seeds(seed=42)

# Keep track of experiment numbers
experiment_number = 0

train_time_start_on_gpu = timer()

# 1. Loop through each DataLoader
for dataloader_name, (train_dataloader, val_dataloader, dataset_params) in dataloaders.items():

    # 2. Loop through each model name and create a new model based on the name
    for i, model_name in enumerate(models):

        gamma = gammas[dataloader_name][1] # 0 - ResNet-101 / 1 - DenseNet-201 / 2 - SEResNeXt

        experiment_number += 1
        if gamma is not None:
            experiment_name = dataloader_name + "_concept_loss_" + str(gamma) + "_" + comment
        else:
            experiment_name = dataloader_name + "_baseline_" + comment
        print(f"[INFO] Experiment number: {experiment_number}")
        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] DataLoader: {dataloader_name}")
        print(f"[INFO] Dataset: {dataset_params.DATASET}")
        print(f"[INFO] Gamma: {gamma}")
        print(f"[INFO] Optimizer: {model_params.OPTIMIZER}\n")

        # Define model
        model = get_ccbm_model(model_name=model_name,
                               params=params,
                               class_ind_vec=class_indicator_vectors,
                               device=device,
                               freeze_layers=True)

        if model_name == "mobilenetv2":
            OUT_DIM = 49
        else:
            OUT_DIM = 196

        # Define loss functions and optimizers
        criterion_s_loss = SemanticLoss(num_concepts=params.NUM_CONCEPTS,
                                        out_dim=OUT_DIM,
                                        emb_dim=params.EMBED_SPACE_DIM,
                                        text_dim=params.TEXT_SPACE_DIM,
                                        word_vectors=word_phrase_vector_tensor,
                                        device=device,
                                        alpha=params.ALPHA)

        if params.OPTIMIZER == 'Adam':
            optimizer_s_loss = torch.optim.Adam(criterion_s_loss.parameters(), lr=params.LEARNING_RATE_1,
                                                weight_decay=params.WEIGHT_DECAY)
        else:
            optimizer_s_loss = torch.optim.RMSprop(criterion_s_loss.parameters(), lr=params.LEARNING_RATE_1,
                                                weight_decay=params.WEIGHT_DECAY)

        criterion_c_loss = CounterLoss(beta=params.BETA,
                                       device=device)

        criterion_u_loss = UniquenessLoss(num_concepts=params.NUM_CONCEPTS,
                                          device=device)

        # criterion_concept_loss = ConceptLoss(device=device)
        if not params.BASELINE:
            criterion_concept_loss = CoherenceLoss(device=device)
        else:
            criterion_concept_loss = None

        criterion_classification_loss = torch.nn.CrossEntropyLoss()
        if params.OPTIMIZER == 'Adam':
            optimizer_classification_loss = torch.optim.Adam(model.parameters(),
                                                             lr=params.LEARNING_RATE_1,
                                                             weight_decay=params.WEIGHT_DECAY)
        else:
            optimizer_classification_loss = torch.optim.RMSprop(model.parameters(),
                                                             lr=params.LEARNING_RATE_1,
                                                             weight_decay=params.WEIGHT_DECAY)

        # 3. Train only newly added variables
        print(f"[INFO] Training stage: 1/3")
        _, model_1 = engine.train(model=model,
                                  train_dataloader=train_dataloader,
                                  val_dataloader=val_dataloader,
                                  optimizer_s_loss=optimizer_s_loss,
                                  optimizer_classification_loss=optimizer_classification_loss,
                                  criterion_s_loss=criterion_s_loss,
                                  criterion_c_loss=criterion_c_loss,
                                  criterion_u_loss=criterion_u_loss,
                                  criterion_concept_loss=criterion_concept_loss,
                                  criterion_classification_loss=criterion_classification_loss,
                                  lambda_value=params.LAMBDA_VALUE,
                                  gamma_value=gamma, #params.GAMMA_VALUE,
                                  epochs=params.MAX_NUM_EPOCHS_1,
                                  last_epochs=0,
                                  device=device,
                                  writer=create_writer(experiment_name=experiment_name,
                                                       model_name=model_name,
                                                       extra=f"{params.MAX_NUM_EPOCHS_1}_epochs"),
                                  params=dataset_params,
                                  comments=comment,
                                  model_name=model_name)

        # 4. Fine-tune all variables in the network
        print(f"[INFO] Training stage: 2/3")
        if params.OPTIMIZER == 'Adam':
            optimizer_classification_loss = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE_2,
                                                             weight_decay=params.WEIGHT_DECAY)
        else:
            optimizer_classification_loss = torch.optim.RMSprop(model.parameters(), lr=params.LEARNING_RATE_2,
                                                             weight_decay=params.WEIGHT_DECAY)

        # Create the model
        model = get_ccbm_model(model_name=model_name,
                               params=params,
                               class_ind_vec=class_indicator_vectors,
                               device=device,
                               freeze_layers=False)

        # Load model weights from model_1 checkpoint
        model.load_state_dict(model_1.state_dict())

        _, model_2 = engine.train(model=model,
                                  train_dataloader=train_dataloader,
                                  val_dataloader=val_dataloader,
                                  optimizer_s_loss=optimizer_s_loss,
                                  optimizer_classification_loss=optimizer_classification_loss,
                                  criterion_s_loss=criterion_s_loss,
                                  criterion_c_loss=criterion_c_loss,
                                  criterion_u_loss=criterion_u_loss,
                                  criterion_concept_loss=criterion_concept_loss,
                                  criterion_classification_loss=criterion_classification_loss,
                                  lambda_value=params.LAMBDA_VALUE,
                                  gamma_value=gamma, #params.GAMMA_VALUE,
                                  epochs=params.MAX_NUM_EPOCHS_2,
                                  last_epochs=params.MAX_NUM_EPOCHS_1,
                                  device=device,
                                  writer=create_writer(experiment_name=experiment_name,
                                                       model_name=model_name,
                                                       extra=f"{params.MAX_NUM_EPOCHS_2}_epochs"),
                                  params=dataset_params,
                                  comments=comment,
                                  model_name=model_name)

        # 5. Fine-tune all the FC layer
        print(f"[INFO] Training stage: 3/3")
        if params.OPTIMIZER == 'Adam':
            optimizer_classification_loss = torch.optim.Adam(model.parameters(), lr=params.LEARNING_RATE_3,
                                                             weight_decay=params.WEIGHT_DECAY)
        else:
            optimizer_classification_loss = torch.optim.RMSprop(model.parameters(), lr=params.LEARNING_RATE_3,
                                                             weight_decay=params.WEIGHT_DECAY)

        # Create the model
        model = get_ccbm_model(model_name=model_name,
                               params=params,
                               class_ind_vec=class_indicator_vectors,
                               device=device,
                               freeze_layers=True,
                               train_classifier=True)

        # Load model weights from model_2 checkpoint
        model.load_state_dict(model_2.state_dict())

        engine.train(model=model,
                     train_dataloader=train_dataloader,
                     val_dataloader=val_dataloader,
                     optimizer_s_loss=optimizer_s_loss,
                     optimizer_classification_loss=optimizer_classification_loss,
                     criterion_s_loss=criterion_s_loss,
                     criterion_c_loss=criterion_c_loss,
                     criterion_u_loss=criterion_u_loss,
                     criterion_concept_loss=criterion_concept_loss,
                     criterion_classification_loss=criterion_classification_loss,
                     lambda_value=params.LAMBDA_VALUE,
                     gamma_value=gamma, #params.GAMMA_VALUE,
                     epochs=params.MAX_NUM_EPOCHS_3,
                     last_epochs=params.MAX_NUM_EPOCHS_2 + params.MAX_NUM_EPOCHS_1,
                     device=device,
                     writer=create_writer(experiment_name=experiment_name,
                                          model_name=model_name,
                                          extra=f"{params.MAX_NUM_EPOCHS_3}_epochs"),
                     params=dataset_params,
                     comments=comment,
                     model_name=model_name)

        print("-" * 100 + "\n")

train_time_end_on_gpu = timer()

# 9. Calculate training time
print_train_time(start=train_time_start_on_gpu,
                 end=train_time_end_on_gpu,
                 device=device)
