import matplotlib.pyplot as plt
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model_parameters import PH2Derm7pt_params, PH2Derm7pt_DLV3_FT_params, PH2Derm7pt_DLV3_params, \
    PH2Derm7pt_Manually_params, \
    PH2_params, PH2_DLV3_FT_params, PH2_DLV3_params, PH2_Manually_params, Derm7pt_params, \
    Derm7pt_DLV3_FT_params, Derm7pt_DLV3_params, Derm7pt_Manually_params, production_raw_params, \
    production_DLV3_params, production_manually_params

import model_params
import model_params as params
from modules import data_setup, engine, model_builder
from modules.utils import set_seeds, create_writer, print_train_time, write_to_txt, plot_roc_curve

# To run with updated APIs, we need torch 1.12+ and torchvision 0.13+
assert int(torch.__version__.split(".")[1]) >= 12, "torch version should be 1.12+"
assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")

# Setup device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using {device} device")

##################################################
# Define transform
##################################################

transform = A.Compose([
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
test_dataloader_ph2derm7pt, class_names = data_setup.create_dataloader_for_evaluation(params=PH2Derm7pt_params,
                                                                                      transform=transform)

# PH2Derm7pt (DeepLabV3 FT on HAM10000)
test_dataloader_ph2derm7pt_dlv3_ft, class_names = data_setup.create_dataloader_for_evaluation(
    params=PH2Derm7pt_DLV3_FT_params,
    transform=transform)

# PH2Derm7pt (Manually)
test_dataloader_ph2derm7pt_manually, class_names = data_setup.create_dataloader_for_evaluation(
    params=PH2Derm7pt_Manually_params,
    transform=transform)

# PH2 (Original)
test_dataloader_ph2, _ = data_setup.create_dataloader_for_evaluation(params=PH2_params,
                                                                     transform=transform)

# PH2 (DeepLabV3 FT on HAM1000)
test_dataloader_ph2_dlv3_ft, _ = data_setup.create_dataloader_for_evaluation(params=PH2_DLV3_FT_params,
                                                                             transform=transform)

# PH2 (Manually)
test_dataloader_ph2_manually, _ = data_setup.create_dataloader_for_evaluation(params=PH2_Manually_params,
                                                                              transform=transform)

# Derm7pt (Original)
test_dataloader_derm7pt, _ = data_setup.create_dataloader_for_evaluation(params=Derm7pt_params,
                                                                         transform=transform)

# Derm7pt (Manually)
test_dataloader_derm7pt_manually, _ = data_setup.create_dataloader_for_evaluation(params=Derm7pt_Manually_params,
                                                                                  transform=transform)

# Derm7pt (DeepLabV3 FT on HAM10000)
test_dataloader_derm7pt_dlv3_ft, _ = data_setup.create_dataloader_for_evaluation(params=Derm7pt_DLV3_FT_params,
                                                                                 transform=transform)

# Production (Original)
test_dataloader_production_raw, _ = data_setup.create_dataloader_for_evaluation(params=production_raw_params,
                                                                                transform=transform)

# Production (Manually)
test_dataloader_production_manually, _ = data_setup.create_dataloader_for_evaluation(params=production_manually_params,
                                                                                     transform=transform)

# Production (DeepLabV3 FT on HAM10000)
test_dataloader_production_dlv3_ft, _ = data_setup.create_dataloader_for_evaluation(params=production_DLV3_params,
                                                                                    transform=transform)

###############################################
# Set up evaluation experiments
###############################################

models = ["densenet201"]  # "resnet101", "seresnext"

# Create dataloaders dictionary for the various dataloaders
dataloaders = {#"ph2": [test_dataloader_ph2, PH2_params],
               #"ph2derm7pt": [test_dataloader_ph2derm7pt, PH2Derm7pt_params],
               #"derm7pt": [test_dataloader_derm7pt, Derm7pt_params],
               "ph2_dlv3_ft": [test_dataloader_ph2_dlv3_ft, PH2_DLV3_FT_params],
               #"ph2derm7pt_dlv3_ft": [test_dataloader_ph2derm7pt_dlv3_ft, PH2Derm7pt_DLV3_FT_params],
               #"derm7pt_dlv3_ft": [test_dataloader_derm7pt_dlv3_ft, Derm7pt_DLV3_FT_params],
               #"ph2_manually": [test_dataloader_ph2_manually, PH2_Manually_params],
               #"ph2derm7pt_manually": [test_dataloader_ph2derm7pt_manually, PH2Derm7pt_Manually_params],
               #"derm7pt_manually": [test_dataloader_derm7pt_manually, Derm7pt_Manually_params]
               }

# OUR METHOD
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

# 1. Set the random seeds
set_seeds(seed=42)

# 2. Load pretrained model
# Keep track of experiment numbers
experiment_number = 0

time_start = timer()

# 2. Loop through each DataLoader
for dataloader_name, (test_dataloader, dataset_params) in dataloaders.items():

    # 3. Loop through each model name and create a new model based on the name
    for i, model_name in enumerate(models):

        gamma = gammas[dataloader_name][1]  # 0 - ResNet-101 / 1 - DenseNet-201 / 2 - SEResNeXt

        experiment_number += 1
        experiment_name = dataloader_name
        print(f"[INFO] Experiment number: {experiment_number}")
        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] DataLoader: {dataloader_name}")
        print(f"[INFO] Dataset: {dataset_params.DATASET}")
        print(f"[INFO] Gamma: {gamma}")

        # 4. Load pretrained model
        model = model_builder.load_ccbm_model(model=model_name,
                                              model_name=model_name,
                                              params=params,
                                              dataset_params=dataset_params,
                                              gamma=gamma,
                                              comments=comment,
                                              device=device)

        # 5. Create the loss function
        loss_fn = nn.CrossEntropyLoss()

        # 6. Evaluate on the test dataloader
        test_loss, test_acc, class_report, conf_matrix, gt_concepts, predicted_concepts, auc, bacc, SE, SP = engine.evaluate(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device,
            params=dataset_params,
            model_name=model_name,
            baseline=True if gamma is None else False,
            plot_results=False)

        # 7. Write results to a TXT file
        if not model_params.CALCULATE_FILTER_DISTRIBUTION:
            write_to_txt(save_dir=f"results/{model_name}",
                         params=dataset_params,
                         accuracy=test_acc,
                         class_report=class_report,
                         conf_matrix=conf_matrix,
                         auc=auc,
                         bacc=bacc,
                         sensitivity=SE,
                         specificity=SP,
                         gt_concepts=np.asarray(gt_concepts),
                         predicted_concepts=np.asarray(predicted_concepts),
                         gamma=gamma,
                         comments=comment,
                         model_name=model_name)

        print("-" * 100 + "\n")

time_stop = timer()

print(f"[INFO] Task completed in {(time_stop - time_start):.4f} seconds.")
