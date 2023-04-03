"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import timm

import numpy as np

from modules.utils import set_seeds
import model_params

class CCBM(nn.Module):
    """Concept Encoder

    Args:
        input_shape: An integer indicating number of input channels.
        num_concepts: An integer indicating number of hidden units between layers.
        num_classes: An integer indicating number of output units.
    """

    def __init__(self, input_shape, num_concepts, num_classes):
        super(CCBM, self).__init__()

        self.concept_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=num_concepts,
                      kernel_size=[1, 1]),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=num_concepts,
                      out_features=num_classes)
        )

    def forward(self, x):
        x = self.concept_layer(x)
        output_concept_layer = x

        # Global Average Pooling
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        output_gap = x
        x = self.classifier(x)

        return x, output_gap, output_concept_layer


def ccbm_vgg16(params,
               class_ind_vec,
               device,
               freeze_layers=True,
               train_classifier=False,
               ):
    """

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained VGG16
    vgg16_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    vgg16 = torchvision.models.vgg16(weights=vgg16_weights)
    vgg16 = nn.Sequential(
        *list(vgg16.features.children())[:-1]
    )

    # Set seeds
    set_seeds()

    # Create the CCBM
    ccbm = CCBM(input_shape=512,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(vgg16, ccbm).to(device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze VGG16 layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze VGG16 layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def ccbm_resnet18(params,
                  class_ind_vec,
                  device,
                  freeze_layers=True,
                  train_classifier=False,
                  ):
    """Creates a ResNet18 Model

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained ResNet101
    resnet18_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    resnet18 = torchvision.models.resnet18(weights=resnet18_weights)
    resnet18 = nn.Sequential(
        *list(resnet18.children())[:-3]
    )

    # Set seeds
    set_seeds()

    # Create the CCbm
    ccbm = CCBM(input_shape=256,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(resnet18, ccbm).to(device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze VGG16 layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze VGG16 layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def ccbm_resnet50(params,
                  class_ind_vec,
                  device,
                  freeze_layers=True,
                  train_classifier=False,
                  ):
    """Creates a ResNet50 Model

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained ResNet101
    resnet50_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    resnet50 = torchvision.models.resnet50(weights=resnet50_weights)
    resnet50 = nn.Sequential(
        *list(resnet50.children())[:-3]
    )

    # Set seeds
    set_seeds()

    # Create the CCbm
    ccbm = CCBM(input_shape=1024,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(resnet50, ccbm).to(device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze VGG16 layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze VGG16 layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def ccbm_resnet101(params,
                   class_ind_vec,
                   device,
                   freeze_layers=True,
                   train_classifier=False,
                   ):
    """Creates a ResNet101 Model

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained ResNet101
    resnet101_weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
    resnet101 = torchvision.models.resnet101(weights=resnet101_weights)
    resnet101 = nn.Sequential(
        *list(resnet101.children())[:-3]
    )

    # Set seeds
    set_seeds()

    # Create the CCbm
    ccbm = CCBM(input_shape=1024,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(resnet101, ccbm).to(device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze VGG16 layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze VGG16 layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def ccbm_densenet121(params,
                     class_ind_vec,
                     device,
                     freeze_layers=True,
                     train_classifier=False,
                     ):
    """Creates a DenseNet-121 Model

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained ResNet101
    densenet121_weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    densenet121 = torchvision.models.densenet121(weights=densenet121_weights)
    densenet121 = nn.Sequential(
        *list(densenet121.features.children())[:-2],
        # *list(densenet121.features.transition3.children())[:-2]
    )

    # Set seeds
    set_seeds()

    # Create the CCbm
    ccbm = CCBM(input_shape=1280,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(densenet121, ccbm).to(device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def ccbm_densenet161(params,
                     class_ind_vec,
                     device,
                     freeze_layers=True,
                     train_classifier=False,
                     ):
    """Creates a DenseNet-161 Model

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained ResNet101
    densenet161_weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
    densenet161 = torchvision.models.densenet161(weights=densenet161_weights)
    densenet161 = nn.Sequential(
        *list(densenet161.features.children())[:-3],
        *list(densenet161.features.transition3.children())[:-1],
        nn.AvgPool2d(kernel_size=(1, 1))
    )

    # Set seeds
    set_seeds()

    # Create the CCbm
    ccbm = CCBM(input_shape=params.INPUT_SHAPE,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(densenet161, ccbm).to(device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def ccbm_seresnext26d_32x4d(params,
                            class_ind_vec,
                            device,
                            freeze_layers=True,
                            train_classifier=False,
                            ):
    """Creates a seresnext26d_32x4d Model

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained seresnext26d_32x4d
    seresnext26d_32x4d = timm.create_model('seresnext26d_32x4d', pretrained=True)

    seresnext26d_32x4d = nn.Sequential(
        *list(seresnext26d_32x4d.children())[:-3],
        # *list(seresnext26d_32x4d.layer4[0].children())[:-12]
    )

    # Set seeds
    set_seeds()

    # Create the CCbm
    ccbm = CCBM(input_shape=1024,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(seresnext26d_32x4d, ccbm).to(
        device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def ccbm_densenet201(params,
                     class_ind_vec,
                     device,
                     freeze_layers=True,
                     train_classifier=False,
                     ):
    """Creates a Densenet-201 Model

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained seresnext26d_32x4d
    densenet201_weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
    densenet201 = torchvision.models.densenet201(weights=densenet201_weights)

    densenet201 = nn.Sequential(
        *list(densenet201.features.children())[:-3],
        *list(densenet201.features.transition3.children())[:-1],
        nn.AvgPool2d(kernel_size=(1, 1))
    )

    # Set seeds
    set_seeds()

    # Create the CCbm
    ccbm = CCBM(input_shape=896,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(densenet201, ccbm).to(
        device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def ccbm_mobilenetv2(params,
                     class_ind_vec,
                     device,
                     freeze_layers=True,
                     train_classifier=False,
                     ):
    """Creates a MobileNetV2 Model

    :param params: A python file containing the hyperparameters values.
    :param class_ind_vec: The class indicator vector.
    :param freeze_layers: If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
    :param train_classifier: If 'True' only FC layers are trainable. Default: False.
    :param device: 'cuda' or 'cpu'.
    :return: A PyTorch Model.
    """

    dtype = torch.FloatTensor

    # Load model weights and pretrained seresnext26d_32x4d
    mobilenetv2_weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    mobilenetv2 = torchvision.models.mobilenet_v2(weights=mobilenetv2_weights)
    mobilenetv2 = nn.Sequential(
        *list(mobilenetv2.children())[:-1]
    )

    # Set seeds
    set_seeds()

    # Create the CCbm
    ccbm = CCBM(input_shape=1280,
                num_concepts=params.NUM_CONCEPTS,
                num_classes=params.NUM_CLASSES)

    # Initialize FC layer with custom weights
    init = np.multiply(0.1 * np.ones((params.NUM_CONCEPTS, params.NUM_CLASSES)), class_ind_vec)
    ccbm.classifier[0].weight.data = torch.from_numpy(init).type(dtype).T

    # Add the CCbm to the backbone model
    model = nn.Sequential(mobilenetv2, ccbm).to(
        device)  # model[0] -> CNN | model[1] -> concept layer + FC layers

    if freeze_layers:  # To train only the concept layer + FC layer
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

    if train_classifier:
        # Freeze Densenet layers
        for param in model[0].parameters():
            param.requires_grad = False

        for param in model[1].concept_layer.parameters():
            param.requires_grad = False

    return model


def load_ccbm_vgg16(model_params,
                    dataset_params,
                    model_name,
                    device):
    """Loads the VGG16 model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained VGG16
    vgg16_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    vgg16 = torchvision.models.vgg16(weights=vgg16_weights)
    vgg16 = nn.Sequential(
        *list(vgg16.features.children())[:-1]
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=512,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(vgg16, ccbm).to(device)

    saved_model_dir_path = f"saved_models/vgg16/model_{dataset_params.DATASET}_{model_name}_best.pth"
    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def load_ccbm_resnet50(model_params,
                       dataset_params,
                       model_name,
                       device):
    """Loads the ResNet101 model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained ResNet101
    resnet50_weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    resnet50 = torchvision.models.resnet50(weights=resnet50_weights)
    resnet50 = nn.Sequential(
        *list(resnet50.children())[:-3]
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=1024,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(resnet50, ccbm).to(device)

    saved_model_dir_path = f"saved_models/resnet50/model_{dataset_params.DATASET}_{model_name}_best.pth"
    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def load_ccbm_resnet18(model_params,
                       dataset_params,
                       model_name,
                       device):
    """Loads the ResNet18 model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained ResNet101
    resnet18_weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    resnet18 = torchvision.models.resnet18(weights=resnet18_weights)
    resnet18 = nn.Sequential(
        *list(resnet18.children())[:-3]
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=256,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(resnet18, ccbm).to(device)

    saved_model_dir_path = f"saved_models/resnet18/model_{dataset_params.DATASET}_{model_name}_best.pth"
    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def load_ccbm_resnet101(model_params,
                        dataset_params,
                        model_name,
                        gamma,
                        comments,
                        device):
    """Loads the ResNet101 model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained ResNet101
    resnet101_weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
    resnet101 = torchvision.models.resnet101(weights=resnet101_weights)
    resnet101 = nn.Sequential(
        *list(resnet101.children())[:-3]
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=1024,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(resnet101, ccbm).to(device)

    if gamma is not None:
        saved_model_dir_path = f"saved_models/resnet101/model_{dataset_params.DATASET}_concept_loss_{comments}{gamma}_{model_name}_best.pth"
    else:
        saved_model_dir_path = f"saved_models/resnet101/model_{dataset_params.DATASET}_baseline_{comments}{model_name}_best.pth"

    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def load_ccbm_densenet121(model_params,
                          dataset_params,
                          model_name,
                          device):
    """Loads the DenseNet-121 model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained ResNet101
    densenet121_weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    densenet121 = torchvision.models.densenet121(weights=densenet121_weights)
    densenet121 = nn.Sequential(
        *list(densenet121.features.children())[:-2],
        # *list(densenet121.features.transition3.children())[:-2]
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=model_params.INPUT_SHAPE,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(densenet121, ccbm).to(device)

    saved_model_dir_path = f"saved_models/model_{dataset_params.DATASET}_{model_name}_best.pth"
    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def load_ccbm_densenet161(model_params,
                          dataset_params,
                          model_name,
                          device):
    """Loads the DenseNet-161 model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained ResNet101
    densenet161_weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
    densenet161 = torchvision.models.densenet161(weights=densenet161_weights)
    densenet161 = nn.Sequential(
        *list(densenet161.features.children())[:-3],
        *list(densenet161.features.transition3.children())[:-1],
        nn.AvgPool2d(kernel_size=(1, 1))
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=model_params.INPUT_SHAPE,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(densenet161, ccbm).to(device)

    saved_model_dir_path = f"saved_models/densenet161/model_{dataset_params.DATASET}_{model_name}_best.pth"
    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def load_ccbm_seresnext26d_32x4d(model_params,
                                 dataset_params,
                                 model_name,
                                 gamma,
                                 comments,
                                 device):
    """Loads the seresnext26d_32x4d model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained ResNet101
    seresnext26d_32x4d = timm.create_model('seresnext26d_32x4d', pretrained=True)

    seresnext26d_32x4d = nn.Sequential(
        *list(seresnext26d_32x4d.children())[:-3],
        # *list(seresnext26d_32x4d.layer4[0].children())[:-12]
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=1024,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(seresnext26d_32x4d, ccbm).to(device)

    if gamma is not None:
        saved_model_dir_path = f"saved_models/seresnext/model_{dataset_params.DATASET}_concept_loss_{comments}{gamma}_{model_name}_best.pth"
    else:
        saved_model_dir_path = f"saved_models/seresnext/model_{dataset_params.DATASET}_baseline_{comments}{model_name}_best.pth"

    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def load_ccbm_densenet201(model_params,
                          dataset_params,
                          model_name,
                          gamma,
                          comments,
                          device):
    """Loads the DenseNet-201 model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained ResNet101
    densenet201_weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
    densenet201 = torchvision.models.densenet201(weights=densenet201_weights)

    densenet201 = nn.Sequential(
        *list(densenet201.features.children())[:-3],
        *list(densenet201.features.transition3.children())[:-1],
        nn.AvgPool2d(kernel_size=(1, 1))
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=896,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(densenet201, ccbm).to(device)

    if gamma is not None:
        saved_model_dir_path = f"saved_models/densenet201/model_{dataset_params.DATASET}_concept_loss_{comments}{gamma}_{model_name}_best.pth"
    else:
        saved_model_dir_path = f"saved_models/densenet201/model_{dataset_params.DATASET}_baseline_{comments}{model_name}_best.pth"

    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def load_ccbm_mobilenetv2(model_params,
                          dataset_params,
                          model_name,
                          device):
    """Loads the MobileNet-V2 model with trained weights.

    :param params: A python file containing the hyperparameters.
    :param model_name: The model name.
    :param device: The torch device: cuda or cpu.
    :return: The PyTorch model.
    """

    # Load model weights and pretrained MobileNetV2
    mobilenetv2_weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    mobilenetv2 = torchvision.models.mobilenet_v2(weights=mobilenetv2_weights)
    mobilenetv2 = nn.Sequential(
        *list(mobilenetv2.children())[:-1]
    )

    # Create the CCbm
    ccbm = CCBM(input_shape=1280,
                num_concepts=model_params.NUM_CONCEPTS,
                num_classes=model_params.NUM_CLASSES)

    model = nn.Sequential(mobilenetv2, ccbm).to(device)

    saved_model_dir_path = f"saved_models/mobilenetv2/model_{dataset_params.DATASET}_{model_name}_best.pth"
    checkpoint = torch.load(saved_model_dir_path)
    best_loss_epoch = checkpoint["epoch"]
    print(f"[INFO] Epoch of best val loss: {best_loss_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[INFO] Model state dict loaded successfully from: {saved_model_dir_path}")

    return model


def get_ccbm_model(model_name: str,
                   params,
                   class_ind_vec,
                   device: torch.device,
                   freeze_layers=True,
                   train_classifier=False) -> torch.nn.Module:
    """Creates the CCbm model with the specified backbone

    Args:
        model_name (str): The model name.
        params (any): A python file containing the hyperparameters.
        class_ind_vec (Numpy array): The class indicator vector.
        device (torch.device): The torch device: cuda or cpu.
        freeze_layers (bool): If 'True', only new variables are trainable, else all model parameters are trainable. Default: True.
        train_classifier (bool): If 'True' only FC layers are trainable. Default: False.

    Returns:
        torch.nn.Module: The PyTorch model.
    """

    if model_name == "vgg16":
        model = ccbm_vgg16(params=params,
                           class_ind_vec=class_ind_vec,
                           device=device,
                           freeze_layers=freeze_layers,
                           train_classifier=train_classifier)
    elif model_name == "resnet18":
        model = ccbm_resnet18(params=params,
                              class_ind_vec=class_ind_vec,
                              device=device,
                              freeze_layers=freeze_layers,
                              train_classifier=train_classifier)
    elif model_name == "resnet50":
        model = ccbm_resnet50(params=params,
                              class_ind_vec=class_ind_vec,
                              device=device,
                              freeze_layers=freeze_layers,
                              train_classifier=train_classifier)
    elif model_name == "resnet101":
        model = ccbm_resnet101(params=params,
                               class_ind_vec=class_ind_vec,
                               device=device,
                               freeze_layers=freeze_layers,
                               train_classifier=train_classifier)
    elif model_name == "mobilenetv2":
        model = ccbm_mobilenetv2(params=params,
                                 class_ind_vec=class_ind_vec,
                                 device=device,
                                 freeze_layers=freeze_layers,
                                 train_classifier=train_classifier)
    elif model_name == "densenet121":
        model = ccbm_densenet121(params=params,
                                 class_ind_vec=class_ind_vec,
                                 device=device,
                                 freeze_layers=freeze_layers,
                                 train_classifier=train_classifier)
    elif model_name == "densenet161":
        model = ccbm_densenet161(params=params,
                                 class_ind_vec=class_ind_vec,
                                 device=device,
                                 freeze_layers=freeze_layers,
                                 train_classifier=train_classifier)
    elif model_name == "densenet201":
        model = ccbm_densenet201(params=params,
                                 class_ind_vec=class_ind_vec,
                                 device=device,
                                 freeze_layers=freeze_layers,
                                 train_classifier=train_classifier)
    elif model_name == "seresnext":
        model = ccbm_seresnext26d_32x4d(params=params,
                                        class_ind_vec=class_ind_vec,
                                        device=device,
                                        freeze_layers=freeze_layers,
                                        train_classifier=train_classifier)
    else:
        raise Exception("The model you provided is not available.")

    return model


def load_ccbm_model(model: str,
                    model_name: str,
                    params,
                    dataset_params,
                    gamma,
                    comments,
                    device) -> torch.nn.Module:
    """Load the trained CCbm model with the specified backbone

    Args:
        model (str): The model name.
        model_name (str): The name of the trained model.
        model_params (any): A python file containing the hyperparameters.
        dataset_params (any): A python file containing the hyperparameters.
        device (torch.device): The torch device: cuda or cpu.

    Returns:
        torch.nn.Module: The PyTorch model.
    """

    if model == "vgg16":
        loaded_model = load_ccbm_vgg16(model_params=params,
                                       dataset_params=dataset_params,
                                       model_name=model_name,
                                       device=device)
    elif model == "resnet18" or model == "ccbm_resnet18":
        loaded_model = load_ccbm_resnet18(model_params=params,
                                          dataset_params=dataset_params,
                                          model_name=model_name,
                                          device=device)
    elif model == "resnet50" or model == "ccbm_resnet50":
        loaded_model = load_ccbm_resnet50(model_params=params,
                                          dataset_params=dataset_params,
                                          model_name=model_name,
                                          device=device)
    elif model == "resnet101" or model == "ccbm_resnet101":
        loaded_model = load_ccbm_resnet101(model_params=params,
                                           dataset_params=dataset_params,
                                           model_name=model_name,
                                           gamma=gamma,
                                           comments=comments,
                                           device=device)
    elif model == "mobilenetv2":
        loaded_model = load_ccbm_mobilenetv2(model_params=params,
                                             dataset_params=dataset_params,
                                             model_name=model_name,
                                             device=device)
    elif model == "densenet121":
        loaded_model = load_ccbm_densenet121(model_params=params,
                                             dataset_params=dataset_params,
                                             model_name=model_name,
                                             device=device)
    elif model == "densenet161":
        loaded_model = load_ccbm_densenet161(model_params=params,
                                             dataset_params=dataset_params,
                                             model_name=model_name,
                                             device=device)
    elif model == "densenet201":
        loaded_model = load_ccbm_densenet201(model_params=params,
                                             dataset_params=dataset_params,
                                             model_name=model_name,
                                             gamma=gamma,
                                             comments=comments,
                                             device=device)
    elif model == "seresnext":
        loaded_model = load_ccbm_seresnext26d_32x4d(model_params=params,
                                                    dataset_params=dataset_params,
                                                    model_name=model_name,
                                                    gamma=gamma,
                                                    comments=comments,
                                                    device=device)
    else:
        raise Exception("The model you provided is not available.")

    return loaded_model
