"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import CCBMDataset

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
        params,
        train_transform: transforms.Compose,
        val_transform: transforms.Compose
):
    """Creates training and validation DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      params: a file containing model parameters.
      train_transform: torchvision transforms to perform on training data.
      val_transform: torchvision transforms to perform on validation data.

    Returns:
      A tuple of (train_dataloader, validation_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        train_dataloader, validation_dataloader, class_names = \
          = create_dataloaders(params=ph2_params,
                               train_transform=some_transform,
                               val_transform=some_transform,
                               )
    """
    # Use ImageFolder to create dataset(s)
    train_data = CCBMDataset(annotations_file=params.TRAIN_FILENAME,
                             dir_path=params.IMAGES_DIR,
                             masks_dir=params.MASKS_DIR,
                             extension=params.FILE_EXTENSION,
                             transform=train_transform)

    validation_data = CCBMDataset(annotations_file=params.VALIDATION_FILENAME,
                                  dir_path=params.IMAGES_DIR,
                                  masks_dir=params.MASKS_DIR,
                                  extension=params.FILE_EXTENSION,
                                  transform=val_transform)

    # Get class names
    class_names = ["Nevus", "Melanoma"]

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=params.BATCH_SIZE,
        shuffle=True,
        num_workers=params.NUM_WORKERS,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_data,
        batch_size=params.BATCH_SIZE,
        shuffle=False,
        num_workers=params.NUM_WORKERS,
        pin_memory=True,
    )

    return train_dataloader, validation_dataloader, class_names


def create_dataloader_for_evaluation(params,
                                     transform: transforms.Compose
                                     ):
    """Creates testing DataLoaders.

    Takes in a testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
      params: a file containing model parameters.
      transform: torchvision transforms to perform on test data.


    Returns:
      A tuple of (test_dataloader, class_names).
      Where class_names is a list of the target classes.
      Example usage:
        test_dataloader, class_names = \
          = create_dataloaders(params=ph2_params,
                               transform=some_transform
                               )
    """
    test_data = CCBMDataset(annotations_file=params.TEST_FILENAME,
                            dir_path=params.IMAGES_DIR,
                            masks_dir=params.MASKS_DIR,
                            extension=params.FILE_EXTENSION,
                            transform=transform)

    # Get class names
    class_names = ["Nevus", "Melanoma"]

    test_dataloader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0, #params.NUM_WORKERS,
        pin_memory=True,
    )

    return test_dataloader, class_names
