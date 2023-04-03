import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import os
from torchvision.io import read_image
import model_params as params
from pathlib import Path
from modules.utils import set_seeds


class CCBMDataset(Dataset):
    def __init__(self, annotations_file, dir_path, masks_dir, extension, transform=None, target_transform=None):
        """Initializes a custom dataset given a CSV file.

        :param annotations_file: The CSV file containing images and labels.
        :param dir_path: The path directory of the images.
        :param masks_dir: The path directory of the segmentation masks.
        :param extension: The file extension of the images (jpg, png).
        :param transform: A PyTorch Transform to be applied to the images.
        :param target_transform: A PyTorch Transform to be applied to the labels
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.dir_path = dir_path
        self.masks_dir = masks_dir
        self.extension = extension
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.dir_path, self.img_labels.iloc[idx, 0])

        # Read image
        if self.extension == "":
            #image = read_image(img_path)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.img_labels.iloc[idx, 0].find("/") != -1:
                mask = cv2.imread(
                    os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0][4:-4] + ".png"), cv2.IMREAD_UNCHANGED
                )
            else:
                mask = cv2.imread(
                    os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0][:-4] + ".png"), cv2.IMREAD_UNCHANGED
                )
        else:
            #image = read_image(img_path + "." + self.extension)
            image = cv2.imread(img_path + "." + self.extension)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.img_labels.iloc[idx, 0].find("/") != -1:
                mask = cv2.imread(
                    os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0][4:] + ".png"), cv2.IMREAD_UNCHANGED
                )
            else:
                mask = cv2.imread(
                    os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0] + ".png"), cv2.IMREAD_UNCHANGED
                )


        # Get image mask
        #img_mask = read_image(os.path.join(self.masks_dir, self.img_labels.iloc[idx, 0] + ".png"))

        # Get respective label
        label = self.img_labels.iloc[idx, 1]

        # Get Indicator Vectors
        indicator_vectors_dict = np.load(params.INDICATOR_VECTORS, allow_pickle=True).item()
        indicator_vectors_dict = {k.replace('.jpg', '').replace('.JPG', ''): v for k, v in indicator_vectors_dict.items()}

        filename = Path(img_path).stem
        indicator_vector = indicator_vectors_dict[filename]

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        if self.target_transform:
            label = self.target_transform(label)

        return {
                'image': image,
                'label': label,
                'ind_vec': indicator_vector,
                'mask': mask,
                'img_path': img_path
        }
