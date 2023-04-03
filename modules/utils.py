"""
Contains various utility functions for PyTorch model training and saving.
"""
import os.path
import random

import numpy as np
import torch
import torchvision
from torchvision import transforms
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, roc_curve, auc
import model_params


def save_model(model: torch.nn.Module,
               epoch: int,
               optimizer: torch.optim,
               valid_epoch_loss: float,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    epoch: Current epoch.
    optimizer: A PyTorch Optimizer.
    valid_epoch_loss: The current validation loss value.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               epoch=epoch,
               optimizer=optimizer,
               valid_epoch_loss= valid_epoch_loss,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj={
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': valid_epoch_loss
    }, f=model_save_path)


def set_seeds(seed: int = 42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set seed for general python operations
    random.seed(seed)
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    # Set the seed for Numpy operations
    np.random.seed(seed)
    # To use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    # TO use deterministic benchmark
    torch.backends.cudnn.benchmark = False


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d")  # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format).
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = (end - start) / 60
    print(f"[INFO] Train time on {device}: {total_time:.3f} minutes")
    return total_time


def write_to_txt(save_dir,
                 params,
                 accuracy,
                 class_report,
                 conf_matrix,
                 auc,
                 bacc,
                 sensitivity,
                 specificity,
                 gt_concepts,
                 predicted_concepts,
                 gamma,
                 comments,
                 model_name):
    """Writes a report to a txt file containing the result of the evaluation metrics

    Args:
        save_dir (str): the path directory to save the txt file.
        params: the file containing the parameters.
        accuracy (float): the accuracy value.
        class_report: the classification report object.
        conf_matrix: the confusion matrix object.
        gt_concepts: The ground-truth concepts.
        predicted_concepts: The predicted concepts.
        model_name: the name of the model

    Returns:
        str: the save path.
    """
    from datetime import datetime
    import numpy as np
    import os

    # Create save_dir if it not exists
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print(f"[INFO] {save_dir} created successfully!")

    if model_params.BASELINE:
        file = open(f"{save_dir}/results_{params.DATASET}_baseline_{comments}{model_name}.txt", "w")
    else:
        file = open(f"{save_dir}/results_{params.DATASET}_concept_loss_{comments}{gamma}_{model_name}.txt", "w")

    file.write("-" * 56 + "\n")
    x = datetime.now()
    file.write(" " * 18 + f"{x.strftime('%Y-%m-%d %H:%M:%S')}" + " " * 18 + "\n")
    file.write("-" * 56 + "\n\n")

    file.write("///////////   Dataset & Model Information   //////////// \n\n")
    if model_params.BASELINE:
        file.write(f"Dataset: {params.DATASET}_baseline_{comments} \n")
    else:
        file.write(f"Dataset: {params.DATASET}_concept_loss_{comments}{gamma} \n")

    file.write(f"Image Size: {params.IMG_SIZE} \n")
    file.write(f"Image Type: {params.IMG_TYPE} \n")
    file.write(f"Model: {model_name} \n")
    file.write(f"Learning Rate: {params.LEARNING_RATE} \n")
    file.write(f"No. Epochs: {params.EPOCHS} \n")
    file.write(f"Batch-Size: {params.BATCH_SIZE} \n")

    file.write(f"///////////  Evaluation Report - Concepts  //////////// \n\n")
    file.write(
        f"Exact Match Ratio: {accuracy_score(gt_concepts, predicted_concepts, normalize=True, sample_weight=None):.4f} \n")
    file.write(f"Hamming loss: {hamming_loss(gt_concepts, predicted_concepts):.4f} \n")
    file.write(
        f"Recall: {precision_score(y_true=gt_concepts, y_pred=predicted_concepts, average='samples'):.4f} \n")
    file.write(
        f"Precision: {recall_score(y_true=gt_concepts, y_pred=predicted_concepts, average='samples'):.4f} \n")
    file.write(
        f"F1 Measure: {f1_score(y_true=gt_concepts, y_pred=predicted_concepts, average='samples'):.4f} \n\n")

    f1s = []
    for i in range(8):
        f1s.append(f1_score(gt_concepts[:, i], predicted_concepts[:, i], average='weighted'))

    file.write(
        f"F1 Measure per Concept: {np.around(f1s, decimals=2)} \n")
    file.write(f"L2 distance: {np.linalg.norm(gt_concepts-predicted_concepts)} \n\n")

    file.write("/////////// Classification Report - Classes //////////// \n\n")
    file.write(f'Accuracy: {accuracy * 100:.4f}% \n\n')
    file.writelines(class_report)
    file.write("\n")
    file.write("Confusion Matrix:\n")
    file.writelines(np.array2string(conf_matrix))
    file.write("\n")
    file.write(f'AUC: {auc}\n')
    file.write(f'Sensitivity: {sensitivity}\n')
    file.write(f'Specificity: {specificity}\n')
    file.write(f'BACC: {bacc}\n')

    file.close()

    msg = f"[INFO] File saved at {file.name}"

    return print(msg)


def view_examples_dataloader(dataloader, class_names):
    """Displays examples of images contained in the dataloader

    :param dataloader: A PyTorch Dataloader object.
    :param class_names: The class names regarding the images.
    :return: None.
    """

    data = next(iter(dataloader))

    images = data["image"]
    labels = data["label"]
    ind_vectors = data["ind_vec"]

    fig = plt.figure(figsize=(9, 9))

    nrows, ncols = 4, 4

    for i in range(len(labels)):
        fig.add_subplot(nrows, ncols, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis(False)
        plt.title(class_names[labels[i]] + "\n" + str(ind_vectors[i].detach().cpu().numpy()))

    plt.show()


def preprocess_masks(masks, num_concepts, shape_concept_layer, ind_vec):
    """Applies a transformation on the masks to match the dimensions of the concept layer

    :param masks: The segmentation masks.
    :param num_concepts: The number of concepts.
    :param shape_concept_layer: The dimension of the concept layer (HEIGHT, WIDTH).
    :return: The resized masks.
    """

    # Define the transformation
    transform = transforms.Resize(size=(shape_concept_layer, shape_concept_layer),
                                  interpolation=InterpolationMode.NEAREST)

    # Apply the transform to masks
    resized_mask = transform(masks)

    # Reshape the resized masks to match the dimension of the output of the concept layer
    resized_mask = torch.unsqueeze(resized_mask, 1)
    resized_mask = torch.tile(resized_mask, (1, num_concepts, 1, 1))

    # Zeroing mask if ind vec is 0
    for b in range(resized_mask.shape[0]):
       for idx, i in enumerate(ind_vec[b]):
           if i == 0:
               resized_mask[b, idx, :, :] *= 0

    return resized_mask


def contribution_to_classification_decision(output_gap, weights_fc):
    """Calculates the contribution of concept k to the decision c

        Args:
            output_gap (Tensor): Output of the concept layer after GAP operation.
            weights_fc (Tensor): The weights of the fully-connected layers.

        Returns:
            Numpy array: the concept contributions to the decision.
    """

    # Calculate the contribution as the multiplication of output_gap and weights_fc
    contrib = torch.relu(output_gap) * weights_fc


    # Scale values to enforce that concepts that are not present has negative values
    scaled_contrib = torch.where(contrib < 0.05, -99, contrib)

    # Apply softmax function to obtain the percentage contributions
    contrib_softmax = torch.softmax(scaled_contrib, dim=1).cpu().numpy().squeeze()

    return contrib_softmax


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def plot_image_and_concepts(image,
                            img_path,
                            test_pred_labels,
                            true_class,
                            output_gap,
                            concept_contributions,
                            weights_fc,
                            ind_vec,
                            output_concept_layer,
                            model_name,
                            params,
                            baseline):
    """Plots the image together with the feature maps of the concept layer.

        Args:
            image: The test image.
            img_path: The path of the image.
            test_pred_labels: The predicted class label.
            true_class: The true class label.
            concept_contributions: The concept contribution to the decision.
            ind_vec: The indicator vectors.
            output_concept_layer: The feature maps of the concept layer.
            params: Tehe dataset parameters.

    :return: None.
    """

    # Define a transform to the test image
    transform = A.Compose([
        A.PadIfNeeded(512, 512),
        A.CenterCrop(width=512, height=512),
        A.Resize(width=224, height=224),
        ToTensorV2(),
    ])

    # List of concepts
    concept_name = ["TPN", "APN", "ISTR", "RSTR", "RDG", "IDG", "BWV", "RS"]

    # List of classes
    classes = {"0": "Nevus",
               "1": "Melanoma"}

    fig = plt.figure(figsize=(20, 3))

    nrows = 1
    ncols = 9

    fig.add_subplot(nrows, ncols, 1)
    plt.title(f"Pred: {classes[str(test_pred_labels.item())]}\nTrue: {classes[str(true_class.item())]}")

    # Read image
    # img = image[0, :, :, :].permute(1, 2, 0).cpu().numpy()
    if params.FILE_EXTENSION == "png":
        img = cv2.imread(img_path[0] + ".png")
    elif params.FILE_EXTENSION == "jpg":
        img = cv2.imread(img_path[0] + ".jpg")
    else:
        img = cv2.imread(img_path[0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transformation
    transformed = transform(image=img)
    img = transformed["image"]

    plt.imshow(img.permute(1, 2, 0))
    if os.path.basename(img_path[0]) in ["Nal066.jpg"]:
        if baseline:
            plt.imsave(f"figures_baseline/{model_name}/{params.DATASET}/{os.path.basename(img_path[0])}_original.png", img.permute(1, 2, 0).numpy())
        else:
            plt.imsave(f"figures_closs/{model_name}/{params.DATASET}/{os.path.basename(img_path[0])}_original.png", img.permute(1, 2, 0).numpy())

    plt.axis(False)

    #mins = np.load(f"colormap_values/colormap_vmin_{model_name}_{params.DATASET}.npy")
    #print(f"[INFO] Vmin values loaded from: colormap_values/colormap_vmin_{model_name}_{params.DATASET}.npy")

    #maxs = np.load(f"colormap_values/colormap_vmax_{model_name}_{params.DATASET}.npy")
    #print(f"[INFO] Vmax values loaded from: colormap_values/colormap_vmax_{model_name}_{params.DATASET}.npy")

    #predicted_concepts = np.where(concept_contributions > 0.05, 1, 0)
    predicted_concepts = torch.where(torch.tanh(output_gap) > 0.7, 1, 0).squeeze().cpu().numpy()


    for i in range(8):
        fig.add_subplot(nrows, ncols, i + 2)
        contribution = concept_contributions[i] * 100
        weight = weights_fc[:, i][0].cpu().numpy()
        plt.title(f"{concept_name[i]}\n({ind_vec.cpu().numpy()[0, i]}) [{predicted_concepts[i]}] | {contribution:.2f} | {weight:.2f}")
        # np.save(f"filter_concept_{i}.npy", output_concept_layer[0, i, :, :].cpu().numpy())
        heatmap = cv2.resize(output_concept_layer[0, i, :, :].cpu().numpy(), (224, 224))

        plt.imsave("img.png", heatmap, cmap='jet') #, vmin=mins[i], vmax=maxs[i])
        im_heatmap = plt.imread("img.png")[:, :, :3]

        rgb_img = img.permute(1, 2, 0).cpu().numpy()

        # Extract meaningful part of the image #######################
        #if os.path.basename(img_path[0]) in ["IMD088", "IMD409", "IMD004"]:
        # if ind_vec.cpu().numpy()[0, i] == 1 and contribution > 0.05:
        #     new_mask = np.where(torch.tanh(torch.tensor(heatmap)).cpu().numpy() > 0.995, 1, 0)
        #
        #     new_img = np.zeros(shape=(224,224,3), dtype=np.uint8)
        #     new_img[:, :, 0] = np.where(new_mask > 0, rgb_img[:, :, 0], 0)
        #     new_img[:, :, 1] = np.where(new_mask > 0, rgb_img[:, :, 1], 0)
        #     new_img[:, :, 2] = np.where(new_mask > 0, rgb_img[:, :, 2], 0)
        #
        #     plt.imsave(
        #         f"concept_patches/{model_name}/{params.DATASET}/{concept_name[i]}/{os.path.basename(img_path[0])}_{concept_name[i]}.png",
        #         new_img)

            # if baseline:
            #     plt.imsave(f"figures_baseline/{model_name}/{params.DATASET}/patch_{i}_{os.path.basename(img_path[0])}_{concept_name[i]}.png", new_img)
            # else:
            #     plt.imsave(
            #         f"figures_closs/{model_name}/{params.DATASET}/patch_{i}_{os.path.basename(img_path[0])}_{concept_name[i]}.png",
            #         new_img)
        ##############################################################

        rgb_img = np.float32(rgb_img) / 255

        res = im_heatmap * 0.3 + rgb_img * 0.5

        plt.imshow(res)
        if os.path.basename(img_path[0]) in ["IMD395"]:
            if baseline:
               plt.imsave(f"figures_baseline/{model_name}/{params.DATASET}/{os.path.basename(img_path[0])}_{concept_name[i]}.png", res)
            else:
               plt.imsave(
                   f"figures_closs/{model_name}/{params.DATASET}/{os.path.basename(img_path[0])}_{concept_name[i]}.png",
                   res)
        plt.axis(False)

    if baseline:
        save_dir = f"figures_baseline/{model_name}/{params.DATASET}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print(f"[INFO] {save_dir} created successfully!")

        #plt.show()
        print(f"[INFO] Saving figure to {save_dir}...")
        plt.savefig(f"{save_dir}/{os.path.basename(img_path[0])}.png")
    else:
        save_dir = f"figures_closs/{model_name}/{params.DATASET}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print(f"[INFO] {save_dir} created successfully!")

        # plt.show()
        print(f"[INFO] Saving figure to {save_dir}...")
        plt.savefig(f"{save_dir}/{os.path.basename(img_path[0])}.png")
    plt.close(fig)


def calculate_pixels_distribution(ind_vec,
                                  output_concept_layer):

    temp_out_con_layer_act = output_concept_layer.cpu().numpy().copy()
    temp_out_con_layer_deac = output_concept_layer.cpu().numpy().copy()

    ind_vec = ind_vec.cpu().numpy().squeeze()

    for idx, i in enumerate(ind_vec):
        if i == 0:
            temp_out_con_layer_act[:, idx, :, :] = np.NAN
        else:
            temp_out_con_layer_deac[:, idx, :, :] = np.NAN

    return temp_out_con_layer_act, temp_out_con_layer_deac


def save_vmix_and_vmax_values(activated_filters, deactivated_filters, model_name, dataset):

    concept_name = ["TPN", "APN", "ISTR", "RSTR", "RDG", "IDG", "BWV", "RS"]
    vmin = []
    vmax = []

    for i in range(len(concept_name)):
        x1 = np.asarray(activated_filters)[:, :, i, :, :].flatten()
        x2 = np.asarray(deactivated_filters)[:, :, i, :, :].flatten()
        mu_activated = np.nanmean(np.asarray(activated_filters)[:, :, i, :, :].flatten())
        sigma_activated = np.nanstd(np.asarray(activated_filters)[:, :, i, :, :].flatten())
        mu_deactivated = np.nanmean(np.asarray(deactivated_filters)[:, :, i, :, :].flatten())

        if np.asarray(activated_filters)[:, :, i, :, :].flatten() is []:
            stop = 1

        sigma_deactivated = np.nanstd(np.asarray(deactivated_filters)[:, :, i, :, :].flatten())
        concept = concept_name[i]

        # Save values to a npy file
        vmin.append(mu_deactivated - 2 * sigma_deactivated)
        vmax.append(mu_activated + 2 * sigma_activated)

        if model_params.PLOT_HISTOGRAM_FILTERS:
            plot_hist_prob_density(mu_activated, sigma_activated, mu_deactivated, sigma_deactivated, x1, x2, concept)

    np.save(f"colormap_values/colormap_vmin_{model_name}_{dataset}.npy", vmin)
    print(f"[INFO] File saved at colormap_values/colormap_vmin_{model_name}_{dataset}.npy")

    np.save(f"colormap_values/colormap_vmax_{model_name}_{dataset}.npy", vmax)
    print(f"[INFO] File saved at colormap_values/colormap_vmin_{model_name}_{dataset}.npy")


def plot_hist_prob_density(mu1, sigma1, mu2, sigma2, x1, x2, concept):
    num_bins = 20

    fig, ax = plt.subplots()

    # the histogram of the data
    ax.hist(x1, num_bins, density=1)
    ax.hist(x2, num_bins, density=1)

    # add a 'best fit' line
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma1)) *
    #     np.exp(-0.5 * (1 / sigma1 * (bins - mu1)) ** 2))
    # y1 = ((1 / (np.sqrt(2 * np.pi) * sigma2)) *
    #     np.exp(-0.5 * (1 / sigma2 * (bins - mu2)) ** 2))
    # ax.plot(bins, y, '--')
    # ax.plot(bins, y1, '--')
    ax.set_xlabel('Filter Values')
    ax.set_ylabel('Probability density')
    mu1 = np.around(mu1, decimals=3)
    sigma1 = np.around(sigma1, decimals=3)
    mu2 = np.around(mu2, decimals=3)
    sigma2 = np.around(sigma2, decimals=3)
    ax.set_title(
        'Histogram of ' + str(concept) + ': (1) $\mu=$' + str(mu1) + ' $\sigma=$' + str(sigma1) + ' | (0) $\mu=$' + str(
            mu2) + ' $\sigma=$' + str(sigma2))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.xlim(xmin=-10, xmax=10)
    plt.show()

def plot_roc_curve(y_true, y_pred):

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    print(f"AUC: {auc(fpr, tpr)}")
    plt.plot(fpr, tpr)
    plt.show()
