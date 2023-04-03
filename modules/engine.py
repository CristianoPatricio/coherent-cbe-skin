"""
Contains functions for training and testing a PyTorch model.
"""
import torch
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, auc, balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import model_params

from modules.utils import save_model, preprocess_masks, contribution_to_classification_decision, \
    plot_image_and_concepts, calculate_pixels_distribution, plot_hist_prob_density, save_vmix_and_vmax_values, plot_roc_curve


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion_s_loss: torch.nn.Module,
               criterion_c_loss: torch.nn.Module,
               criterion_u_loss: torch.nn.Module,
               criterion_concept_loss: torch.nn.Module,
               criterion_classification_loss: torch.nn.Module,
               optimizer_s_loss: torch.optim.Optimizer,
               optimizer_classification_loss: torch.optim.Optimizer,
               lambda_value: float,
               gamma_value: float,
               device: torch.device) -> Tuple[float, float, float, float, float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    criterion_s_loss: A PyTorch loss function to minimize.
    criterion_c_loss: A PyTorch loss function to minimize.
    criterion_u_loss: A PyTorch loss function to minimize.
    criterion_concept_loss: A PyTorch loss function to minimize.
    criterion_classification_loss: A PyTorch loss function to minimize.
    optimizer_s_loss: A PyTorch optimizer to help minimize the loss function.
    optimizer_classification_loss: A PyTorch optimizer to help minimize the loss function.
    lambda_value: A hyperparameter to control de influence of the interpretation loss.
    gamma_value: A hyperparameter to control de influence of the concept loss.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    c_loss_for_epoch = 0
    u_loss_for_epoch = 0
    s_loss_for_epoch = 0
    concept_loss_for_epoch = 0

    # Loop through data loader data batches
    for batch, data in enumerate(dataloader):
        # Send data to target device
        X, y, ind_vec, masks, _ = data['image'].to(device), data['label'].to(device), data['ind_vec'].to(device), data[
            'mask'].to(device), data['img_path']

        # 1. Forward pass
        y_pred, output_gap, output_concept_layer = model(X)

        # 2. Calculate and accumulate loss
        s_loss, positive = criterion_s_loss(output_concept_layer, ind_vec)
        c_loss = criterion_c_loss(ind_vec, positive)
        u_loss = criterion_u_loss(output_gap, ind_vec.type(torch.FloatTensor).to(device))
        classification_loss = criterion_classification_loss((y_pred + torch.tensor(1e-5)), y)
        interpretation_loss = s_loss + c_loss
        interpretation_loss = torch.mean(interpretation_loss, dim=1, keepdim=True)

        if criterion_concept_loss is not None:
            preprocessed_masks = preprocess_masks(masks, output_concept_layer.shape[1], output_concept_layer.shape[2], ind_vec)
            concept_loss = criterion_concept_loss(output_concept_layer, preprocessed_masks)
            loss = torch.mean(torch.add(torch.add(classification_loss, lambda_value * (u_loss + interpretation_loss)), gamma_value * concept_loss))
        else:
            loss = torch.mean(torch.add(classification_loss, lambda_value * (u_loss + interpretation_loss)))

        s_loss_for_epoch += torch.mean(s_loss)
        c_loss_for_epoch += torch.mean(c_loss)
        u_loss_for_epoch += torch.mean(u_loss)

        if criterion_concept_loss is not None:
            concept_loss_for_epoch += concept_loss.item()

        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer_s_loss.zero_grad()
        optimizer_classification_loss.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer_s_loss.step()
        optimizer_classification_loss.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    s_loss_for_epoch = s_loss_for_epoch / len(dataloader)
    c_loss_for_epoch = c_loss_for_epoch / len(dataloader)
    u_loss_for_epoch = u_loss_for_epoch / len(dataloader)

    if criterion_concept_loss is not None:
        concept_loss_for_epoch = concept_loss_for_epoch / len(dataloader)
    else:
        concept_loss_for_epoch = -1

    return train_loss, s_loss_for_epoch, c_loss_for_epoch, u_loss_for_epoch, concept_loss_for_epoch, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (val_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, data in enumerate(dataloader):
            # Send data to target device
            X, y, _, _, _ = data['image'].to(device), data['label'].to(device), data['ind_vec'].to(device), data[
                'mask'].to(device), data['img_path']

            # 1. Forward pass
            test_pred_logits, _, _ = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            val_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer_s_loss: torch.optim.Optimizer,
          optimizer_classification_loss: torch.optim.Optimizer,
          criterion_s_loss: torch.nn.Module,
          criterion_c_loss: torch.nn.Module,
          criterion_u_loss: torch.nn.Module,
          criterion_concept_loss: torch.nn.Module,
          criterion_classification_loss: torch.nn.Module,
          lambda_value: float,
          gamma_value: float,
          epochs: int,
          last_epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter,
          params,
          comments,
          model_name) -> Tuple[Dict[str, List], torch.nn.Module]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    val_dataloader: A DataLoader instance for the model to be tested on.
    criterion_s_loss: A PyTorch loss function to minimize.
    criterion_c_loss: A PyTorch loss function to minimize.
    criterion_u_loss: A PyTorch loss function to minimize.
    criterion_concept_loss: A PyTorch loss function to minimize.
    criterion_classification_loss: A PyTorch loss function to minimize.
    optimizer_s_loss: A PyTorch optimizer to help minimize the loss function.
    optimizer_classification_loss: A PyTorch optimizer to help minimize the loss function.
    lambda_value: A hyperparameter to control de influence of the interpretation loss.
    gamma_value: A hyperparameter to control de influence of the concept loss.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    writer: A SummaryWriter() instance to log model results to.
    params: A python file containing model info and hyperparameters
    model_name: The model name.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              val_loss: [...],
              val_acc: [...]}
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              val_loss: [1.2641, 1.5706],
              val_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }

    # Make sure model on target device
    model.to(device)

    # Set up a scheduler to reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer_classification_loss) #, patience=4, factor=0.1)

    best_val_loss = np.inf
    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, s_loss, c_loss, u_loss, concept_loss, train_acc = train_step(model=model,
                                                                                 dataloader=train_dataloader,
                                                                                 criterion_s_loss=criterion_s_loss,
                                                                                 criterion_c_loss=criterion_c_loss,
                                                                                 criterion_u_loss=criterion_u_loss,
                                                                                 criterion_concept_loss=criterion_concept_loss,
                                                                                 criterion_classification_loss=criterion_classification_loss,
                                                                                 optimizer_s_loss=optimizer_s_loss,
                                                                                 optimizer_classification_loss=optimizer_classification_loss,
                                                                                 lambda_value=lambda_value,
                                                                                 gamma_value=gamma_value,
                                                                                 device=device)

        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=criterion_classification_loss,
                                     device=device)

        # Scheduler step
        scheduler.step(val_loss)

        print(f"\nLearning rate: {optimizer_classification_loss.param_groups[0]['lr']}")

        # Save the best model to a file
        if val_loss < best_val_loss:
            print(f"\nValidation loss improved from {best_val_loss} to {val_loss}")
            best_val_loss = val_loss

            if gamma_value is not None:
                save_filepath = f"{params.MODEL_TO_SAVE_NAME}_concept_loss_{comments}{gamma_value}_{model_name}_best.pth"
            else:
                save_filepath = f"{params.MODEL_TO_SAVE_NAME}_baseline_{comments}{model_name}_best.pth"
            save_model(model=model,
                       epoch=epoch + last_epochs,
                       optimizer=optimizer_classification_loss,
                       valid_epoch_loss=val_loss,
                       target_dir=f"ablation_models/{model_name}",
                       model_name=save_filepath)

        # Print out what's happening
        print(
            f"\nEpoch: {epoch + 1 + last_epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"semantic_loss: {s_loss:.4f} | "
            f"count_loss: {c_loss:.4f} | "
            f"uniqueness_loss: {u_loss:.4f} | "
            f"concept_loss: {concept_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "val_loss": val_loss},
                               global_step=epoch + last_epochs)
            writer.add_scalars(main_tag="Accuracy",
                               tag_scalar_dict={"train_acc": train_acc,
                                                "val_acc": val_acc},
                               global_step=epoch + last_epochs)

            # Close the writer
            writer.close()
        else:
            pass

    # Save the last model
    if model_params.SAVE_LAST:
        if gamma_value is not None:
            save_filepath = f"{params.MODEL_TO_SAVE_NAME}_concept_loss_{comments}{gamma_value}_{model_name}_last.pth"
        else:
            save_filepath = f"{params.MODEL_TO_SAVE_NAME}_baseline_{comments}{model_name}_last.pth"
        save_model(model=model,
                   epoch=epoch + last_epochs,
                   optimizer=optimizer_classification_loss,
                   valid_epoch_loss=val_loss,
                   target_dir=f"ablation_models/{model_name}",
                   model_name=save_filepath)

    # Return the filled results at the end of the epochs
    return results, model


def evaluate(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device,
             params,
             model_name,
             baseline: False,
             plot_results: False):
    """Evaluates a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    params: The dataset parameters.
    plot_results: If True, the concept feat maps are plotted jointly with the image and the predictions.

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (val_loss, test_accuracy, class_report, conf_matrix). For example:

    (0.0223, 0.8985, class_report, conf_matrix, gt_concepts, predicted_concepts)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0
    y_true, y_pred = [], []
    gt_concepts, predicted_concepts = [], []
    activated_filters = []
    deactivated_filters = []
    filters = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, data in enumerate(dataloader):
            # Send data to target device
            X, y, ind_vec, _, img_path = data['image'].to(device), data['label'].to(device), data['ind_vec'].to(device), data[
                'mask'].to(device), data['img_path']
            y_true.append(y.item())
            gt_concepts.append(ind_vec.squeeze().cpu().numpy())

            # 1. Forward pass
            test_pred_logits, output_gap, output_concept_layer = model(X)

            predicted_concepts.append(torch.where(torch.tanh(output_gap) > 0.7, 1, 0).squeeze().cpu().numpy())

            weights_fc = []
            for name, param in model.named_parameters():
                if name == "1.classifier.0.weight":
                    weights_fc = param

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            val_loss += loss.item()

            # 3. Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            y_pred.append(test_pred_labels.item())
            val_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

            # 4. Calculate contribution to classification decision
            concept_contributions = contribution_to_classification_decision(output_gap=output_gap,
                                                                            weights_fc=weights_fc[test_pred_labels])



            # 5. Calculate filter distribution if CALCULATE_FILTER_DISTRIBUTION is True
            if model_params.CALCULATE_FILTER_DISTRIBUTION:
                act, deac = calculate_pixels_distribution(ind_vec,
                                                          output_concept_layer)
                activated_filters.append(act)
                deactivated_filters.append(deac)

            # 6. Plot feature maps of the concept layer
            if plot_results and not model_params.CALCULATE_FILTER_DISTRIBUTION:
                plot_image_and_concepts(image=X,
                                        img_path=img_path,
                                        test_pred_labels=test_pred_labels,
                                        true_class=y,
                                        output_gap=output_gap,
                                        concept_contributions=concept_contributions,
                                        weights_fc=weights_fc[test_pred_labels],
                                        ind_vec=ind_vec,
                                        output_concept_layer=output_concept_layer,
                                        model_name=model_name,
                                        params=params,
                                        baseline=baseline)

    if model_params.CALCULATE_FILTER_DISTRIBUTION:
        save_vmix_and_vmax_values(activated_filters,
                                  deactivated_filters,
                                  model_name,
                                  params.DATASET)

    #print(f"Activated (mean): {np.nanmean(np.asarray(activated_filters), axis=(0,1,3,4))}")
    #print(f"Activated (std): {np.nanmean(np.asarray(activated_filters), axis=(0,1,3,4))}")

    #print(f"Deactivated (mean): {np.nanstd(np.asarray(deactivated_filters), axis=(0, 1, 3, 4))}")
    #print(f"Deactivated (std): {np.nanstd(np.asarray(deactivated_filters), axis=(0, 1, 3, 4))}")

    #print("Max per image:")
    #print(np.max(np.asarray(filters), axis=(1,3,4)))

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)

    # Calculate the classification report
    class_report = classification_report(y_true, y_pred, target_names=["Nevus", "Melanoma"])

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    # Calculate AUC score
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # BACC
    bacc = balanced_accuracy_score(y_true, y_pred)

    # Sensitivity
    SE = TP / (TP + FN)

    # Specificity
    SP = TN / (TN + FP)

    #plot_roc_curve(y_true, y_pred)

    return val_loss, val_acc, class_report, conf_matrix, gt_concepts, predicted_concepts, roc_auc, bacc, SE, SP
