r"""
Deep Neural Networks (DNN) helper module.
This module provides a simple interface for creating and training deep neural networks with PyTorch without reimplementing common functions.

Author: Gabriele Scorpaniti, 2025
"""

# Standard Libraries
import os
import random

# Data Science Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# PyTorch Libraries
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from typing import Type, Optional, List, Dict, Any

# Scikit-learn Library
from sklearn.model_selection import KFold

# Dataclass Library
from dataclasses import dataclass, field


class EarlyStopping:
    """
    Implements early stopping to terminate training when the validation loss does not improve for a specified number of epochs.
    
    Args:
    save_path : str
        Path to save the model checkpoint.
    patience : int
        Number of epochs with no improvement after which training will be stopped.
    min_delta : float
        Minimum change in the monitored quantity to qualify as an improvement.
    """
    def __init__(self, save_path, patience=5, min_delta=0.0):
        # Initialize the early stopping parameters
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.min_val_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss, model):
        # First Epoch
        if self.min_val_loss is None:     
            self.min_val_loss = validation_loss
            self.save_checkpoint(model)
        # Epoch with improvement
        elif (self.min_val_loss - validation_loss) > self.min_delta:
            self.min_val_loss = validation_loss
            self.save_checkpoint(model)
            self.counter = 0
        # No improvement, stop training if patience is reached
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)


@dataclass
class Experiment:
    """
    Class to manage the training of a deep neural network. Requires a model, loss function, and optimizer.
    
    Args:
    name : str
        Name of the experiment.
    checkpoints_folder : str
        Path to the folder where checkpoints will be saved.
    checkpoint_name : str
        Name of the checkpoint file.
    model : object
        The model class to be trained.
    loss_fn : object
        The loss function class to be used for training.
    optimizer : object
        The optimizer class to be used for training.
    val_mse : float, optional
        Initial validation mean squared error (MSE) value. Default is None.
    lr : float, optional
        Learning rate for the optimizer. Default is 1e-5.
    epochs : int, optional
        Number of epochs for training. Default is 600.
    metrics : list, optional
        List of metrics to be used for evaluation. Default is an empty list.
    use_early_stopping : bool
        Whether to use early stopping. Default is False.
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped. Default is 10.
    min_delta : float, optional
        Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
    early_stopping : object, optional
        Early stopping object. Default is None.
    epoch_count : list, optional
        List to store the epoch count. Default is an empty list.
    train_loss_values : list, optional
        List to store the training loss values. Default is an empty list.
    val_loss_values : list, optional
        List to store the validation loss values. Default is an empty list.
    color : str
        Color for plotting. Default is 'blue'.
    alpha : float
        Alpha value for plotting. Default is 0.5.
    plt_args_training : dict, optional
        Additional arguments for training plot. Default is an empty dictionary.
    plt_args_validation : dict, optional
        Additional arguments for validation plot. Default is an empty dictionary.
    """
    #General parameters
    name: str
    checkpoints_folder: str
    checkpoint_name: str
    model: nn.Module
    metrics: List[str]  # [accuracy, precision, recall, f1]
    n_classes: int

    #Model hyperparameters
    loss_fn: Type[nn.Module]
    optimizer: Type[optim.Optimizer]
    lr: float = 1e-5
    lr_scheduler: bool = False
    lr_gamma: float = 0.1
    lr_step: int = 5

    #Loss values
    train_loss_values: List[float] = field(default_factory=list)
    val_loss_values: List[float] = field(default_factory=list)
    epoch_count: List[int] = field(default_factory=list)
    
    # Early stopping
    use_early_stopping: bool = False
    patience: int = 5
    min_delta: float = 0.0
    
    #Epochs
    epochs: int = 50

    #Metrics objects
    train_metrics_objects: Dict[str, Any] = field(default_factory=dict)
    val_metrics_objects: Dict[str, Any] = field(default_factory=dict)

    #Metrics values
    val_accuracy_values: List[float] = field(default_factory=list)
    val_precision_values: List[float] = field(default_factory=list)
    val_f1_values: List[float] = field(default_factory=list)
    val_recall_values: List[float] = field(default_factory=list)
    
    #Plotting arguments
    color: str = "blue"
    alpha: float = 0.5
    plt_args_training: Dict[str, Any] = field(default_factory=dict)
    plt_args_validation: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Initializes the Experiment class by setting up the model, loss function, optimizer, and early stopping parameters.
        """

        # Set the device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model, loss function, and optimizer
        self.model = self.model.to(self.device)

        self.loss_fn = self.loss_fn()
        self.loss_fn = self.loss_fn.to(self.device)

        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

        # Initialize loss lists
        self.train_loss_values = []
        self.val_loss_values = []
        
        # Initialize the early stopping object if required
        if self.use_early_stopping:

            early_stopping_folder = os.path.join(self.checkpoints_folder, "early_stoppings", self.name)
            os.makedirs(early_stopping_folder, exist_ok=True)
            
            self.early_stopping = EarlyStopping(save_path=os.path.join(early_stopping_folder, self.checkpoint_name),
                                                patience=self.patience,
                                                min_delta=self.min_delta)
            
        # Initialize metrics
        if "accuracy" in self.metrics:
            
            self.accuracy_train = MulticlassAccuracy(num_classes=self.n_classes).to(self.device)
            self.accuracy_val = MulticlassAccuracy(num_classes=self.n_classes).to(self.device)
            self.train_metrics_objects["accuracy"] = self.accuracy_train
            self.val_metrics_objects["accuracy"] = self.accuracy_val

        if "precision" in self.metrics:

            self.precision_train = MulticlassPrecision(num_classes=self.n_classes).to(self.device)
            self.precision_val = MulticlassPrecision(num_classes=self.n_classes).to(self.device)
            self.train_metrics_objects["precision"] = self.precision_train
            self.val_metrics_objects["precision"] = self.precision_val

        if "recall" in self.metrics:

            self.recall_train = MulticlassRecall(num_classes=self.n_classes).to(self.device)
            self.recall_val = MulticlassRecall(num_classes=self.n_classes).to(self.device)
            self.train_metrics_objects["recall"] = self.recall_train
            self.val_metrics_objects["recall"] = self.recall_val

        if "f1" in self.metrics:

            self.f1_train = MulticlassF1Score(num_classes=self.n_classes).to(self.device)
            self.f1_val = MulticlassF1Score(num_classes=self.n_classes).to(self.device)
            self.train_metrics_objects["f1"] = self.f1_train
            self.val_metrics_objects["f1"] = self.f1_val
        
        # Checkpointing setup
        self.checkpoints_folder = os.path.join(self.checkpoints_folder, "checkpoints", self.name)
        os.makedirs(self.checkpoints_folder, exist_ok=True)
        self.checkpoint_save_path = os.path.join(self.checkpoints_folder, self.checkpoint_name)

    def save_checkpoint(self):
        """
        Save the model checkpoint.
        """
        torch.save(self.model.state_dict(), self.checkpoint_save_path)


class Helper:
    """
    Helper functions for CNN training and evaluation.
    """

    @staticmethod
    def plot_images(dataset, classes, iteration=0, num_row=3, num_col=5, validation=False, y_pred=None, show_only_wrong=False):
        """
        Displays a batch of images from the dataset, also showing the predicted label if provided.
        Args:
            dataset: PyTorch dataset.
            classes: List of class names.
            iteration: Iteration number for batch visualization.
            num_row: Number of rows.
            num_col: Number of columns.
            validation: If True, also shows the predicted label (default False).
            y_pred: List of predicted labels (optional, required if validation=True).
        """

        fig, axes = plt.subplots(num_row, num_col, figsize=(10*num_row, 2*num_col))

        for i in range(num_row * num_col):

            idx = iteration * num_row * num_col + i
            if idx >= len(dataset):
                break 

            true_label = int(dataset[idx][1])

            # Avoids ValueErrorer if y_pred is None
            if validation:
                if y_pred is None:
                    raise ValueError("y_pred must be provided for validation.")
                
                pred_label = int(y_pred[idx])
            
            # Skip in case of correct prediction if show_only_wrong is True
            if show_only_wrong:
                if pred_label == true_label:
                    continue

            # Get the current axis
            ax = axes[i // num_col, i % num_col]

            # Get image and true label
            image = Helper.back_to_image(dataset[idx][0])

            # Display the image
            ax.imshow(image)

            # Set the title to show the true label
            title = f'True: {classes[true_label]}'
            ax.axis('off')

            # If validation is True, show the predicted label
            if validation:
                if pred_label != true_label:
                    title += f' | Pred: {classes[pred_label]}'
                    ax.set_title(title, color='red')
                else:
                    title += f' | Pred: {classes[pred_label]}'
                    ax.set_title(title)
                    
            else:
                ax.set_title(title)

        plt.tight_layout()
        plt.show()
        iteration += 1

    @staticmethod
    def plot_class_distribution(dataset, type="training"):
        """
        Plots the class distribution of a dataset.
        Args:
            dataset: PyTorch dataset.
            type: Type of dataset (training, validation or test).
        """

        #Check if the dataset is a valid PyTorch dataset
        if not hasattr(dataset, 'targets'):
            raise ValueError("Il dataset non è un dataset PyTorch valido.")

        #Dataset classes count
        df = pd.DataFrame(dataset.targets, columns=['label'])
        df['label'] = df['label'].map(lambda x: dataset.classes[x])
        
        #Plotting
        df['label'].value_counts().plot(kind='bar', figsize=(12, 6))
        plt.title(f'Distribuzione delle classi nel {type} set')


        plt.xlabel('Classi')
        plt.ylabel('Numero di samples')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def plot_loss(exp: Experiment):
        """
        Plots the training and validation loss over epochs.
        
        Args:
            exp: Experiment object containing the training history.
        """

        # Check if the experiment has loss values
        if not exp.train_loss_values or not exp.val_loss_values:
            raise ValueError("Loss values are not available. Please train the model first.")
        
        # Check if the experiment has epoch count
        if not exp.epoch_count:
            raise ValueError("Epoch count is not available. Please train the model first.")
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(exp.epoch_count, exp.train_loss_values, label='Training Loss', color=exp.color, alpha=exp.alpha, **exp.plt_args_training)
        plt.plot(exp.epoch_count, exp.val_loss_values, label='Validation Loss', color=exp.color, alpha=exp.alpha, **exp.plt_args_validation)

        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.title(f'Loss over Epochs for {exp.name}')

        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, title=None, cmap="Blues"):
        """
        Plot the confusion matrix.
        """

        # Check if y_true and y_pred are of the same length
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        
        # Check if classes is a list
        if not isinstance(classes, list):
            raise ValueError("classes must be a list of class names.")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap=cmap,
                    xticklabels=classes, yticklabels=classes)
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        if title:
            plt.title(title)

        plt.show()

    @staticmethod
    def back_to_image(img):
        """
        Convert a tensor to an image.
        
        Args:
        tensor : torch.Tensor
            The input tensor to be converted to an image.
        
        Returns:
        numpy.ndarray
            The converted image as a NumPy array.
        """

        img = img / 2 + 0.5
        npimg = img.numpy()
        return np.transpose(npimg, (1, 2, 0))

    @staticmethod
    def set_seed(seed):
        """
        Set the random seed for reproducibility.
        
        Args:
        seed : int
            The random seed to be set.
        """

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def set_device():
        """
        Set the device to GPU if available, otherwise CPU.
        
        Returns:
        torch.device
            The device to be used for computations.
        """

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Trainer:
    """
    Manages training and evaluation of a deep neural network.
    """

    NA_MESSAGE = "NA"

    @staticmethod
    def _compute_classification_metrics(exp: Experiment, verbose=False):
        """
        Compute metrics. Internal method for fit and evaluate.

        Params:
            exp: Experiment: the experiment to get values from
        """

        # Store metrics values
        if "accuracy" in exp.metrics:
            accuracy = exp.val_metrics_objects["accuracy"].compute()
            exp.val_accuracy_values.append(accuracy)
        else:
            accuracy = Trainer.NA_MESSAGE
        
        if "precision" in exp.metrics:
            precision = exp.val_metrics_objects["precision"].compute()
            exp.val_precision_values.append(precision)
        else:
            precision = Trainer.NA_MESSAGE

        if "f1" in exp.metrics:
            f1 = exp.val_metrics_objects["f1"].compute()
            exp.val_f1_values.append(f1)
        else:
            f1 = Trainer.NA_MESSAGE

        if "recall" in exp.metrics:
            recall = exp.val_metrics_objects["recall"].compute()
            exp.val_recall_values.append(recall)
        else:
            recall = Trainer.NA_MESSAGE

        if verbose:
            print(f"Metrics. Accuracy: {accuracy}, Precision: {precision}, F1: {f1}, recall: {recall}.")

        return accuracy, precision, f1, recall


    @staticmethod
    def fit(exp: Experiment, train_dl, val_dl, verbose=True):
        """
        Train the model for a specified number of epochs, computing exp.metrics.
        """

        # Reset tracking lists
        exp.epoch_count = []
        exp.train_loss_values = []
        exp.val_loss_values = []

        # Reset early stopping
        if exp.use_early_stopping:
            exp.early_stopping.counter = 0
            exp.early_stopping.min_val_loss = None
            exp.early_stopping.early_stop = False

        if exp.lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(exp.optimizer, step_size=exp.lr_step, gamma=exp.lr_gamma)

        if verbose:
            print(f"Training {exp.name}. Epochs: {exp.epochs} | Learning Rate: {exp.lr} | Batch Size: {train_dl.batch_size} | Loss Function: {exp.loss_fn.__class__.__name__} | Optimizer: {exp.optimizer.__class__.__name__}")

        # Reset Loss values before training
        exp.train_loss_values = []
        exp.val_loss_values = []

        # Reset Metrics values 
        exp.val_accuracy_values = []
        exp.val_precision_values = []


        for epoch in range(exp.epochs):

            # Reset metrics for the next epoch
            for i, metric in enumerate(exp.train_metrics_objects):
                exp.train_metrics_objects[metric].reset()
            for i, metric in enumerate(exp.val_metrics_objects):
                exp.val_metrics_objects[metric].reset()

            exp.model.train()
            loss_epoch = 0

            for _, data in enumerate(train_dl, 0):

                X = data[0].to(exp.device)
                y = data[1].to(exp.device)

                y_pred = exp.model(X)
                loss = exp.loss_fn(y_pred, y)

                # Compute metrics
                for i, metric in enumerate(exp.train_metrics_objects):
                    exp.train_metrics_objects[metric].update(y_pred, y)

                loss_epoch += loss.item()

                # Backpropagation
                exp.optimizer.zero_grad()
                loss.backward()
                exp.optimizer.step() #run before lr_scheduler.step()

            loss_val = 0
            exp.model.eval()

            for _, data in enumerate(val_dl, 0):

                X = data[0].to(exp.device)
                y = data[1].to(exp.device)

                with torch.no_grad():

                    # Compute loss
                    y_pred = exp.model(X)

                    loss = exp.loss_fn(y_pred, y)
                    loss_val += loss.item()

                    # Compute metrics
                    for i, metric in enumerate(exp.val_metrics_objects):
                        exp.val_metrics_objects[metric].update(y_pred, y)

            # Store loss values
            exp.train_loss_values.append(loss_epoch/len(train_dl))
            exp.val_loss_values.append(loss_val/len(val_dl))
            exp.epoch_count.append(epoch)

            # Store metrics values
            epoch_val_accuracy, epoch_val_precision, epoch_val_f1, epoch_val_recall = Trainer._compute_classification_metrics(exp)

            # Print metrics
            if verbose:
                print(f"""Epoch: {epoch} |  Train Loss: {exp.train_loss_values[-1]:.4f} | 
                      Val Loss: {exp.val_loss_values[-1]:.4f} | 
                      Val Accuracy: {epoch_val_accuracy} | 
                      Val Precision: {epoch_val_precision} | 
                      Val F1: {epoch_val_f1} | 
                      Val Recall: {epoch_val_recall}"""
                      )

            # LR Step, if applicable
            if exp.lr_scheduler:
                lr_scheduler.step()

            # Early Stopping if set
            if exp.use_early_stopping:
                exp.early_stopping(loss_val/len(val_dl), exp.model)
                if exp.early_stopping.early_stop:
                    print("Early stopping all'epoca:", epoch)
                    exp.model.load_state_dict(torch.load(exp.checkpoint_save_path))
                    break

            # Save the model checkpoint every 5 epochs
            if epoch % 5 == 0:
                exp.save_checkpoint()

    @staticmethod
    def evaluate(exp: Experiment, testloader):
        """
        Evaluate the model on the test set.
        
        Args:
            exp: Experiment object containing the model and evaluation parameters.
            testloader: DataLoader for the test set.

        Returns:
            loss_test: float
                The average loss on the test set.
            accuracy: float
                The accuracy of the model on the test set.
            precision: float
                The precision of the model on the test set.
            f1: float
                The F1 Score of the model on the test set.
            Recall: float
                The recall score of the model on the test set.
        """

        exp.model.eval()
        loss_test = 0

        for _, data in enumerate(testloader, 0):

            X = data[0].to(exp.device)
            y = data[1].to(exp.device)

            with torch.no_grad():

                y_pred = exp.model(X)
                loss = exp.loss_fn(y_pred, y)
                loss_test += loss.item()

                # Compute metrics
                for i, metric in enumerate(exp.val_metrics_objects):
                    exp.val_metrics_objects[metric].update(y_pred, y)

        # Store metrics values
        epoch_accuracy, epoch_precision, epoch_f1, epoch_recall = Trainer._compute_classification_metrics(exp)

        # Reset metrics
        for i, metric in enumerate(exp.val_metrics_objects):
            exp.val_metrics_objects[metric].reset()

        return loss_test/len(testloader), epoch_accuracy, epoch_precision, epoch_f1, epoch_recall
    
    @staticmethod
    def predict(exp: Experiment, testloader):
        """
        Predict the class labels for a given dataset.
        
        Args:
            exp: Experiment object containing the model and evaluation parameters.
            testloader: DataLoader for the features to execute predict on.

        Returns:
            y_pred: list
                The predicted class labels (as 1D array).
        """

        exp.model.eval()
        y_pred = []

        for _, data in enumerate(testloader, 0):

            X = data[0].to(exp.device)

            with torch.no_grad():

                y_pred_batch = exp.model(X)
                y_pred_batch = torch.argmax(y_pred_batch, dim=1).cpu().numpy()
                y_pred.append(y_pred_batch)

        return np.concatenate(y_pred).tolist()


class CrossValidation():
    """
    Manages cross-validation for training and evaluating a model.
    """
    def __init__(
                self, 
                experiments: List[Experiment], 
                train_ds: data_utils.Dataset,
                val_ds: Optional[data_utils.Dataset], 
                n_splits: int,
                batch_size: int = 128, 
                shuffle: bool = True, 
                seed: Optional[int] = None,
                verbose: bool = True
            ):
        """
        Initialize the cross-validation object.
        
        Args:
            n_splits: Number of splits for cross-validation.
            shuffle: Whether to shuffle the data before splitting.
            random_state: Random seed for reproducibility.
        """

        # Instance variables
        self.experiments = experiments
        self.train_ds = train_ds
        self.val_ds = val_ds

        # Set hyperparamter
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose

    def run(self):
        """
        Cross Validate the model using K-Fold cross-validation.
        Handles fold splitting, training, and evaluation.
        """
        
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        experiment_results = {}

        for exp in self.experiments:

            print(f"Cross Validation for {exp.name} with {self.n_splits} folds")

            # Initialize metrics lists
            cross_val_loss = []
            cross_val_accuracy = []
            cross_val_precision = []
            cross_val_f1 = []
            cross_val_recall = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(self.train_ds)):

                # Reset the model to initial state
                for i, layer in enumerate(exp.model.children()):
                    if hasattr(layer, 'reset_parameters') and i >= 173: # Skip the ResNet backbone
                        layer.reset_parameters()

                exp.optimizer = exp.optimizer.__class__(exp.model.parameters(), lr=exp.lr)

                # Define samplers for the current fold
                train_sampler = data_utils.SubsetRandomSampler(train_idx)
                val_sampler = data_utils.SubsetRandomSampler(val_idx)

                # Creating DataLoaders for this fold
                train_dl_split = data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, sampler=train_sampler)
                val_dl_split = data_utils.DataLoader(self.train_ds, batch_size=self.batch_size, sampler=val_sampler)

                # Training models for this fold
                Trainer.fit(exp, train_dl_split, val_dl_split, verbose=self.verbose)

                # Evaluate the model on the validation set (val metrics)
                val_loss, accuracy, precision, f1, recall = Trainer.evaluate(exp, val_dl_split)

                cross_val_loss.append(float(val_loss))

                if accuracy is not Trainer.NA_MESSAGE:
                    cross_val_accuracy.append(float(accuracy.cpu()) if torch.is_tensor(accuracy) else float(accuracy))

                if precision is not Trainer.NA_MESSAGE:
                    cross_val_precision.append(float(precision.cpu()) if torch.is_tensor(precision) else float(precision))

                if f1 is not Trainer.NA_MESSAGE:
                    cross_val_f1.append(float(f1.cpu()) if torch.is_tensor(f1) else float(f1))

                if recall is not Trainer.NA_MESSAGE:
                    cross_val_recall.append(float(recall.cpu()) if torch.is_tensor(recall) else float(recall))

                if self.verbose:
                    print(f"Fold {fold+1}/{self.n_splits} | Val Loss: {val_loss} | Val Accuracy: {accuracy} | Val Precision: {precision} | Val F1: {f1} | Val Recall: {recall}")

            # Compute average metrics for this experiment
            experiment_results[exp.name] = {
                "val_loss": np.mean(cross_val_loss),
                "val_accuracy": np.mean(cross_val_accuracy),
                "val_precision": np.mean(cross_val_precision),
                "val_f1": np.mean(cross_val_f1),
                "val_recall": np.mean(cross_val_recall)
            }

            if self.verbose:
                print(f"Cross Validation Results for {exp.name}:")
                print(f"Val Loss: {experiment_results[exp.name]['val_loss']}")
                print(f"Val Accuracy: {experiment_results[exp.name]['val_accuracy']}")
                print(f"Val Precision: {experiment_results[exp.name]['val_precision']}")
                print(f"Val F1: {experiment_results[exp.name]['val_f1']}")
                print(f"Val Recall: {experiment_results[exp.name]['val_recall']}")

        return experiment_results





