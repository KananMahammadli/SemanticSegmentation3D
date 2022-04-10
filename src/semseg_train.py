# importing necessary libraries
import numpy as np
import torch
import torch.nn as nn
import argparse
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from dataloader import CustomDataLoader
from semseg_model import SemanticSegmentation
from visualize import DataVisualizer
from typing import List, Tuple, Union

# for adding results to the tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def args_from_terminal() -> argparse.Namespace:
    """
    Allows to take arguments from terminal with corresponding flags and stors in args variable.

    Parameters
    ----------
    None

    Returns
    -------
    args : argparse.Namespace
        Argparse Namespace object containing the arguments with corresponding values
    """

    parser = argparse.ArgumentParser(description='Model Training Program.')
    optional = parser._action_groups.pop() 
    required = parser.add_argument_group('required arguments')
    
    required.add_argument('--datapath', \
        help='path to the training data.', required=True)

    required.add_argument('--in_ch_size', \
        help='channel size of the input features.', type=int, required=True)

    optional.add_argument('--batch_size', \
        help='size of each batch for training.', default=32, type=int)

    optional.add_argument('--epochs', \
        help='number of epochs to train.', default=10, type=int)

    optional.add_argument('--shuffle', \
        help='pass it without any values to shuffle train data before splitting into batches (recommended to shuffle).', action='store_true')

    optional.add_argument('--lr', \
        help='learning rate for the optimizer.', default=0.001, type=float)

    optional.add_argument('--filters_list', \
        help='list containing filter sizes for hidden layers.', nargs='+', default=[16, 32, 64], type=int)

    optional.add_argument('--kernel_size', \
        help='kernel size for the filters', nargs='+', default=[3, 3, 3], type=int)

    optional.add_argument('--plot_training', \
        help='pass it without any value to draw training loss and accuracy.', action='store_true')

    optional.add_argument('--plot_destination', \
        help='if you have passed --plot_training argument, use this flag with corresponsing path to store the plot.', default='./loss_acc_plot.png')

    optional.add_argument('--dpi', \
        help='if you have passed --plot_training argument, use this flag to define dpi value to store the plot.', default=300, type=int)

    optional.add_argument('--plot_point_cloud', \
        help='if passed without any value, point cloud will be visualized.', action='store_true')

    optional.add_argument('--voxel_size', \
        help='starting voxel size for training.', default=0.02, type=float)

    parser._action_groups.append(optional)
    args = parser.parse_args()

    return args


class Trainer:
    """
    A class to train our model.

    Attributes
    ----------
    data_path : str
        Path to the data we will use for training and testing
    batch_size : int
        Batch size for splitting training data into batches
    shuffle : bool
        If True, then training examples will be shuffled before splitting into batches
    in_ch : int
        Number of input channels
    filters_list : List[int, int, int]
        A list of integeres representing the number of filters for hidden layers
    kernel_size : List[int, int, int]
        A list of integers for kernel size of each filter
    learning_rate : float
        Learning rate for optimizer
    epochs : int
        Number of epochs to train our model
    voxel_size : float
        Voxel size for the input features

    Methods
    -------
    prepare_data():
        Preparing the custom DataLoader for training
    train():
        Training our model
    """

    def __init__(self,
            data_path : str,
            batch_size : int,
            shuffle : bool,
            in_channels : int,
            filters_list :  List[Union[int, int, int]],
            kernel_size : List[Union[int, int, int]],
            learning_rate : float,
            epochs : int,
            voxel_size : float) -> None:
        """
        Constructs all the necessary attributes for the Trainer object

        Parameters
        ----------
            data_path : str
                Path to the data we will use for training and testing
            batch_size : int
                Batch size for splitting training data into batches
            shuffle : bool
                If True, then training examples will be shuffled before splitting into batches
            in_ch : int
                Number of input channels
            filters_list : List[int, int, int]
                A list of integeres representing the number of filters for hidden layers
            kernel_size : List[int, int, int], int
                A list of integer or a single integer for kernel size of each filter
            learning_rate : float
                Learning rate for optimizer
            epochs : int
                Number of epochs to train our model
            voxel_size : float
                Voxel size for the input features
        """

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.in_ch = in_channels
        self.filters_list = filters_list
        self.kernel_size = kernel_size
        self.lr = learning_rate
        self.epochs = epochs
        self.voxel_size = voxel_size

    def prepare_data(self) -> torch.utils.data.dataloader.DataLoader:
        """
        Preparing the custom DataLoader for training

        Parameters
        ---------
        None

        Returns
        -------
        torch.utils.data.dataloader.DataLoader
            DataLoader object containing the training data
        """

        # loading the data from path
        self.data = np.load(self.data_path)
        self.N = self.data.shape[0]

        # converting our numpy arrays to pytorch tensors
        self.train_data = torch.from_numpy(self.data[:, :-1])
        self.labels = self.data[:, -1]
        self.labels = torch.from_numpy(self.labels).type(torch.LongTensor)

        # putting our tensors into data loader
        data_loader = CustomDataLoader(self.train_data, self.labels, \
            batch_size=self.batch_size, shuffle=self.shuffle).load_data()

        return data_loader

    def train(self) -> Tuple[Union[SemanticSegmentation, dict]]:
        """
        Training our model

        Parameter
        ---------
        None

        Returns
        -------
        Tuple[model : semseg_model.SemanticSegmentation, history : dict]
            Returns a tuple: trained model and history dictionary containing the training loss and accuracy
        """
        data_loader = self.prepare_data()
        model = SemanticSegmentation(in_ch=self.in_ch, filters_list=self.filters_list,\
            kernel_size=self.kernel_size, voxel_size=self.voxel_size)

        # we use Cross Entropy Loss as our error function and Adam optimizer to update the weights
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr)

        train_loss = []
        train_acc = []

        for epoch in range(self.epochs):
            train_corr = 0
            for batch_idx, (X_train, y_train) in enumerate(data_loader):
                batch_idx += 1
                
                # first 3 columns - x, y, z coordinates; last 3 columns - R, G, B features
                in_pos = X_train[:, :3]
                in_feat = X_train[:, 3:]
                
                # calculating the loss and adding number representing predicted correct classes for the current batch 
                y_pred = model(in_feat, in_pos, in_pos)
                loss = criterion(y_pred, y_train)
                
                y_pred = torch.max(y_pred.data, 1)[1]
                train_corr += (y_pred == y_train).sum().item()
                
                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # analyzing the loss and training accuracy after each 400 batch
                if batch_idx % 400 == 0:
                    print(f"Epoch {epoch} batch {batch_idx} loss: {loss.item():.3f} accuracy: {((train_corr / (batch_idx * 32)) * 100):.3f} %")
                    train_loss.append(loss.item())
                    train_acc.append(train_corr / self.N)

            # writing loss, number of correct classes and accuracy to the tensorboard
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Correct/train", train_corr, epoch)
            writer.add_scalar("Accuracy/train", train_corr / self.N, epoch)

            # writing the biases of SparseConv layers to the tensorboard
            writer.add_histogram("conv1.bias", model.conv1.bias, epoch)
            writer.add_histogram("conv2.bias", model.conv2.bias, epoch)
            writer.add_histogram("conv3.bias", model.conv3.bias, epoch)

            # writing the biases of SparseConvTranspose lyers to the tensorboard
            writer.add_histogram("deconv1.bias", model.deconv1.bias, epoch)
            writer.add_histogram("deconv2.bias", model.deconv2.bias, epoch)
            writer.add_histogram("deconv3.bias", model.deconv3.bias, epoch)

            # lastly we add our final loss and training accuracy to furhter analyzing of changes over epoches
            train_loss.append(loss.item())
            train_acc.append(train_corr / self.N)

        history = {
            'training loss': train_loss,
            'training accuracy': train_acc
        }
        
        return (model, history)


if __name__ == '__main__':
    args = args_from_terminal()
    
    train_obj = Trainer(
        data_path=args.datapath,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        in_channels=args.in_ch_size,
        filters_list=args.filters_list,
        kernel_size=args.kernel_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        voxel_size=args.voxel_size
    )

    model, history = train_obj.train()

    positions = train_obj.data[:, :3]
    labels = train_obj.data[:, -1]
    vis = DataVisualizer(history=history, filename=args.plot_destination, dpi=args.dpi)

    if args.plot_training:
        vis.plot_history()
    
    if args.plot_point_cloud:
        vis.plot_data_cloud(positions, labels)

    # final result on the whole data
    in_pos = train_obj.train_data[:, :3]
    in_features = train_obj.train_data[:, 3:]
    y_pred = model(in_features, in_pos, in_pos)
    y_pred = torch.max(y_pred, 1)[1]
    y_pred_np = y_pred.detach().numpy()

    print(f'Final Accuracy is {((y_pred == train_obj.labels).sum().item() * 100 / train_obj.N):.3f}%')

    if args.plot_point_cloud:
        vis.plot_data_cloud(positions, y_pred_np)
    