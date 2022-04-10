# use relative import for all imports within ml3d.
from open3d.ml.torch.layers import SparseConv, SparseConvTranspose
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from typing import List, Union

class MyModel(BaseModel):
    """
    A class to creating 3D sementic segmentation model with SparseConv and SparseConvTranspose layers

    Attributes
    ---------
    in_ch : int
        Number of input channels
    filters_list : List[int, int, int]
        A list of integeres representing the number of filters for hidden layers
    kernel_size : List[int, int, int]
        A list of integers for kernel size of each filter
    voxel_size : float
        Voxel size for the input features
    name : str
        Name of the model

    Methods
    ------
    forward(in_feat, in_pos, out_pos):
        Returns a tensor of siz (N, num_of_classes) where N is number of samples,
        A result of feed forward network
    get_optimizer(cfg_pipeline):
        Returns optimizer and scheduler
    get_loss(cfg_pipeline):
        Returns the loss values, labels and the results
    process(data, attr):
        Return the data
    """

    def __init__(self,  
            in_ch : int,
            filters_list : List[Union[int, int, int]], 
            kernel_size: List[Union[int, int, int]], 
            voxel_size : float,
            name : str="Semantic Segmentation") -> None:
        
        super().__init__(name=name)
        """
        Comstructs all the necessary attributes for the SemanticSegmentation class

        Parameters
        ---------
            in_ch : int
                Number of input channels
            filters_list : List[int, int, int]
                A list of integeres representing the number of filters for hidden layers
            kernel_size : List[int, int, int]
                A list of integers for kernel size of each filter
            voxel_size : float
                Voxel size for the input features
            name : str
                Name of the model
        """

        self.voxel_size = voxel_size
        self.conv1 = SparseConv(in_channels=in_ch, filters=filters_list[0], kernel_size=kernel_size)
        self.conv2 = SparseConv(in_channels=filters_list[0], filters=filters_list[1], kernel_size=kernel_size)
        self.conv3 = SparseConv(in_channels=filters_list[1], filters=filters_list[2], kernel_size=kernel_size)
        
        self.deconv1 = SparseConvTranspose(in_channels=filters_list[2], filters=filters_list[2], kernel_size=kernel_size)
        self.deconv2 = SparseConvTranspose(in_channels=filters_list[2], filters=filters_list[1], kernel_size=kernel_size)
        self.deconv3 = SparseConvTranspose(in_channels=filters_list[1], filters=13, kernel_size=kernel_size)

    def forward(self, in_feat : torch.Tensor, in_pos : torch.tensor, out_pos : torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of siz (N, num_of_classes) where N is number of samples,
        A result of feed forward network

        Parameters
        ---------
        in_feat : torch.Tensor
            Input features with R, G, B values
        in_pos : torch.Tensor
            Input positions with x, y, z coordinates
        out_pos : torch.Tensor
            Output positions with x, y, z coordinates

        Returns
        ------
        out_feat : torch.Tensor
            A tensor with size of (N, number_of_classes), where N is the number of samples
        """

        out_feat = F.relu(self.conv1(in_feat, in_pos, out_pos, self.voxel_size))
        out_feat = F.relu(self.conv2(out_feat, in_pos, out_pos, self.voxel_size*2))
        out_feat = F.relu(self.conv3(out_feat, in_pos, out_pos, self.voxel_size*4))
        
        out_feat = F.relu(self.deconv1(out_feat, in_pos, out_pos, self.voxel_size*4))
        out_feat = F.relu(self.deconv2(out_feat, in_pos, out_pos, self.voxel_size*2))
        out_feat = self.deconv3(out_feat, in_pos, out_pos, self.voxel_size)
        return out_feat

    def get_optimizer(self, cfg_pipeline):
        """
        Returns optimizer and scheuler
        
        Parameters
        ----------
        cfg_pipeline : object
            Configuration pipeline containing the optimizer and scheduler
        
        Returns
        -------
        optimizer
            Optimizer for updating weights
        scheduler
            Scheduler for changing the learning rate
        """

        optimizer = torch.optim.Adam(self.parameters(), lr=cfg_pipeline.adam_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg_pipeline.scheduler_gamma)
        return optimizer, scheduler

    def get_loss(self, Loss, results, inputs):
        """
        Returns the loss, labels, and results
        
        Parameters
        ----------
        loss : object
            Weighted Cross Entropy loss function
        labels : torch.Tensor
            Filtered labels
        results : torch.Tensor
            Filtered results
        """
        labels = inputs['data'].labels # processed data from model.preprocess and/or model.transform.

        # Loss is an object of type SemSegLoss. Any new loss can be added to `ml3d/{tf, torch}/modules/semseg_loss.py`
        loss = Loss.weighted_CrossEntropyLoss(results, labels)
        results, labels = Loss.filter_valid_label(results, labels) # remove ignored indices if present.
        return loss, labels, results

    def preprocess(self, data, attr):
        """
        Preparing the data
        
        Parameters
        ----------
        data : torch.Tensor
            Data that will be processed
        attr : any
            No info
            
        Returns
        -------
        data : torch.Tensor
            Processed data"""
        return data