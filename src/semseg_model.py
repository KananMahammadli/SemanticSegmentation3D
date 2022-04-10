from open3d.ml.torch.layers import SparseConv, SparseConvTranspose
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union

class SemanticSegmentation(nn.Module):
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

    Methods
    ------
    forward(in_feat, in_pos, out_pos):
        Returns a tensor of siz (N, num_of_classes) where N is number of samples,
        A result of feed forward network

    """
    def __init__(self, in_ch : int, filters_list : List[Union[int, int, int]], kernel_size: List[Union[int, int, int]], voxel_size : float) -> None:
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
        """
        super().__init__()

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
        