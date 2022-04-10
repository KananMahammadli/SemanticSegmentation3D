import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union

class MyDataset(Dataset):
    """
    A class to create custom dataset from tensors.

    Attributes
    --------
    data : torch.Tensor
        Input features and positions
    targets: torch.Tensor
        Target column
    """
    def __init__(self, data : torch.Tensor, targets : torch.Tensor) -> None:
        """
        Constructs all the necessary attributes for the MyDataset object.

        Parameters
        ----------
            data : torch.Tensor
                Input features and positions
            targets: torch.Tensor
                Target column
        """
        self.data = data
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index : torch.Tensor) -> Tuple[Union[torch.Tensor, torch.Tensor]]:
        x = self.data[index]
        y = self.targets[index]
        
        return (x, y)
    
    def __len__(self) -> int:
        return len(self.data)

class CustomDataLoader:
    """
    A class to create custom data loader for batch learning.

    Attributes
    ---------
    X_train : torch.Tensor
        Tensor containing input features and positions
    y_train : torch.Tensor
        Tensor containing the target class
    batch_size : int
        Batch size for splitting training data into batches
    shuffle : bool
        If True, then training examples will be shuffled before splitting into batches

    Methods
    ------
    load_data():
        Return a DataLoader object containing the training data
    """
    def __init__(self, X_train : torch.Tensor, y_train : torch.tensor, batch_size : int, shuffle : bool=False) -> None:
        """
        Constructs all the necessary attributes for the CustomDataLoader object.

        Parameters
        ----------
            X_train : torch.Tensor
                Tensor containing input features and positions
            y_train : torch.Tensor
                Tensor containing the target class
            batch_size : int
                Batch size for splitting training data into batches
            shuffle : bool
                If True, then training examples will be shuffled before splitting into batches
        """
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load_data(self) -> torch.utils.data.dataloader.DataLoader:
        """
        Loads the training data into custom data loader

        Parameters
        ----------
            None
        
        Returns
        ------
        data_loader : torch.utils.data.dataloader.DataLoader
            DataLoader object containing the training data
        """
        dataset = MyDataset(self.X_train, self.y_train)
        data_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return data_loader
        