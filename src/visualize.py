import open3d.ml.torch as ml3d
import matplotlib.pyplot as plt
import numpy

class DataVisualizer:
    """
     A class to visualize the training accuracy/loss and also data point cloud
    
    Attributes
    ----------
    points : numpy.array
        Numpy array containing x, y, z coordinates of each point
    labels : numpy.array
        Numpy array containing the correct labels for each point
    history : dict
        A dictionary containing the loss and accuracy for training process
    filename : string
        A path to store the result of loss and accuracy plot over epochs
    dpi : int
        dpi value for saved plot

    Methods
    ------
    plot_history():
        Plots accuracy and loss for the training process over epochs
    plot_data_cloud(point : numpy.array, labels : numpy.array):
        Plots the data points
    """

    def __init__(self, history : dict,  filename: str='loss_and_accuracy.png', dpi : int=300) -> None:
        """
        Constructs all the necessary attributes for the DataVisualizer object

        Parameters
        ----------
            history : dict
                A dictionary containing the loss and accuracy for training process
            filename : string
                A path to store the result of loss and accuracy plot over epochs
            dpi : int
                dpi value for saved plot
        """

        self.history = history
        self.filename = filename
        self.dpi = dpi

    def plot_history(self) -> None:
        """
        Plots accuracy and loss for the training process over epochs

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 8))
        ax[0].set_title("Loss vs Epoch")
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].plot(self.history['training loss'], label = "training loss")

        ax[1].set_title("Accuracy vs Epoch")
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].plot([acc * 100 for acc in self.history['training accuracy']], label = "training accuracy")
        plt.show()
        
        fig.savefig(fname=self.filename, dpi=self.dpi)

    def plot_data_cloud(self, points : numpy.array, labels : numpy.array) -> None:
        """
        Plots the data points

        Parameters
        ----------
        points : numpy.array
            Numpy array containing x, y, z coordinates of each point
        labels : numpy.array
             Numpy array containing the correct labels for each point

        Returns
        -------
        None
        """

        data_dict = [
            {
                'name': 'true_point_cloud',
                'points': points,
                'random_colors': labels.astype('int32'),
                'int_attr': (points[:,0]*5).astype('int32'),
            }
        ]

        vis = ml3d.vis.Visualizer()
        vis.visualize(data_dict)
