import glob
import numpy as np
import torch
from math import pi
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class SineData(Dataset):
    """
    Dataset of functions f(x) = a * sin(x - b) where a and b are randomly
    sampled. The function is evaluated from -pi to pi.

    Parameters
    ----------
    amplitude_range : tuple of float
        Defines the range from which the amplitude (i.e. a) of the sine function
        is sampled.

    shift_range : tuple of float
        Defines the range from which the shift (i.e. b) of the sine function is
        sampled.

    num_samples : int
        Number of samples of the function contained in dataset.

    num_points : int
        Number of points at which to evaluate f(x) for x in [-pi, pi].
    """
    def __init__(self, amplitude_range=(-1., 1.), shift_range=(-.5, .5),
                 num_samples=1000, num_points=100):
        self.amplitude_range = amplitude_range
        self.shift_range = shift_range
        self.num_samples = num_samples
        self.num_points = num_points
        self.x_dim = 1  # x and y dim are fixed for this dataset.
        self.y_dim = 1

        # Generate data
        self.data = []
        a_min, a_max = amplitude_range
        b_min, b_max = shift_range
        for i in range(num_samples):
            # Sample random amplitude
            a = (a_max - a_min) * np.random.rand() + a_min
            # Sample random shift
            b = (b_max - b_min) * np.random.rand() + b_min
            # Shape (num_points, x_dim)
            x = torch.linspace(-pi, pi, num_points).unsqueeze(1)
            # Shape (num_points, y_dim)
            y = a * torch.sin(x - b)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

class QuadraticData(Dataset):
    def __init__(self, num_samples, num_xvals, sigma, sigma_x):
        self.name='quadratic'
        self.num_samples = num_samples
        self.num_xvals = num_xvals
        self.sigma = sigma
        self.sigma_x = sigma_x
        self.xvals = np.linspace(-2*sigma_x, 2*sigma_x, num_xvals)
        self.xmin = -2*sigma_x
        self.xmax = 2*sigma_x
        self.ymin = -5*sigma_x**2
        self.ymax = 5*sigma_x**2

    def __getitem__(self, index):
        if type(index)==int:
            sign = np.random.choice(np.array([-1,1]))
            offset = self.sigma * np.random.normal()
            sample_ = sign * self.xvals**2 + offset
            sample_ = torch.from_numpy(np.expand_dims(sample_, -1)).float()
            x_ = torch.from_numpy(np.expand_dims(self.xvals, -1)).float()
            return (x_, sample_)
        else:
            ns = len(index)
            samples=[]
            for i in range(ns):
                sign = np.random.choice(np.array([-1,1]))
                offset = self.sigma * np.random.normal()
                sample_ = sign * self.xvals**2 + offset
                sample_ = torch.from_numpy(np.expand_dims(sample_, -1)).float()
                x_ = torch.from_numpy(np.expand_dims(self.xvals, -1)).float()
                samples.append((x_, sample_))
            return samples

    def __len__(self):
        return self.num_samples
    

class CSVDataset(Dataset):
    def __init__(self, file, name):
        self.name=name
        self.file = file
        if name == 'melbourne':
            self.xvals = np.arange(24)
            self.num_xvals = 24
            self.ymin = -2
            self.ymax = 6
        elif name == 'gridwatch':
            self.xvals = np.linspace(0,24-1/288,288)
            self.num_xvals = 288
            self.ymin = -3
            self.ymax = 3
        elif name == 'quadratic':
            self.ymin = -3
            self.ymax = 3
            self.num_xvals = 100
            self.xvals = np.linspace(-10, 10, 100)
        else:
            NotImplementedError()

        self.data = self.get_samples()

    def get_samples(self):
        if 'csv' in self.file:
            data_ = np.genfromtxt(self.file, delimiter=',')
        elif 'npy' in self.file:
            data_ = np.load(self.file, allow_pickle=True)
        if len(data_.shape) == 3:
            data_ = data_[:,:,0]
        self.num_samples = data_.shape[0]
        samples = []
        for j in range(data_.shape[0]):
            x_ = torch.from_numpy(np.expand_dims(self.xvals, -1)).float()
            sample_ = torch.from_numpy(np.expand_dims(data_[j], -1)).float()
            samples.append((x_, sample_))
        return samples

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def mnist(batch_size=16, size=28, path_to_data='../../mnist_data'):
    """MNIST dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image. Default is 28 for no resizing.

    path_to_data : string
        Path to MNIST data files.
    """
    all_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    train_data = datasets.MNIST(path_to_data, train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST(path_to_data, train=False,
                               transform=all_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def celeba(batch_size=16, size=32, crop=89, path_to_data='../celeba_data',
           shuffle=True):
    """CelebA dataloader.

    Parameters
    ----------
    batch_size : int

    size : int
        Size (height and width) of each image.

    crop : int
        Size of center crop. This crop happens *before* the resizing.

    path_to_data : string
        Path to CelebA data files.
    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop),
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    celeba_data = CelebADataset(path_to_data,
                                transform=transform)
    celeba_loader = DataLoader(celeba_data, batch_size=batch_size,
                               shuffle=shuffle)
    return celeba_loader


class CelebADataset(Dataset):
    """CelebA dataset."""
    def __init__(self, path_to_data, subsample=1, transform=None):
        """
        Parameters
        ----------
        path_to_data : string
            Path to CelebA data files.

        subsample : int
            Only load every |subsample| number of images.

        transform : torchvision.transforms
            Torchvision transforms to be applied to each image.
        """
        self.img_paths = glob.glob(path_to_data + '/*.jpg')[::subsample]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        sample_path = self.img_paths[idx]
        sample = Image.open(sample_path)

        if self.transform:
            sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        return sample, 0
