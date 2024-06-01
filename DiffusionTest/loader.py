import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define the path to your dataset
celeb_data_dir = r"/teamspace/studios/this_studio/datasets/celeba_hq_root"
bridge_data_dir = r"/teamspace/studios/this_studio/datasets/lsun/bridge_root"


def get_loaders(name : str , batch_sz : int , img_sz : int) : 
    """
    Given the name of a dataset, the batch size, and the image size, returns the corresponding data loaders for training and validation.

    Parameters:
        name (str): The name of the dataset. Must be one of 'celeb', 'bridge', 'cifar10', or 'fashion'.
        batch_sz (int): The size of each batch.
        img_sz (int): The size of each image.

    Returns:
        tuple: A tuple containing the data loaders for training and validation. If the dataset is 'celeb' or 'bridge', the tuple contains two data loaders. If the dataset is 'cifar10' or 'fashion', the tuple contains two data loaders for training and testing.

    Raises:
        ValueError: If the dataset name is not one of 'celeb', 'bridge', 'cifar10', or 'fashion'.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((img_sz, img_sz)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    if name == 'celeb' :
        dataset = datasets.ImageFolder(celeb_data_dir, transform=transform)

        train , val = torch.utils.data.random_split(dataset, [0.8, 0.2])
        trainloader = DataLoader(train, batch_size=batch_sz, shuffle=True)
        valloader = DataLoader(val, batch_size=batch_sz, shuffle=False)

        return trainloader , valloader

    elif name == 'bridge' :
        dataset = datasets.ImageFolder(bridge_data_dir, transform=transform)

        train , val = torch.utils.data.random_split(dataset, [0.8, 0.2])
        trainloader = DataLoader(train, batch_size=batch_sz, shuffle=True)
        valloader = DataLoader(val, batch_size=batch_sz, shuffle=False)

        return trainloader , valloader

    elif name == 'cifar10' :

        # Download and load the training dataset
        trainset = datasets.CIFAR10(
            root="/teamspace/studios/this_studio/datasets",
            train=True,
            download=True,
            transform=transform,
        )
        trainloader = DataLoader(trainset, batch_size=batch_sz, shuffle=True, num_workers=2)

        # Download and load the test dataset
        testset = datasets.CIFAR10(
            root="/teamspace/studios/this_studio/datasets",
            train=False,
            download=True,
            transform=transform,
        )
        testloader = DataLoader(testset, batch_size=batch_sz, shuffle=False, num_workers=2)

        return trainloader , testloader

    elif name == 'fashion' :

        # Download and load the training dataset
        trainset = datasets.FashionMNIST(
            root="/teamspace/studios/this_studio/datasets",
            train=True,
            download=True,
            transform=transform,
        )
        trainloader = DataLoader(trainset, batch_size=batch_sz, shuffle=True, num_workers=2)  

        testset = datasets.FashionMNIST(
            root="/teamspace/studios/this_studio/datasets",
            train=False,
            download=True,
            transform=transform,
        )
        testloader = DataLoader(testset, batch_size=batch_sz, shuffle=False, num_workers=2)

        return trainloader , testloader
