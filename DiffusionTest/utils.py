import torch as T
import matplotlib.pyplot as plt
import os


def save_images(imgs, rows=1, cols=10, location=r"src/results", epoch_n=0, version=0):
    """
        Save a list of images to a specified location.

        Args:
            imgs (list): A list of images to be saved.
            rows (int, optional): The number of rows in the grid of images. Defaults to 1.
            cols (int, optional): The number of columns in the grid of images. Defaults to 10.
            location (str, optional): The directory path where the images will be saved. Defaults to "src/results".
            epoch_n (int, optional): The epoch number associated with the images. Defaults to 0.
            version (int, optional): The version number associated with the images. Defaults to 0.

        Returns:
            None

    This function saves a grid of images to a specified location. The images are displayed in a grid of specified rows and columns. The images are first converted to the CPU, then normalized to the range [0, 1], and finally permuted to the format (channel, height, width). The grid of images is saved as a PNG file with the name "epoch_{epoch_n}_version{version}.png". The last image in the list is saved separately as a PNG file with the name "epoch_{epoch_n}_version{version}_last.png".

        Note:
            This function requires the matplotlib and PyTorch libraries to be installed.

    """

    os.makedirs(location + f"/version{version}", exist_ok=True)
    fig, axs = plt.subplots(rows, cols, figsize=(20, 2))
    for i in range(rows):
        for j in range(cols):
            img = imgs[i * cols + j].to("cpu")
            img = 0.5 * img + 0.5
            img = T.clip(img, 0, 1)
            img = img[0].permute(1, 2, 0)
            axs[j].imshow(img)
            axs[j].axis("off")
    plt.savefig(f"{location}/version{version}/epoch_{epoch_n}.png")
    plt.close(fig)
    last_img = imgs[-1].to("cpu")
    last_img = 0.5 * last_img + 0.5
    last_img = T.clip(last_img, 0, 1)
    last_img = last_img[0].permute(1, 2, 0).numpy()
    plt.imsave(f"{location}/version{version}/epoch_{epoch_n}_last.png", last_img)


def save_losses_graph(
    val_losses: list[float],
    train_losses: list[float],
    location=r"src/results",
    version=0,
):
    """
    Saves the losses of the validation and training datasets as a plot in the specified location.

    Args:
        val_losses (list[float]): A list of validation losses.
        train_losses (list[float]): A list of training losses.
        location (str, optional): The directory path where the plot will be saved. Defaults to "src/results".
        version (int, optional): The version number to be appended to the directory path. Defaults to 0.

    Returns:
        None
    """

    os.makedirs(location + f"/version{version}", exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.plot(val_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(f"{location}/version{version}/losses.png")
    plt.close()


def save_model(
    model: T.nn.Module,
    location: str = "/teamspace/studios/this_studio/src/models",
    epoch_n: int = 0,
    version=0,
):
    """
    Save the state dictionary of a PyTorch model to a file.

    Args:
        model (torch.nn.Module): The model to save.
        location (str, optional): The directory to save the model file. Defaults to "/teamspace/studios/this_studio/src/models".
        epoch_n (int, optional): The epoch number associated with the model. Defaults to 0.
        version (int, optional): The version number associated with the model. Defaults to 0.

    Returns:
        None

    This function saves the state dictionary of a PyTorch model to a file. The state dictionary contains the parameters of the model. The file is saved in the specified location with the name "version{version}_epoch_{epoch_n}.pt".

    Note:
        This function requires the PyTorch library to be installed.

    """
    os.makedirs(location + f"/version{version}", exist_ok=True)
    T.save(model.state_dict(), f"{location}/version{version}/epoch_{epoch_n}.pt")
