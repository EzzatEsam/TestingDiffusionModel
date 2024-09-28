import torch as T
import matplotlib.pyplot as plt
import os
import pandas as pd

import os
import matplotlib.pyplot as plt


def save_images(
    imgs,
    location=r"/teamspace/studios/this_studio/runs",
    epoch_n=0,
    version=0,
):
    """
    Save a list of tensors, each containing multiple images generated together at the same timestep.

    Args:
        imgs (list): A list of tensors, each containing multiple images at the same timestep.
        location (str, optional): The directory path where the images will be saved. Defaults to "src/results".
        epoch_n (int, optional): The epoch number associated with the images. Defaults to 0.
        version (int, optional): The version number associated with the images. Defaults to 0.

    Returns:
        None
    """

    save_path = f"{location}/version{version}/imgs"
    os.makedirs(save_path, exist_ok=True)

    rows = imgs[0].shape[0]
    cols = len(imgs)

    fig, axs = plt.subplots(rows, cols, figsize=(20, 2 * rows))

    for i in range(rows):
        for j in range(cols):
            img = imgs[j][i].to("cpu")
            img = 0.5 * img + 0.5
            img = T.clip(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                img = img[:, :, 0]
            # Handle potential 1D axs array
            if rows == 1:
                axs[j].imshow(img, cmap="gray" if len(img.shape) == 2 else None)
                axs[j].axis("off")
            else:
                axs[i, j].imshow(img, cmap="gray" if len(img.shape) == 2  else None)
                axs[i, j].axis("off")

    plt.suptitle(f"Version {version} - Epoch {epoch_n}", fontsize=26)
    plt.savefig(f"{save_path}/epoch_{epoch_n}.png")
    plt.close(fig)

    # Save the last image set
    last_img_set = imgs[-1].to("cpu")
    for k in range(last_img_set.shape[0]):
        last_img = last_img_set[k]
        last_img = 0.5 * last_img + 0.5
        last_img = T.clip(last_img, 0, 1)
        last_img = last_img.permute(1, 2, 0).numpy()
        if last_img.shape[2] == 1:
            last_img = last_img[:, :, 0]

        plt.imsave(
            f"{save_path}/epoch_{epoch_n}_last_{k}.png",
            last_img,
            cmap="gray" if last_img.ndim == 2 else None,
        )
    print(f"Saved last image set with {last_img_set.shape[0]} images.")


def save_losses_graph(
    val_losses: list[float],
    train_losses: list[float],
    location=r"/teamspace/studios/this_studio/runs",
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

    os.makedirs(location + f"/version{version}/graphs", exist_ok=True)

    losses_df = pd.DataFrame(
        {
            "Epoch": range(1, len(train_losses) + 1),
            "Training Loss": train_losses,
            "Validation Loss": val_losses,
        }
    )
    losses_df.to_csv(f"{location}/version{version}/graphs/losses.csv", index=False)

    plt.figure(figsize=(12, 8))

    # Enhanced plotting
    plt.plot(val_losses, label="Validation Loss", color='blue', linestyle='--', marker='o', markersize=5)
    plt.plot(train_losses, label="Training Loss", color='orange', linestyle='-', marker='x', markersize=5)

    plt.title("Training and Validation Loss Over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.axhline(y=min(val_losses), color='red', linestyle=':', label='Minimum Val Loss')
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{location}/version{version}/graphs/losses.png")
    plt.close()


def save_model(
    model: T.nn.Module,
    location: str = "/teamspace/studios/this_studio/runs",
    epoch_n: int = 0,
    version=0,
):
    """
    Save the state dictionary of a PyTorch model to a file.

    Args:
        model (torch.nn.Module): The model to save.
        location (str, optional): The directory to save the model file. Defaults to "/teamspace/studios/this_studio/runs/models".
        epoch_n (int, optional): The epoch number associated with the model. Defaults to 0.
        version (int, optional): The version number associated with the model. Defaults to 0.

    Returns:
        None

    This function saves the state dictionary of a PyTorch model to a file. The state dictionary contains the parameters of the model. The file is saved in the specified location with the name "version{version}_epoch_{epoch_n}.pt".

    Note:
        This function requires the PyTorch library to be installed.

    """
    os.makedirs(location + f"/version{version}/models", exist_ok=True)
    T.save(model.state_dict(), f"{location}/version{version}/models/epoch_{epoch_n}.pt")
