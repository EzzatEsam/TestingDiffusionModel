import argparse
import torch as T
import torch.optim as optim
import torch.nn.functional as F
from DiffusionTest import train
from DiffusionTest.model import SmallUnetWithEmb
from DiffusionTest.diffusion import Diffusion
from DiffusionTest.loader import get_loaders
from DiffusionTest.utils import save_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a diffusion model or generate images."
    )

    # Arguments related to device and training mode
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if T.cuda.is_available() else "cpu",
        help="Device to run the model on, 'cuda' or 'cpu'. Default is based on availability.",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        default=False,
        help="Flag to train the model. If not set, images will be generated instead.",
    )

    # Arguments related to saving, dataset and model configurations
    parser.add_argument(
        "--saving_path",
        type=str,
        default="/teamspace/studios/this_studio/runs",
        help="Path where model checkpoints and images are saved.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=12,
        help="Version number for saving models and images.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["celeb", "bridge", "cifar10", "fashion"],
        default="celeb",
        help="Dataset to use. Default is 'celeb'.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Image size to use for training and generation. Default is 64x64.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for the dataloader. Default is 64.",
    )

    # Arguments related to training specifics
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="The starting epoch number for resuming training. Default is 0.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1001,
        help="Maximum number of epochs for training. Default is 1001.",
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        default=2,
        help="Frequency (in epochs) to save the model checkpoint and generate images. Default is every 2 epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer. Default is 1e-4.",
    )

    parser.add_argument(
        "--conditional",
        action="store_true",
        default=False,
        help="Flag to use conditional generation. Default is False.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = T.device(args.device)
    print(f"Using device: {device}")

    print(f"Mode is {'training' if args.training else 'generating'}")
    saving_path = args.saving_path
    training = args.training
    version = args.version
    ds = args.dataset
    img_size = args.img_size
    batch_size = args.batch_size
    start_epoch = args.start_epoch
    epochs = args.epochs
    conditional = args.conditional
    save_every_n = args.save_every_n
    lr = args.learning_rate

    chans = 1 if ds == "fashion" else 3
    n_classes = None
    if conditional:
        if ds == "fashion":
            n_classes = 10
            class_names = [
                "T-shirt/top",  # 0
                "Trouser",  # 1
                "Pullover",  # 2
                "Dress",  # 3
                "Coat",  # 4
                "Sandal",  # 5
                "Shirt",  # 6
                "Sneaker",  # 7
                "Bag",  # 8
                "Ankle boot",  # 9
            ]

        elif ds == "cifar10":
            class_names = [
                "Airplane",  # 0
                "Automobile",  # 1
                "Bird",  # 2
                "Cat",  # 3
                "Deer",  # 4
                "Dog",  # 5
                "Frog",  # 6
                "Horse",  # 7
                "Ship",  # 8
                "Truck",  # 9
            ]

            n_classes = 10

    model = SmallUnetWithEmb(img_channels=chans, n_classes=n_classes).to(device)

    if start_epoch > 0:
        model.load_state_dict(
            T.load(
                saving_path + f"/version{version}/models/epoch_{start_epoch-1}.pt",
                map_location=device,
            )
        )
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    diff = Diffusion(device=device, img_channels=chans, img_size=img_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = F.mse_loss

    if not training:
        print("Generating images")
        if n_classes is not None:
            print("Generating images with conditional generation")
            if n_classes is not None:
                classes = []
                for cls in range(n_classes):
                    classes += [cls] * 2

                classes = T.tensor(classes).to(device)
                imgs = diff.generate_sample(
                    model, n_images=len(classes), labels=classes
                )
                class_names = [class_names[cls] for cls in classes.cpu().tolist()]
                save_images(
                    imgs,
                    version=version,
                    epoch_n=start_epoch,
                    classes_list=class_names,
                )
        else:
            print("Generating images without conditional generation")
            imgs = diff.generate_sample(model, n_images=10)
            save_images(imgs, version=version, epoch_n=start_epoch)

    else:
        print("Training model")
        print(f"Using dataset: {ds} with image size: {img_size}")
        print(f"Using batch size: {batch_size}")
        print(f"Is conditional: {conditional}")
        print(f"Number of classes: {n_classes}")
        train_loader, val_loader = get_loaders(ds, batch_size, img_size)
        train(
            model=model,
            diff=diff,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            apply_ema=False,
            epochs=epochs,
            version=version,
            start_epoch=start_epoch,
            n_classes=n_classes,
            save_every_n=save_every_n,
        )
