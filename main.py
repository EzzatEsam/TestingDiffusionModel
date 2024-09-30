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
        help="Total number of epochs for training. Default is 1001.",
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        default=2,
        help="Frequency (in epochs) to save the model checkpoint. Default is every 2 epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer. Default is 1e-4.",
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
    chans = 3 if ds == "celeb" else 1
    img_size = args.img_size
    batch_size = args.batch_size
    start_epoch = args.start_epoch
    epochs = args.epochs
    save_every_n = args.save_every_n
    lr = args.learning_rate

    model = SmallUnetWithEmb(img_channels=chans).to(device)

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
        imgs = diff.generate_sample(model, n_images=10)
        save_images(imgs, version=version, epoch_n=start_epoch)

    else:
        print("Training model")
        print(f"Using dataset: {ds} with image size: {img_size}")
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
            save_every_n=save_every_n,
        )
