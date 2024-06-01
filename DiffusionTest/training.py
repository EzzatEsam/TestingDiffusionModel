from DiffusionTest.utils import save_images, save_losses_graph, save_model
from .diffusion import Diffusion
import torch as T
from tqdm import tqdm
from typing import Callable
from .ema_updater import EmaUpdater

def train(
    epochs: int,
    model: T.nn.Module,
    diff: Diffusion,
    optimizer: T.optim.Optimizer,
    criterion: Callable,
    train_loader  , 
    val_loader,
    device,
    is_conditional: bool = False,
    hg_model: bool = False,
    apply_ema: bool = False,
    version: int = 0,
    start_epoch: int = 0,
    save_every_n: int = 2
):
    """
    Trains a model using the diffusion training algorithm.

    Args:
        epochs (int): The number of training epochs.
        model (T.nn.Module): The model to be trained.
        diff (Diffusion): The diffusion object used for adding noise to the data.
        optimizer (T.optim.Optimizer): The optimizer used for updating the model's parameters.
        criterion (Callable): The loss function used for calculating the training loss.
        device: The device (e.g., 'cpu', 'cuda') on which the training will be performed.
        hg_model (bool, optional): Whether the model is a hugging face diffusion model. Defaults to False.
        apply_ema (bool, optional): Whether to apply exponential moving average (EMA) to the model's parameters. 
            Defaults to False.
        version (int, optional): The version number of the training. Defaults to 0.
        start_epoch (int, optional): The starting epoch number. Defaults to 0.
        save_every_n (int, optional): The interval at which to save the model and generated samples. Defaults to 2.
    """
    if apply_ema:
        updater = EmaUpdater(model, 0.995, start_after=200)
    train_loss: list[float] = []
    val_loss: list[float] = []

    for epoch in range(epochs):
        epoch += start_epoch
        model.train()
        pbar = tqdm(train_loader, position=0, leave=True)
        losses = []
        for _, (data, y) in enumerate(pbar):
            data = data.to(device)
            pbar.set_description(f"Epoch {epoch} training")
            t = diff.get_time_samples(data.shape[0])
            noisy_imgs, noise = diff.add_noise(data, t)

            noisy_imgs = noisy_imgs.to(device)
            noise = noise.to(device)
                
            # print(noisy_imgs._version)
            # print(noise._version)
            optimizer.zero_grad()
            if hg_model:
                time_vec = t.to(device)
            else:
                time_vec = diff.time_encode(t)

            if is_conditional:
                y = y.to(device)
                out = model(noisy_imgs, time_vec , y)
            else :
                out = model(noisy_imgs, time_vec)
            
            if hg_model:
                out = out.sample


            loss = criterion(out, noise)
            loss.backward()
            optimizer.step()
            if apply_ema:
                updater.update(model)

            losses += [loss.item()]

            pbar.set_postfix(
                {
                    "training_loss": f"{loss.item() :.4f}",
                    "training_loss_avg": f"{sum(losses) / len(losses)  : .4f}",
                }
            )

        train_loss += [sum(losses) / len(losses)]
        model.eval()
        losses = []
        with T.no_grad():
            pbar = tqdm(val_loader, position=0, leave=True)
            for _, (data, y) in enumerate(pbar):
                data = data.to(device)
                pbar.set_description(f"Epoch {epoch}")
                t = diff.get_time_samples(data.shape[0])
                noisy_imgs, noise = diff.add_noise(data, t)
                time_vec = diff.time_encode(t).to(device)

                    
                if hg_model:
                    time_vec = t.to(device)
                else:
                    time_vec = diff.time_encode(t)

                if is_conditional:
                    y = y.to(device)
                    out = model(noisy_imgs, time_vec , y)
                else :
                    out = model(noisy_imgs, time_vec)
                
                if hg_model:
                    out = out.sample
                loss = criterion(out, noise)
                losses += [loss.item()]
                pbar.set_postfix(
                    {
                        "validation_loss": f"{loss.item() :.4f}",
                        "validation_loss_avg":f"{sum(losses) / len(losses) : .4f}",
                    }
                )

        val_loss += [sum(losses) / len(losses)]
        save_losses_graph(val_loss, train_loss, version=version)

        if epoch % save_every_n == 0:
            print("Generating a sample...")
            imgs = diff.generate_sample(model)
            save_images(imgs, version=version, epoch_n=epoch)
            save_model(model, epoch_n=epoch, version=version)
