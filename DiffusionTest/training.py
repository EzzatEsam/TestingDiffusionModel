from DiffusionTest.utils import save_images, save_losses_graph, save_model
from .diffusion import Diffusion
from .model import SmallUnetWithEmb
import torch as T
import torch.optim as optim
import torch.nn.functional as F
from .loader import train_loader, val_loader
from tqdm import tqdm
from typing import Callable


def train(
    epochs: int,
    model: T.nn.Module,
    diff: Diffusion,
    optimizer: T.optim.Optimizer,
    criterion: Callable,
    device,
    version=0,
):
    
    train_loss : list[float] = []
    val_loss : list[float] = []
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, position=0, leave=True)
        losses = []
        for _, (data, _) in enumerate(pbar):
            data = data.to(device)
            pbar.set_description(f"Epoch {epoch} training")
            t = diff.get_time_samples(data.shape[0])
            noisy_imgs, noise = diff.add_noise(data, t)

            noisy_imgs = noisy_imgs.to(device)
            noise = noise.to(device)
            # print(noisy_imgs._version)
            # print(noise._version)
            time_vec = diff.time_encode(t)

            optimizer.zero_grad()
            out = model(noisy_imgs, time_vec)
            loss = criterion(out, noise)
            loss.backward()
            optimizer.step()

            losses += [loss.item()]
            
            pbar.set_postfix(
                {
                    "training_loss": loss.item(),
                    "training_loss_avg": sum(losses) / len(losses),
                }
            )

        train_loss += [sum(losses) / len(losses)]
        model.eval()
        losses = []
        with T.no_grad():
            pbar = tqdm(val_loader, position=0, leave=True)
            for _, (data, _) in enumerate(pbar):
                data = data.to(device)
                pbar.set_description(f"Epoch {epoch}")
                t = diff.get_time_samples(data.shape[0])
                noisy_imgs, noise = diff.add_noise(data, t)
                time_vec = diff.time_encode(t).to(device)

                out = model(noisy_imgs, time_vec)
                loss = criterion(out, noise)
                losses += [loss.item()]
                pbar.set_postfix(
                    {
                        "validation_loss": loss.item(),
                        "validation_loss_avg": sum(losses) / len(losses),
                    }
                )
                
        val_loss += [sum(losses) / len(losses)]
        save_losses_graph(val_loss, train_loss, version= version)

        if epoch % 2 == 0 :
            print("Generating a sample...")
            imgs = diff.generate_sample(model)
            save_images(imgs ,version= version , epoch_n= epoch)
            save_model(model, epoch_n= epoch , version= version)
