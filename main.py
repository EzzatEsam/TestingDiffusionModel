from DiffusionTest import train
import torch as T
import torch.optim as optim
from DiffusionTest.model import SmallUnetWithEmb
from DiffusionTest.diffusion import Diffusion
import torch.nn.functional as F
from diffusers.models import UNet2DModel
from DiffusionTest.loader import get_loaders

if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # import logging
    # logging.basicConfig( level=logging.INFO  , format="%(asctime)s - %(levelname)s - %(message)s")

    # model = SmallUnetWithEmb( img_channels=1).to(device)
    # model.load_state_dict(
    #     T.load("/teamspace/studios/this_studio/runs/models/version5/epoch_495.pt" , map_location=device)
    # )

    # model = UNet2DModel(   # Version 3
    #     in_channels=3,
    #     out_channels=3,
    #     block_out_channels=(32, 64, 128, 512),
    #     norm_num_groups=8,
    # ).to(device)
    model = UNet2DModel(
        in_channels=1,
        out_channels=1,
        block_out_channels=(16, 32, 64, 64),
        norm_num_groups=8,
    ).to(device)
    is_hg = True
    print(is_hg , type(model) )

    # print(model)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    diff = Diffusion(device=device, is_hg=is_hg , img_channels=1)

    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion = F.mse_loss

    train_loader , val_loader = get_loaders("fashion" , 64 , 32)

    # version 0 -> no attention
    # version 1 -> self attention added (using nn.MultiheadAttention)
    # version 2 -> Changed model architecture and used Silu instead of ReLU
    # version 3 -> Used hugging face diffusers.UNet2DModel  UNet2DModel( in_channels= 3 , out_channels = 3, block_out_channels = (32,64 , 128 , 512) , norm_num_groups=8).to(device)
    # version 4 -> Normal model in Version 3 .. changed dataset to Lsun bridges
    # version 5 -> Added EMA
    # Version 6 -> used fashion mnist 32*32 dataset
    # version 7 -> used fashion mnist 32*32 dataset with the diffusers UNet2DModel ..lr -> 2e-3

    version = 7

    train(
        model=model,
        diff=diff,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device= device,
        apply_ema=False,
        epochs=1001,
        version=version,
        start_epoch=0,
        hg_model=is_hg,
        save_every_n=5,
    )
