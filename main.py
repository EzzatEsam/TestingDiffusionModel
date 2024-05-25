from DiffusionTest import train
import torch as T
import torch.optim as optim
from DiffusionTest.model import SmallUnetWithEmb
from DiffusionTest.diffusion import Diffusion
import torch.nn.functional as F

if __name__ == "__main__":
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # import logging
    # logging.basicConfig( level=logging.INFO  , format="%(asctime)s - %(levelname)s - %(message)s")

    epochs = 50

    model = SmallUnetWithEmb().to(device)
    diff = Diffusion(device=device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = F.mse_loss
    
    
    # version 0 -> no attention
    # version 1 -> self attention added (using nn.MultiheadAttention)
    
    version = 1

    train(epochs, model, diff, optimizer, criterion, device, version=version)
