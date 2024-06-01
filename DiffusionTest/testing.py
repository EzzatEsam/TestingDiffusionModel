import torch as T
from .model import SmallUnetWithEmb

m = SmallUnetWithEmb()

print(m(T.rand(8, 3, 128, 128)  , T.rand(8, 256)).shape)