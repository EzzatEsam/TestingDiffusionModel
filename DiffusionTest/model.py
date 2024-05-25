import torch as T
import torch.nn as nn

print_versions = False


class SmallUnetWithEmb(nn.Module):
    def __init__(self, img_channels=3, embedding_dim=256, device="cuda"):
        super(SmallUnetWithEmb, self).__init__()
        self.device = device
        self.encoder = Encoder(img_channels, embedding_dim)
        self.decoder = Decoder(1024, embedding_dim)
        self.final = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.embedding_dim = embedding_dim
        self.bottleneck =  DoubleConv(512, 512, embedding_dim , has_attention=True)
    def forward(self, x: T.Tensor, embeddings: T.tensor):
        x = self.encoder(x, embeddings)
        x1 , x2, x3, x4 , x5 = x
        
        x5 = self.bottleneck(x5 , embeddings)
        x = [x1, x2, x3, x4, x5]
        x = self.decoder(*x, embeddings)
        x = self.final(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.mha = nn.MultiheadAttention(in_channels, heads, batch_first=True)
        self.ln = nn.LayerNorm([in_channels])

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(0, 2, 1)
        x = self.ln(x)
        attn_out = self.mha.forward(x, x, x, need_weights=False)
        x, _ = attn_out
        return x.permute(0, 2, 1).reshape(*orig_shape)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embedding_dim=256,
        has_attention=False,
        residual=True,
    ):
        super(DoubleConv, self).__init__()
        self.has_attention = has_attention
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),   
        )
        
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.embed = nn.Sequential(nn.SELU(), nn.Linear(embedding_dim, out_channels))
        if has_attention:
            self.attn = SelfAttention(out_channels)
        if residual:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: T.Tensor, time_vec: T.Tensor):
        # print( "Double conv input" , x.shape)
        x_old = x
        x = self.conv1(x)
        t = self.embed(time_vec)
        t = t[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        x = x + t
        
        x = self.conv2(x)
        if self.residual:
            x = x + self.residual(x_old)
        if self.has_attention:
            x = x + self.attn(x)
        return x


class DownModule(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, has_attention=False):
        super(DownModule, self).__init__()
        self.conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_dim=embedding_dim,
            has_attention=has_attention,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, time_vec):
        x1 = self.conv.forward(x, time_vec)
        x = self.pool(x1)
        return x , x1


class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, has_attention=False):
        super(UpModule, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(
            in_channels, out_channels, embedding_dim, has_attention=has_attention
        )

    def forward(self, x: T.Tensor, x_skip: T.Tensor, time_vec: T.Tensor):
        # print(x.shape, x_skip.shape)
        x = self.up(x)
        x = T.cat((x, x_skip), dim=1)
        x = self.conv.forward(x, time_vec)
        # print("output shape" , x.shape)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, embedding_dim):
        super(Encoder, self).__init__()

        self.down1 = DownModule(in_channels, 64, embedding_dim)
        self.down2 = DownModule(64, 128, embedding_dim)
        self.down3 = DownModule(128, 256, embedding_dim)
        self.down4 = DownModule(256, 512, embedding_dim , has_attention=True)
        self.down5 = DownModule(512, 1024, embedding_dim , has_attention=True)

    def forward(self, x, time_vec):
        x , x1 = self.down1(x, time_vec)  # x1 has shape (batch_size, 64, 256, 256)
        x , x2 = self.down2(x, time_vec)  # x2 has shape (batch_size, 128, 128, 128)
        x , x3 = self.down3(x, time_vec)  # x3 has shape (batch_size, 256, 64, 64)
        x5 ,x4 = self.down4(x, time_vec)  # x4 has shape (batch_size, 512, 32, 32)
        return x1, x2, x3, x4 , x5


class Decoder(nn.Module):
    def __init__(self, in_channels, time_dim):
        super(Decoder, self).__init__()
        self.up1 = UpModule(in_channels, 256, time_dim , has_attention=True)
        self.up2 = UpModule(512, 128, time_dim)
        self.up3 = UpModule(256, 64, time_dim)
        self.up4 = UpModule(128, 64, time_dim)

    def forward(self, x1, x2, x3, x4, x5, time_vec):
        x = self.up1(x5, x4, time_vec)
        x = self.up2(x, x3, time_vec)
        x = self.up3(x, x2, time_vec)
        x = self.up4(x, x1, time_vec)
        return x
