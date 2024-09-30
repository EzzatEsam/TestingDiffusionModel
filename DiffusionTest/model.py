import torch as T
import torch.nn as nn

print_versions = False


class SmallUnetWithEmb(nn.Module):
    def __init__(
        self,
        img_channels=3,
        embedding_dim=256,
        n_classes: int | None = None,
        device="cuda",
    ):
        super(SmallUnetWithEmb, self).__init__()
        self.device = device
        self.encoder = Encoder(
            img_channels,
            embedding_dim,
            use_attention_layers=[False, False, True, True],
            channels_list=[64, 128, 256, 512],
        )
        self.decoder = Decoder(
            512,
            embedding_dim,
            use_attention_layers=[False, False, True, True],
            channels_list=[256, 128, 64, 64],
        )
        self.final = nn.Conv2d(64, img_channels, kernel_size=3, padding=1)
        self.embedding_dim = embedding_dim
        self.bottleneck = DoubleConv(512, 512, embedding_dim, has_attention=True)
        self.n_classes = n_classes
        if n_classes:
            self.cls_embedding = nn.Embedding(n_classes, embedding_dim)

    def forward(self, x: T.Tensor, embeddings: T.tensor, y: T.Tensor | None = None):

        if y and self.n_classes:
            cls_embedding = self.cls_embedding(y)
            embeddings = embeddings + cls_embedding
        x = self.encoder(x, embeddings)

        x[-1] = self.bottleneck(x[-1], embeddings)
        x = self.decoder(x, time_vec=embeddings)
        x = self.final(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(SelfAttention, self).__init__()
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
        self.embed = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, out_channels))
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

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x, time_vec):
        x1 = self.conv.forward(x, time_vec)
        x = self.pool(x1)
        return x, x1


class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim, has_attention=False):
        super(UpModule, self).__init__()
        # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(
            in_channels * 2, out_channels, embedding_dim, has_attention=has_attention
        )

    def forward(self, x: T.Tensor, x_skip: T.Tensor, time_vec: T.Tensor):
        x = self.up(x)
        x = T.cat((x, x_skip), dim=1)
        x = self.conv.forward(x, time_vec)
        return x


class Encoder(nn.Module):
    """
    An encoder module that downsamples feature maps while optionally applying attention.

    This class consists of multiple downsampling layers, each configured to take
    input from the previous layer and produce skip connections for the decoder.
    The number of output channels and the application of attention can be customized
    through the constructor parameters.
    """

    def __init__(
        self,
        in_channels,
        embedding_dim,
        channels_list=[64, 128, 256, 512],
        use_attention_layers=[False, False, False, True],
    ):
        """
        Initializes the Encoder with the specified input channels and configuration.

        Args:
            in_channels (int): The number of input channels for the first downsampling layer.
            embedding_dim (int): The dimension of the embedding used in the downsampling modules.
            channels_list (list[int], optional): A list of integers specifying the output
                channels for each downsampling layer. Defaults to [64, 128, 256, 512].
            use_attention_layers (list[bool], optional): A list of booleans indicating
                whether to apply attention in each downsampling layer. Defaults to
                [False, False, False, True].

        Raises:
            AssertionError: If the length of channels_list and use_attention_layers
                do not match.
        """
        super(Encoder, self).__init__()

        # Ensure the length of channels_list matches the length of use_attention_layers
        assert len(channels_list) == len(
            use_attention_layers
        ), "channels_list and use_attention_layers must have the same length."

        # Create downsampling modules dynamically based on the channels_list and use_attention_layers
        self.down_modules = nn.ModuleList()
        prev_channels = in_channels
        for i in range(len(channels_list)):
            out_channels = channels_list[i]
            has_attention = use_attention_layers[i]
            self.down_modules.append(
                DownModule(
                    prev_channels,
                    out_channels,
                    embedding_dim,
                    has_attention=has_attention,
                )
            )
            prev_channels = out_channels

    def forward(self, x: T.Tensor, time_vec: T.Tensor):
        """
        Performs the forward pass through the encoder.

        Takes an input tensor and a time vector, processes them through the
        defined downsampling modules, and returns a list of skip connections
        for use in the decoder.

        Args:
            x (T.Tensor): The input tensor to be encoded.
            time_vec (T.Tensor): A tensor representing the time vector for each
                downsampling operation.

        Returns:
            list[T.Tensor]: A list of tensors representing the skip connections
            from each downsampling layer, with the final output of the encoder
            as the last element.
        """
        skip_connections = []

        for i, down_module in enumerate(
            self.down_modules
        ):  # Loop through all down modules except the last one
            x, skip_connection = down_module(x, time_vec)
            skip_connections.append(skip_connection)

        skip_connections.append(x)
        return skip_connections


class Decoder(nn.Module):
    """
    A decoder module that upsamples feature maps while optionally applying attention.

    This class consists of multiple upsampling layers, each configured to take
    input from the previous layer and a corresponding skip connection from an
    encoder. The number of output channels and the application of attention can
    be customized through the constructor parameters.
    """

    def __init__(
        self,
        in_channels: int,
        time_dim: int,
        channels_list: list[int] = [256, 128, 64, 64],
        use_attention_layers: list[bool] = [True, False, False, False],
    ):  # Default attention configuration
        """
        Initializes the Decoder with the specified input channels and configuration.

        Args:
            in_channels (int): The number of input channels for the first upsampling layer.
            time_dim (int): The dimension of the time vector used in the upsampling modules.
            channels_list (list[int], optional): A list of integers specifying the output
                channels for each upsampling layer. Defaults to [256, 128, 64, 64].
            use_attention_layers (list[bool], optional): A list of booleans indicating
                whether to apply attention in each upsampling layer. Defaults to
                [True, False, False, False].

        Raises:
            AssertionError: If the length of channels_list and use_attention_layers
                do not match.
        """
        super(Decoder, self).__init__()

        # Ensure channels_list and use_attention_layers are of equal length
        assert len(channels_list) == len(
            use_attention_layers
        ), "channels_list and use_attention_layers must have the same length."

        # Create upsampling modules dynamically based on the channels_list and use_attention_layers
        self.up_modules = nn.ModuleList()
        prev_channels = in_channels
        for i in range(len(channels_list)):
            out_channels = channels_list[i]
            has_attention = use_attention_layers[i]
            self.up_modules.append(
                UpModule(
                    prev_channels, out_channels, time_dim, has_attention=has_attention
                )
            )
            prev_channels = out_channels

    def forward(self, x_list: list[T.Tensor], time_vec: T.Tensor) -> T.Tensor:
        """
        Performs the forward pass through the decoder.

        Takes a list of tensors representing skip connections from the encoder
        and a time vector, and returns the upsampled feature map after
        processing through the defined upsampling modules.

        Args:
            x_list (list[T.Tensor]): A list of tensors containing the skip connections
                from the encoder, where the last tensor is the output of the previous
                decoder layer.
            time_vec (T.Tensor): A tensor representing the time vector for each
                upsampling operation.

        Returns:
            T.Tensor: The final upsampled feature map after processing through
            all upsampling modules.
        """
        x_last = x_list[-1]
        for i, up_module in enumerate(self.up_modules):  # 0 1 2 3
            x = up_module(x_last, x_list[-i - 2], time_vec)  # -2 -3 -4 -5
            x_last = x
        return x

        # x1 , x2, x3, x4, x5 = x
        # x = self.up_modules[0](x5, x4, time_vec)  # First upsampling layer
        # x = self.up_modules[1](x, x3, time_vec)   # Second upsampling layer
        # x = self.up_modules[2](x, x2, time_vec)   # Third upsampling layer
        # x = self.up_modules[3](x, x1, time_vec)   # Fourth upsampling layer
        # return x
