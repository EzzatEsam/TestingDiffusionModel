import torch as T


class Diffusion:

    def __init__(
        self,
        time_steps: int = 1000,
        start_beta: float = 1e-4,
        end_beta: float = 0.02,
        img_size: int = 128,
        img_channels: int = 3,
        embedding_dim: int = 256,
        device: T.device = T.device("cuda" if T.cuda.is_available() else "cpu"),
        is_hg: bool = False,
        n_classes: int | None = None,
    ):
        self.time_steps = time_steps
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.beta_schedule = T.linspace(start_beta, end_beta, time_steps).to(device)
        self.alpha_schedule = 1 - self.beta_schedule
        self.alpha_hat_schedule = T.cumprod(input=self.alpha_schedule, axis=0)
        self.is_hg = is_hg
        self.n_classes = n_classes
        self.img_channels = img_channels

    def add_noise(self, x: T.Tensor, t: T.Tensor) -> tuple[T.Tensor, T.Tensor]:

        noise = T.randn_like(x)
        alpha_hat_t = self.alpha_hat_schedule[t]
        noisy_tensor = (
            x * T.sqrt(alpha_hat_t)[:, None, None, None]
            + T.sqrt(1 - alpha_hat_t)[:, None, None, None] * noise
        )

        return noisy_tensor.to(self.device), noise.to(self.device)

    def remove_noise(self, img: T.Tensor, noise: T.Tensor, t: int) -> T.Tensor:
        """
        Remove noise from an image using the given parameters.
        Based on (Algorithm 2 Sampling) in https://arxiv.org/abs/2006.11239v2
        Args:

                img (T.Tensor): The image tensor from which to remove noise.
                noise (T.Tensor): The noise tensor to be subtracted from the image.
                t (int): The time step at which the noise removal is performed.

        Returns:
                T.Tensor: The image tensor with the noise removed.
        """

        alpha_hat_t = self.alpha_hat_schedule[t]
        alpha_t = self.alpha_schedule[t]
        beta_t = self.beta_schedule[t]
        if t == 0:
            extra_noise = T.zeros_like(img)
        else:
            extra_noise = T.randn_like(img)

        noise_removed = 1 / T.sqrt(alpha_t) * (
            img - ((1 - alpha_t) / T.sqrt(1 - alpha_hat_t)) * noise
        ) + extra_noise * T.sqrt(beta_t)

        return noise_removed.to(self.device)

    def time_encode(self, t: T.Tensor) -> T.Tensor:
        """
        Encodes the given batch of time values into a batch of time-encoded vectors.

        Parameters:
                t (Tensor): The batch of time values to be encoded, with shape (batch_size,).

        Returns:
                Tensor: The batch of time-encoded vectors, with shape (batch_size, time_dim).
        """

        inv_freq = 1.0 / (
            10000 ** (T.arange(0, self.embedding_dim, 2).float() / self.embedding_dim)
        ).to(t.device)

        enc_sin = T.sin(t[:, None] * inv_freq)
        enc_cos = T.cos(t[:, None] * inv_freq)
        return T.cat([enc_sin, enc_cos], dim=-1).to(self.device)

    def get_time_samples(self, batch_size=1) -> T.Tensor:
        return T.randint(low=0, high=self.time_steps, size=(batch_size,)).to("cpu")

    def generate_sample(self, model: T.nn.Module, n=10, n_images : int = 1 ,label: int | None = None):
        with T.no_grad():
            model.eval()
            img = T.randn(
                (n_images, self.img_channels, self.img_size, self.img_size)
            ).to(self.device)
            imgs = []
            for t in reversed(range(self.time_steps)):
                if not self.is_hg:
                    t_vec = self.time_encode(T.tensor([t] * n_images).to(self.device))
                    predicted_noise = model(img, t_vec)
                else:
                    predicted_noise = model(img, t).sample
                img = self.remove_noise(img, predicted_noise, t)

                every_n = self.time_steps // n

                if t % every_n == 0:
                    imgs.append(img)

        return imgs
