# gans/gan_model.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def gradient_penalty(critic, real, fake, labels, device):
    """
    WGAN-GP gradient penalty, computed on random interpolations between
    real and fake samples.
    """
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real)

    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated, labels)

    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


class ConditionalWGANGP:
    """
    Conditional Wasserstein GAN with gradient penalty.

    A thin, modular training wrapper around a generator + critic, mirroring
    the role of `ConditionalVAE` for the CVAE. It owns the optimizers, the
    training loop and sample generation, so a notebook only needs to build
    the two networks and call `.fit(...)`.

    Parameters
    ----------
    generator, critic : nn.Module
        Built via `build_conditional_generator` / `build_conditional_critic`.
    latent_dim : int
        Dimension of the latent noise vector.
    lr : float
        Learning rate for both Adam optimizers.
    n_critic : int
        Critic updates per generator update.
    lambda_gp : float
        Gradient-penalty weight.
    betas : tuple
        Adam betas (WGAN-GP default is (0.0, 0.9)).
    device : str or torch.device
    """

    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        latent_dim: int,
        lr: float = 5e-4,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        betas=(0.0, 0.9),
        device=None,
        continuous: bool = False,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.generator = generator.to(self.device)
        self.critic = critic.to(self.device)
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        # continuous=True -> conditioning label is a real-valued vector
        # (concatenated), e.g. a KPI target; False -> integer class + embedding.
        self.continuous = continuous

        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=betas
        )

        self.history = {"loss_gen": [], "loss_critic": [], "gp": []}

    def _make_loader(self, X, y, batch_size, shuffle=True):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if self.continuous:
            y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
            if y.ndim == 1:
                y = y.view(-1, 1)          # (n,) -> (n, 1); (n, k) kept as-is
        else:
            y = torch.as_tensor(np.asarray(y)).view(-1).long()
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=True)

    def fit(self, X, y, epochs=200, batch_size=100, log_every=10, verbose=True):
        """
        Train the WGAN-GP.

        X : array (n_samples, input_dim)   scaled features
        y : array (n_samples,)             integer class labels
        Returns the history dict.
        """
        loader = self._make_loader(X, y, batch_size)

        for epoch in range(epochs):
            gen_losses, critic_losses, gp_losses = [], [], []

            for real, labels in loader:
                real = real.to(self.device)
                labels = labels.to(self.device)
                bs = real.size(0)

                # ----- Train critic -----
                for _ in range(self.n_critic):
                    z = torch.randn(bs, self.latent_dim, device=self.device)
                    fake = self.generator(z, labels)

                    critic_real = self.critic(real, labels).reshape(-1)
                    critic_fake = self.critic(fake.detach(), labels).reshape(-1)
                    gp = gradient_penalty(
                        self.critic, real, fake.detach(), labels, self.device
                    )
                    loss_critic = -(critic_real.mean() - critic_fake.mean()) \
                        + self.lambda_gp * gp

                    self.opt_critic.zero_grad()
                    loss_critic.backward()
                    self.opt_critic.step()

                    critic_losses.append(loss_critic.item())
                    gp_losses.append(gp.item())

                # ----- Train generator -----
                z = torch.randn(bs, self.latent_dim, device=self.device)
                fake = self.generator(z, labels)
                loss_gen = -self.critic(fake, labels).reshape(-1).mean()

                self.opt_gen.zero_grad()
                loss_gen.backward()
                self.opt_gen.step()
                gen_losses.append(loss_gen.item())

            mean_gen = float(np.mean(gen_losses))
            mean_critic = float(np.mean(critic_losses))
            mean_gp = float(np.mean(gp_losses))
            self.history["loss_gen"].append(mean_gen)
            self.history["loss_critic"].append(mean_critic)
            self.history["gp"].append(mean_gp)

            if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d} | loss_gen: {mean_gen:+.4f} "
                      f"| loss_critic: {mean_critic:+.4f} | gp: {mean_gp:.4f}")

        return self.history

    @torch.no_grad()
    def generate(self, n_samples, label=1, temperature=1.0, context=None):
        """
        Sample `n_samples` from the generator.

        context : array-like, optional
            Continuous conditioning vector(s) of shape (label_dim,) or
            (n_samples, label_dim). Required when the model was built with
            `continuous=True`; a single row is broadcast to all samples.
        label : int
            Integer class used for the discrete (embedding) case when
            `context is None`.
        Returns a numpy array (n_samples, input_dim) in the *scaled* space.
        """
        self.generator.eval()
        z = temperature * torch.randn(
            n_samples, self.latent_dim, device=self.device
        )
        if self.continuous or context is not None:
            context = np.atleast_2d(np.asarray(context, dtype=np.float32))
            if context.shape[0] == 1:
                context = np.repeat(context, n_samples, axis=0)
            if context.shape[0] != n_samples:
                raise ValueError(
                    f"context has {context.shape[0]} rows but n_samples={n_samples}"
                )
            labels = torch.as_tensor(context, dtype=torch.float32, device=self.device)
        else:
            labels = torch.full((n_samples,), int(label),
                                dtype=torch.long, device=self.device)
        samples = self.generator(z, labels).cpu().numpy()
        return samples
