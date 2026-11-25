# nf/nf_model.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class ConditionalRealNVP:
    """
    Training / sampling wrapper around a conditional RealNVP flow.

    Mirrors the role of `ConditionalVAE` (CVAE) and `ConditionalWGANGP` (GAN):
    a notebook builds the flow with `build_conditional_realnvp` and then calls
    `.fit(...)` and `.generate(...)` here. The flow is trained by maximum
    likelihood (minimizing negative log-likelihood of the data under the
    label-conditioned distribution).

    Parameters
    ----------
    flow : nflows.flows.base.Flow
        Built via `build_conditional_realnvp`.
    lr : float
        AdamW learning rate.
    grad_clip : float
        Max grad norm for clipping (0/None disables).
    weight_decay : float
        AdamW weight decay.
    device : str or torch.device
    """

    def __init__(self, flow, lr=1e-3, grad_clip=5.0, weight_decay=0.0,
                 device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.flow = flow.to(self.device)
        self.opt = torch.optim.AdamW(
            self.flow.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.grad_clip = grad_clip
        self.history = {"nll": []}

    def _make_loader(self, X, y, batch_size, shuffle=True):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        if y.ndim == 1:
            y = y.view(-1, 1)          # (n,) -> (n, 1); (n, k) kept as-is
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=True)

    def fit(self, X, y, epochs=300, batch_size=512, log_every=10, verbose=True):
        """
        Train by maximum likelihood.

        X : array (n_samples, input_dim)   scaled features
        y : array (n_samples,) or (n, 1)   conditioning label
        Returns the history dict.
        """
        loader = self._make_loader(X, y, batch_size)
        self.flow.train()

        for epoch in range(1, epochs + 1):
            total_nll, total = 0.0, 0
            for th, ctx in loader:
                th = th.to(self.device)
                ctx = ctx.to(self.device)

                logp = self.flow.log_prob(th, context=ctx)
                loss = -logp.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.flow.parameters(), self.grad_clip
                    )
                self.opt.step()

                total_nll += loss.item() * th.size(0)
                total += th.size(0)

            mean_nll = total_nll / total
            self.history["nll"].append(mean_nll)
            if verbose and (epoch % log_every == 0 or epoch == 1
                            or epoch == epochs):
                print(f"Epoch {epoch:3d} | NLL {mean_nll:.4f}")

        return self.history

    @torch.no_grad()
    def generate(self, n_samples, context=None, label=1, temperature=1.0):
        """
        Sample from the flow by inverting base noise z ~ N(0, temperature^2 I),
        conditioned on a label / KPI vector.

        Parameters
        ----------
        context : array-like, optional
            Continuous conditioning vector(s) in the model's (scaled) label
            space, of shape (context_dim,) or (n_samples, context_dim). A single
            row is broadcast to all samples. Use this for continuous / multi-KPI
            conditioning.
        label : float
            Fallback used only when `context is None`: a single scalar broadcast
            over one context dimension (discrete-label behaviour).

        Returns a numpy array (n_samples, input_dim) in the *scaled* space.
        """
        self.flow.eval()
        input_dim = self.flow._distribution._shape[0]
        z = temperature * torch.randn(n_samples, input_dim, device=self.device)

        if context is None:
            ctx = torch.full((n_samples, 1), float(label),
                             dtype=torch.float32, device=self.device)
        else:
            context = np.atleast_2d(np.asarray(context, dtype=np.float32))
            if context.shape[0] == 1:
                context = np.repeat(context, n_samples, axis=0)
            if context.shape[0] != n_samples:
                raise ValueError(
                    f"context has {context.shape[0]} rows but "
                    f"n_samples={n_samples}"
                )
            ctx = torch.as_tensor(context, dtype=torch.float32,
                                  device=self.device)

        samples, _ = self.flow._transform.inverse(z, context=ctx)
        return samples.cpu().numpy()
