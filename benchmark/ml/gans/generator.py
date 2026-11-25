# gans/generator.py
import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _get_activation(name):
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Choose from {list(_ACTIVATIONS)}.")
    return _ACTIVATIONS[name]()


class ConditionalGenerator(nn.Module):
    """
    Conditional generator for a (WGAN-GP) setup.

    Maps a latent vector z and a class label into a feature vector x_hat.
    Mirrors the modular design of the CVAE decoder: architecture is fully
    configurable through the builder function `build_conditional_generator`.

    Conditioning:
      - use_label_embedding=True  -> integer class labels via nn.Embedding
      - use_label_embedding=False -> label vector concatenated directly
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_layers=(128, 256),
        activation="relu",
        output_activation=None,
        use_label_embedding: bool = True,
        n_classes: int = 2,
        label_dim: int = 1,
        label_embed_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_label_embedding = use_label_embedding
        self.n_classes = n_classes
        self.label_dim = label_dim
        self.label_embed_dim = label_embed_dim

        if use_label_embedding:
            self.label_emb = nn.Embedding(n_classes, label_embed_dim)
            cond_dim = label_embed_dim
        else:
            self.label_emb = None
            cond_dim = label_dim

        layers = []
        in_features = latent_dim + cond_dim
        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))
            layers.append(_get_activation(activation))
            in_features = units
        layers.append(nn.Linear(in_features, input_dim))
        if output_activation is not None:
            layers.append(_get_activation(output_activation))

        self.model = nn.Sequential(*layers)

    def _cond_features(self, labels):
        if self.use_label_embedding:
            return self.label_emb(labels.long().view(-1))
        return labels.float().view(labels.size(0), -1)

    def forward(self, z, labels):
        c = self._cond_features(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x)


def build_conditional_generator(
    input_dim: int,
    latent_dim: int,
    hidden_layers=(128, 256),
    activation="relu",
    output_activation=None,
    use_label_embedding: bool = True,
    n_classes: int = 2,
    label_dim: int = 1,
    label_embed_dim: int = 32,
):
    """Factory for a ConditionalGenerator (mirrors build_conditional_decoder)."""
    return ConditionalGenerator(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        output_activation=output_activation,
        use_label_embedding=use_label_embedding,
        n_classes=n_classes,
        label_dim=label_dim,
        label_embed_dim=label_embed_dim,
    )
