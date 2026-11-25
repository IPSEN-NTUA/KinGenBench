# gans/critic.py
import torch
import torch.nn as nn

from .generator import _get_activation


class ConditionalCritic(nn.Module):
    """
    Conditional critic (discriminator) for a WGAN-GP setup.

    Scores a feature vector x given a class label. No sigmoid on the output
    (Wasserstein critic outputs a real-valued score). Mirrors the modular
    design of the CVAE encoder.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers=(256, 128),
        activation="leaky_relu",
        use_label_embedding: bool = True,
        n_classes: int = 2,
        label_dim: int = 1,
        label_embed_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
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
        in_features = input_dim + cond_dim
        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))
            layers.append(_get_activation(activation))
            in_features = units
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def _cond_features(self, labels):
        if self.use_label_embedding:
            return self.label_emb(labels.long().view(-1))
        return labels.float().view(labels.size(0), -1)

    def forward(self, x, labels):
        c = self._cond_features(labels)
        x = torch.cat([x, c], dim=1)
        return self.model(x)


def build_conditional_critic(
    input_dim: int,
    hidden_layers=(256, 128),
    activation="leaky_relu",
    use_label_embedding: bool = True,
    n_classes: int = 2,
    label_dim: int = 1,
    label_embed_dim: int = 32,
):
    """Factory for a ConditionalCritic (mirrors build_conditional_encoder)."""
    return ConditionalCritic(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        activation=activation,
        use_label_embedding=use_label_embedding,
        n_classes=n_classes,
        label_dim=label_dim,
        label_embed_dim=label_embed_dim,
    )
