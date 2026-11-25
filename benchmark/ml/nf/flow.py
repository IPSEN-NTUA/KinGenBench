# nf/flow.py
import torch
import torch.nn as nn

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet


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


def build_conditional_realnvp(
    input_dim: int,
    context_dim: int = 1,
    num_layers: int = 10,
    hidden_features: int = 256,
    num_blocks: int = 2,
    activation: str = "relu",
    use_batch_norm: bool = False,
):
    """
    Build a class-conditional RealNVP normalizing flow (nflows).

    Alternating affine coupling transforms, each followed by a reverse
    permutation, over a StandardNormal base. The coupling networks receive
    the label as a `context` vector, making the flow conditional.

    Mirrors the modular builder style of the CVAE encoder/decoder: the whole
    architecture is specified through this single function.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the data (theta_dim, e.g. number of parameters).
    context_dim : int
        Dimensionality of the conditioning label.
    num_layers : int
        Number of coupling+permutation blocks.
    hidden_features : int
        Width of each ResidualNet coupling network.
    num_blocks : int
        Residual blocks per coupling network.
    activation : str
        Activation name for the coupling networks.
    use_batch_norm : bool
        Batch norm inside the coupling networks.

    Returns
    -------
    nflows.flows.base.Flow
    """
    transforms = []
    mask = (torch.arange(input_dim) % 2).float()

    def create_net(in_features, out_features):
        return ResidualNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_dim,
            num_blocks=num_blocks,
            activation=_get_activation(activation),
            use_batch_norm=use_batch_norm,
        )

    for i in range(num_layers):
        current_mask = mask if i % 2 == 0 else 1 - mask
        transforms.append(
            AffineCouplingTransform(
                mask=current_mask,
                transform_net_create_fn=create_net,
            )
        )
        transforms.append(ReversePermutation(input_dim))

    transform = CompositeTransform(transforms)
    base = StandardNormal([input_dim])

    return Flow(transform, base)
