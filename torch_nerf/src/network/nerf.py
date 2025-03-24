"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()

        # TODO
        # raise NotImplementedError("Task 1")

        dims = [pos_dim] + [feat_dim]*9 + [128] + [3]
        self.num_layers = 11
        self.no_relu = 8
        self.pos_concat = 5
        self.view_dir_concat = 9
        
        for layer in range(0, self.num_layers):
            if layer == self.pos_concat:
                in_dim = dims[layer] + pos_dim
                out_dim = dims[layer+1]   
                setattr(self, "lin" + str(layer), nn.Linear(in_dim, out_dim))
            elif layer == self.view_dir_concat:
                in_dim = dims[layer] + view_dir_dim
                out_dim = dims[layer+1]   
                setattr(self, "lin" + str(layer), nn.Linear(in_dim, out_dim))
            elif layer == self.no_relu:
                out_dim = dims[layer+1] + 1 # 257
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))
            else:
                out_dim = dims[layer+1]   
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

        # setattr(self, "out_sigma", nn.Linear(feat_dim, 1))

        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    @jaxtyped
    @typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """

        # TODO
        # raise NotImplementedError("Task 1")
        x = pos

        for layer in range(0, self.num_layers):
            lin = getattr(self, "lin"+str(layer))
            if layer == self.pos_concat:
                x = torch.cat([x, pos], 1)
            elif layer == self.view_dir_concat:
                x = torch.cat([x, view_dir], 1)

            x = lin(x)
            
            if layer == self.num_layers - 1:
                radiance = self.sigmoid(x)
            elif layer == self.no_relu:
                sigma = self.relu(x[:, 0]).unsqueeze(-1)
                x = x[:, 1:]
            else:
                x = self.relu(x)
        
        return (sigma, radiance)

