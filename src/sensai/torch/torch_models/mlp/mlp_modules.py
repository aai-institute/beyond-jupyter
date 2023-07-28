from typing import Callable, Optional, Sequence

import torch
from torch import nn

from ...torch_base import MCDropoutCapableNNModule
from ....util.string import object_repr, function_name


class MultiLayerPerceptron(MCDropoutCapableNNModule):
    def __init__(self, input_dim: float, output_dim: float, hidden_dims: Sequence[int],
            hid_activation_fn: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            output_activation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.sigmoid,
            p_dropout: Optional[float] = None):
        super().__init__()
        self.inputDim = input_dim
        self.outputDim = output_dim
        self.hiddenDims = hidden_dims
        self.hidActivationFn = hid_activation_fn
        self.outputActivationFn = output_activation_fn
        self.pDropout = p_dropout
        self.layers = nn.ModuleList()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p=p_dropout)
        else:
            self.dropout = None
        prev_dim = input_dim
        for dim in [*hidden_dims, output_dim]:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim

    def __str__(self):
        return object_repr(self, dict(inputDim=self.inputDim, outputDim=self.outputDim, hiddenDims=self.hiddenDims,
            hidActivationFn=function_name(self.hidActivationFn) if self.hidActivationFn is not None else None,
            outputActivationFn=function_name(self.outputActivationFn) if self.outputActivationFn is not None else None,
            pDropout=self.pDropout))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            is_last = i+1 == len(self.layers)
            x = layer(x)
            if not is_last and self.dropout is not None:
                x = self.dropout(x)
            activation = self.hidActivationFn if not is_last else self.outputActivationFn
            if activation is not None:
                x = activation(x)
        return x
