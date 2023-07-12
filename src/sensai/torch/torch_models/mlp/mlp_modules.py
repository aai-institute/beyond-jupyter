from typing import Callable, Optional, Sequence

import torch
from torch import nn

from ...torch_base import MCDropoutCapableNNModule
from ....util.string import objectRepr, functionName


class MultiLayerPerceptron(MCDropoutCapableNNModule):
    def __init__(self, inputDim: float, outputDim: float, hiddenDims: Sequence[int],
            hidActivationFn: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
            outputActivationFn: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.sigmoid,
            pDropout: Optional[float] = None):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.hiddenDims = hiddenDims
        self.hidActivationFn = hidActivationFn
        self.outputActivationFn = outputActivationFn
        self.pDropout = pDropout
        self.layers = nn.ModuleList()
        if pDropout is not None:
            self.dropout = nn.Dropout(p=pDropout)
        else:
            self.dropout = None
        prevDim = inputDim
        for dim in [*hiddenDims, outputDim]:
            self.layers.append(nn.Linear(prevDim, dim))
            prevDim = dim

    def __str__(self):
        return objectRepr(self, dict(inputDim=self.inputDim, outputDim=self.outputDim, hiddenDims=self.hiddenDims,
            hidActivationFn=functionName(self.hidActivationFn) if self.hidActivationFn is not None else None,
            outputActivationFn=functionName(self.outputActivationFn) if self.outputActivationFn is not None else None,
            pDropout=self.pDropout))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            isLast = i+1 == len(self.layers)
            x = layer(x)
            if not isLast and self.dropout is not None:
                x = self.dropout(x)
            activation = self.hidActivationFn if not isLast else self.outputActivationFn
            if activation is not None:
                x = activation(x)
        return x
