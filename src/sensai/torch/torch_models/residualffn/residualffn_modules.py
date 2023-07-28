from abc import ABC, abstractmethod
from typing import Sequence, Optional

import torch
from torch import nn


class ResidualFeedForwardNetwork(nn.Module):
    """
    A feed-forward network consisting of a fully connected input layer, a configurable number of residual blocks and a
    fully connected output layer. Similar architecture are described in e.g. [1] and [2] and are all inspired by
    ResNet [3]. Each residual block consists of two fully connected layers with (optionally) batch normalisation and
    dropout, which can all be bypassed by a so-called skip connection. The skip path and the non-skip path are added as
    the last step within each block.

    More precisely, the non-skip path consists of the following layers:
    batch normalization -> ReLU, dropout -> fully-connected -> batch normalization -> ReLU, dropout -> fully-connected
    The use of the activation function before the connected layers is called "pre-activation" [4].

    The skip path does nothing for the case where the input dimension of the block equals the output dimension. If these
    dimensions are different, the skip-path consists of a fully-connected layer, but with no activation, normalization,
    or dropout.

    Within each block, the dimension can be reduced by a certain factor. This is known as "bottleneck" design. It has
    been shown for the original ResNet, that such a bottleneck design can reduce the number of parameters of the models
    and improve the training behaviour without compromising the results.

    Batch normalisation can be deactivated, but normally it improves the results, since it not only provides some
    regularisation, but also normalises the distribution of the inputs of each layer and therefore addresses the problem
    of "internal covariate shift"[5]. The mechanism behind this is not yet fully understood (see e.g. the Wikipedia
    article on batch normalisation for further references).
    Our batch normalisation module will normalise batches per dimension C in 2D tensors of shape (N, C) or 3D tensors
    of shape (N, L, C).

    References:

      * [1] Chen, Dongwei et al. "Deep Residual Learning for Nonlinear Regression."
        Entropy 22, no. 2 (February 2020): 193. https://doi.org/10.3390/e22020193.
      * [2] Kiprijanovska, et al. "HousEEC: Day-Ahead Household Electrical Energy Consumption Forecasting Using Deep Learning."
        Energies 13, no. 10 (January 2020): 2672. https://doi.org/10.3390/en13102672.
      * [3] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition."
        ArXiv:1512.03385 [Cs], December 10, 2015. http://arxiv.org/abs/1512.03385.
      * [4] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Identity Mappings in Deep Residual Networks."
        ArXiv:1603.05027 [Cs], July 25, 2016. http://arxiv.org/abs/1603.05027.
      * [5] Ioffe, Sergey, and Christian Szegedy. "Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift." ArXiv:1502.03167 [Cs], March 2, 2015. http://arxiv.org/abs/1502.03167.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int], bottleneck_dimension_factor: float = 1,
            p_dropout: Optional[float] = None, use_batch_normalisation: bool = True) -> None:
        """
        :param input_dim: the input dimension of the model
        :param output_dim: the output dimension of the model
        :param hidden_dims: a list of dimensions; for each list item, a residual block with the corresponding dimension
                is created
        :param bottleneck_dimension_factor: an optional factor that specifies the hidden dimension within each block
        :param p_dropout: the dropout probability to use during training (defaults to None for no dropout)
        :param use_batch_normalisation: whether to use batch normalisation (defaults to True)
        """
        super().__init__()
        self.inputDim = input_dim
        self.outputDim = output_dim
        self.hiddenDims = hidden_dims
        self.useBatchNormalisation = use_batch_normalisation

        if p_dropout is not None:
            self.dropout = nn.Dropout(p=p_dropout)
        else:
            self.dropout = None

        self.inputLayer = nn.Linear(self.inputDim, self.hiddenDims[0])

        inner_hidden_dims = lambda x: max(1, round(x * bottleneck_dimension_factor))
        blocks = []
        prev_dim = self.hiddenDims[0]
        for hidden_dim in self.hiddenDims[1:]:
            if hidden_dim == prev_dim:
                block = self._IdentityBlock(hidden_dim, inner_hidden_dims(hidden_dim), self.dropout, use_batch_normalisation)
            else:
                block = self._DenseBlock(prev_dim, inner_hidden_dims(hidden_dim), hidden_dim, self.dropout, use_batch_normalisation)
            blocks.append(block)
            prev_dim = hidden_dim

        self.bnOutput = self._BatchNorm(self.hiddenDims[-1]) if self.useBatchNormalisation else None
        self.outputLayer = nn.Linear(self.hiddenDims[-1], output_dim)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):

        x = self.inputLayer(x)

        for block in self.blocks:
            x = block(x)

        x = self.bnOutput(x) if self.useBatchNormalisation else x
        x = self.dropout(x) if self.dropout is not None else x
        x = self.outputLayer(x)
        return x

    class _BatchNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.bn = nn.BatchNorm1d(dim)

        def forward(self, x):
            # BatchNorm1D normalises a 3D tensor per dimension C for shape (N, C, SeqLen).
            # For a 3D tensor (N, SeqLen, C), we thus permute to obtain (N, C, SeqLen), adopting the "regular" broadcasting semantics.
            is_3d = len(x.shape) == 3
            if is_3d:
                x = x.permute((0, 2, 1))
            x = self.bn(x)
            if is_3d:
                x = x.permute((0, 2, 1))
            return x

    class _ResidualBlock(nn.Module, ABC):
        """
        A generic residual block which need to be specified by defining the skip path.
        """

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: Optional[nn.Dropout],
                use_batch_normalisation: bool) -> None:
            super().__init__()
            self.inputDim = input_dim
            self.hiddenDim = hidden_dim
            self.outputDim = output_dim
            self.dropout = dropout
            self.useBatchNormalisation = use_batch_normalisation
            self.bnIn = ResidualFeedForwardNetwork._BatchNorm(self.inputDim) if use_batch_normalisation else None
            self.denseIn = nn.Linear(self.inputDim, self.hiddenDim)
            self.bnOut = ResidualFeedForwardNetwork._BatchNorm(self.hiddenDim) if use_batch_normalisation else None
            self.denseOut = nn.Linear(self.hiddenDim, self.outputDim)

        def forward(self, x):
            x_skipped = self._skip(x)

            x = self.bnIn(x) if self.useBatchNormalisation else x
            x = torch.relu(x)
            x = self.dropout(x) if self.dropout is not None else x
            x = self.denseIn(x)

            x = self.bnOut(x) if self.useBatchNormalisation else x
            x = torch.relu(x)
            x = self.dropout(x) if self.dropout is not None else x
            x = self.denseOut(x)

            return x + x_skipped

        @abstractmethod
        def _skip(self, x):
            """
            Defines the skip path of the residual block. The input is identical to the argument passed to forward.
            """
            pass

    class _IdentityBlock(_ResidualBlock):
        """
        A residual block preserving the dimension of the input
        """

        def __init__(self, input_output_dim: int, hidden_dim: int, dropout: Optional[nn.Dropout], use_batch_normalisation: bool) -> None:
            super().__init__(input_output_dim, hidden_dim, input_output_dim, dropout, use_batch_normalisation)

        def _skip(self, x):
            """
            Defines the skip path as the identity function.
            """
            return x

    class _DenseBlock(_ResidualBlock):
        """
        A residual block changing the dimension of the input to the given value.
        """

        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: Optional[nn.Dropout],
                use_batch_normalisation: bool) -> None:
            super().__init__(input_dim, hidden_dim, output_dim, dropout, use_batch_normalisation)
            self.denseSkip = nn.Linear(self.inputDim, self.outputDim)

        def _skip(self, x):
            """
            Defines the skip path as a fully connected linear layer which changes the dimension as required by this
            block.
            """
            return self.denseSkip(x)
