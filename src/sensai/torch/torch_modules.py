# imports required for backward compatibility (with pickled objects)
from .torch_base import MCDropoutCapableNNModule
from .torch_models.mlp.mlp_modules import MultiLayerPerceptron
from .torch_models.lstnet.lstnet_modules import LSTNetwork
from .torch_models.residualffn.residualffn_modules import ResidualFeedForwardNetwork
from ..util import markUsed

markUsed(MCDropoutCapableNNModule, MultiLayerPerceptron, LSTNetwork, ResidualFeedForwardNetwork)