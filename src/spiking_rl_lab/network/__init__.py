"""Network package public API and built-in node registration."""

from spiking_rl_lab.network.network import Network, NetworkCfg, build_network
from spiking_rl_lab.network.nodes.activations import (
    LIFNode,
    LIFNodeCfg,
    LINode,
    LINodeCfg,
    ReLUNode,
    ReLUNodeCfg,
    SiLUNode,
    SiLUNodeCfg,
)
from spiking_rl_lab.network.nodes.base_node import BaseNode, BaseNodeCfg, ListState
from spiking_rl_lab.network.nodes.common_nodes import (
    BatchNormNode,
    BatchNormNodeCfg,
    ConvNode,
    ConvNodeCfg,
    LinearNode,
    LinearNodeCfg,
)
from spiking_rl_lab.network.nodes.register import register_node
from spiking_rl_lab.network.shape import (
    DenseTensorShape,
    ImageTensorShape,
    SequenceTensorShape,
    TensorShape,
    TensorShapeKind,
)

__all__ = [
    "BaseNode",
    "BaseNodeCfg",
    "BatchNormNode",
    "BatchNormNodeCfg",
    "ConvNode",
    "ConvNodeCfg",
    "DenseTensorShape",
    "ImageTensorShape",
    "LIFNode",
    "LIFNodeCfg",
    "LINode",
    "LINodeCfg",
    "LinearNode",
    "LinearNodeCfg",
    "ListState",
    "Network",
    "NetworkCfg",
    "ReLUNode",
    "ReLUNodeCfg",
    "SequenceTensorShape",
    "SiLUNode",
    "SiLUNodeCfg",
    "TensorShape",
    "TensorShapeKind",
    "build_network",
    "register_node",
]
