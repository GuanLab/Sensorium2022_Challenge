
import warnings
from collections import Iterable, OrderedDict
from functools import partial

import torch
import torchvision
from torch import nn

from neuralpredictors import regularizers
from neuralpredictors.layers.activations import AdaptiveELU
from neuralpredictors.layers.affine import Bias2DLayer, Scale2DLayer
from neuralpredictors.layers.attention import AttentionConv
from neuralpredictors.layers.conv import DepthSeparableConv2d

from .base import Core

# logger = logging.getLogger(__name__)


class Stacked2dCore(Core, nn.Module):
    """
    An instantiation of the Core base class. Made up of layers layers of nn.sequential modules.
    Allows for the flexible implementations of many different architectures, such as convolutional layers,
    or self-attention layers.
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        layers=3,
        gamma_hidden=0,
        gamma_input=0.0,
        skip=0,
        stride=1,
        final_nonlinearity=True,
        scale=None,
        elu_shift=(0, 0),
        bias=True,
        momentum=0.1,
        pad_input=True,
        hidden_padding=None,
        batch_norm=True,
        batch_norm_scale=True,
        independent_bn_bias=True,
        hidden_dilation=1,
        laplace_padding=0,
        input_regularizer="LaplaceL2",
        stack=None,
        use_avg_reg=True,
        depth_separable=False,
        attention_conv=False,
        linear=False,
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see neuralpredictors.regularizers)
            skip:           Adds a skip connection
            stride:         stride of the 2d conv layer.
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            elu_shift: a tuple to shift the elu in the following way: Elu(x - elu_xshift) + elu_yshift
            bias:           Adds a bias layer.
            momentum:       momentum in the batchnorm layer.
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            hidden_padding: int or list of int. Padding for hidden layers. Note that this will apply to all the layers
                            except the first (input) layer.
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            batch_norm_scale: If True, a scaling factor after BN will be learned.
            independent_bn_bias:    If False, will allow for scaling the batch norm, so that batchnorm
                                    and bias can both be true. Defaults to True.
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            input_regularizer:  String that must match one of the regularizers in ..regularizers
            stack:        Int or iterable. Selects which layers of the core should be stacked for the readout.
                            default value will stack all layers on top of each other.
                            Implemented as layers_to_stack = layers[stack:]. thus:
                                stack = -1 will only select the last layer as the readout layer.
                                stack of -2 will read out from the last two layers.
                                And stack of 1 will read out from layer 1 (0 indexed) until the last layer.
            use_avg_reg:    bool. Whether to use the averaged value of regularizer(s) or the summed.
            depth_separable: Boolean, if True, uses depth-separable convolutions in all layers after the first one.
            attention_conv: Boolean, if True, uses self-attention instead of convolution for all layers after the first one.
            linear:         Boolean, if True, removes all nonlinearities

            To enable learning batch_norms bias and scale independently, the arguments bias, batch_norm and batch_norm_scale
            work together: By default, all are true. In this case there won't be a bias learned in the convolutional layer, but
            batch_norm will learn both its bias and scale. If batch_norm is false, but bias true, a bias will be learned in the
            convolutional layer. If batch_norm and bias are true, but batch_norm_scale is false, batch_norm won't have learnable
            parameters and a BiasLayer will be added after the batch_norm layer.
        """

        if depth_separable and attention_conv:
            raise ValueError("depth_separable and attention_conv can not both be true")

        super().__init__()
        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)
        self.num_layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.stride = stride
        self.use_avg_reg = use_avg_reg
        if use_avg_reg:
            warnings.warn("The averaged value of regularizer will be used.", UserWarning)
        self.hidden_padding = hidden_padding
        self.input_kern = input_kern
        self.hidden_kern = hidden_kern
        self.laplace_padding = laplace_padding
        self.hidden_dilation = hidden_dilation
        self.final_nonlinearity = final_nonlinearity
        self.elu_xshift, self.elu_yshift = elu_shift
        self.bias = bias
        self.momentum = momentum
        self.pad_input = pad_input
        self.batch_norm = batch_norm
        self.batch_norm_scale = batch_norm_scale
        self.independent_bn_bias = independent_bn_bias
        if stack is None:
            self.stack = range(self.num_layers+1)
        else:
            self.stack = [*range(self.num_layers+1)[stack:]] if isinstance(stack, int) else stack
        self.linear = linear

        if depth_separable:
            self.conv_layer_name = "ds_conv"
            self.ConvLayer = DepthSeparableConv2d
            self.ignore_group_sparsity = True
        elif attention_conv:
            # TODO: check if name attention_conv is backwards compatible
            self.conv_layer_name = "attention_conv"
            self.ConvLayer = self.AttentionConvWrapper
            self.ignore_group_sparsity = True
        else:
            self.conv_layer_name = "conv"
            self.ConvLayer = nn.Conv2d
            self.ignore_group_sparsity = False

        if (self.ignore_group_sparsity) and (gamma_hidden > 0):
            warnings.warn(
                "group sparsity can not be calculated for the requested conv type. Hidden channels will not be regularized and gamma_hidden is ignored."
            )
        self.set_batchnorm_type()
        self.features = nn.Sequential()
        self.add_scale_layer()
        self.add_first_layer()
        self.add_subsequent_layers()
        self.initialize()

    def set_batchnorm_type(self):
        self.batchnorm_layer_cls = nn.BatchNorm2d
        self.bias_layer_cls = Bias2DLayer
        self.scale_layer_cls = Scale2DLayer

    def add_bn_layer(self, layer):
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = self.batchnorm_layer_cls(self.hidden_channels, momentum=self.momentum)
            else:
                layer["norm"] = self.batchnorm_layer_cls(
                    self.hidden_channels, momentum=self.momentum, affine=self.bias and self.batch_norm_scale
                )
                if self.bias:
                    if not self.batch_norm_scale:
                        layer["bias"] = self.bias_layer_cls(self.hidden_channels)
                elif self.batch_norm_scale:
                    layer["scale"] = self.scale_layer_cls(self.hidden_channels)

    def add_activation(self, layer):
        if self.linear:
            return
        if len(self.features) < self.num_layers - 1 or self.final_nonlinearity:
            layer["nonlin"] = AdaptiveELU(self.elu_xshift, self.elu_yshift)

    def add_first_layer(self):
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            self.hidden_channels//4,
            self.hidden_channels,
            self.input_kern,
            padding=self.input_kern // 2 if self.pad_input else 0,
            bias=self.bias and not self.batch_norm,
        )
        layer["norm"] = nn.BatchNorm2d(self.hidden_channels, momentum=self.momentum)
        self.add_activation(layer)
        self.features.add_module("layer0", nn.Sequential(layer))

    def add_subsequent_layers(self):
        if not isinstance(self.hidden_kern, Iterable):
            self.hidden_kern = [self.hidden_kern] * (self.num_layers - 1)

        for l in range(1, self.num_layers):
            layer = OrderedDict()
            if self.hidden_padding is None:
                self.hidden_padding = ((self.hidden_kern[l - 1] - 1) * self.hidden_dilation + 1) // 2
            layer[self.conv_layer_name] = self.ConvLayer(
                in_channels=self.hidden_channels*2 if l > 1 else self.hidden_channels,
                out_channels=self.hidden_channels*2,
                kernel_size=self.hidden_kern[l - 1],
                stride=self.stride,
                padding=self.hidden_padding,
                dilation=self.hidden_dilation,
                bias=self.bias,
            )
            layer["norm"] = nn.BatchNorm2d(self.hidden_channels*2, momentum=self.momentum)
            self.add_activation(layer)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))
            
    def add_scale_layer(self):
        layer = OrderedDict()
        layer["scale"] = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels//4,
            kernel_size=4, 
            stride=4,
            bias=self.bias)
        layer["norm"] = nn.BatchNorm2d(self.hidden_channels//4, momentum=self.momentum)
        self.add_activation(layer)
        self.features.add_module("layer-1", nn.Sequential(layer))

    class AttentionConvWrapper(AttentionConv):
        def __init__(self, dilation=None, **kwargs):
            """
            Helper class to make an attention conv layer accept input args of a pytorch.nn.Conv2d layer.
            Args:
                dilation: catches this argument from the input args, and ignores it
                **kwargs:
            """
            super().__init__(**kwargs)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            # print("call:", feat)
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l) :], dim=1))
            # print("out shape:", input_.size())
            ret.append(input_)
        # print([e.size() for e in ret])
        # print([ret[-1].size()])
        # print((torch.cat([ret[-1]], dim=1)).size())
        return torch.cat([ret[ind] for ind in self.stack], dim=1)

    def laplace(self):
        """
        Laplace regularization for the filters of the first conv2d layer.
        """
        reg_scale = self._input_weights_regularizer(self.features[0].scale.weight, avg=self.use_avg_reg)
        reg_first_conv = self._input_weights_regularizer(self.features[1].conv.weight, avg=self.use_avg_reg)
        return reg_scale 

    def group_sparsity(self):
        """
        Sparsity regularization on the filters of all the conv2d layers except the first one.
        """
        ret = 0
        if self.ignore_group_sparsity:
            return ret

        for feature in self.features[1:]:
            ret = ret + feature.conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.num_layers - 1) if self.num_layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels

