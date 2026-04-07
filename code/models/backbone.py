# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # 针对小目标检测：使用 dilation 替代 stride，减小下采样倍率
        # layer3 和 layer4 使用 dilation，使最终 stride 从 32 变为 16
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, True, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.
    Crucial for small object detection as it combines high-resolution shallow features
    with semantic-rich deep features.
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            layer_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.inner_blocks.append(inner_conv)
            self.layer_blocks.append(layer_conv)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: Dict[str, NestedTensor]) -> List[NestedTensor]:
        # Extract tensors from NestedTensor
        feature_tensors = [f.tensors for f in features.values()]
        
        # Last layer inner connection
        last_inner = self.inner_blocks[-1](feature_tensors[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        # Top-down pathway with lateral connections
        for i in range(len(feature_tensors) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](feature_tensors[i])
            
            # Upsample the top feature map to match the current resolution
            upsampled = F.interpolate(
                last_inner, 
                size=inner_lateral.shape[-2:], 
                mode='nearest'
            )
            last_inner = inner_lateral + upsampled
            results.insert(0, self.layer_blocks[i](last_inner))
        
        return results


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    # Enable multi-scale features for small object detection
    return_interm_layers = True
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    
    # Build FPN for multi-scale feature fusion
    # ResNet-50: layer1=256, layer2=512, layer3=1024, layer4=2048
    in_channels_list = [256, 512, 1024, 2048]
    fpn = FeaturePyramidNetwork(in_channels_list, out_channels=256)
    
    model = Joiner(backbone, position_embedding)
    model.num_channels = 256  # FPN output channels
    model.fpn = fpn  # Attach FPN to model
    return model
