#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class LoRALayer():
    def __init__(
            self,
            r_nat: int,
            lora_alpha: int
    ):
        self.r_nat = r_nat
        self.lora_alpha = lora_alpha


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r_nat=0, lora_alpha=1, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r_nat=r_nat, lora_alpha=lora_alpha)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        self.r_nat = r_nat
        if r_nat > 0:
            self.lora_A_nat = nn.Parameter(
                self.conv.weight.new_zeros((r_nat * kernel_size, in_channels * kernel_size))
            )
            self.lora_B_nat = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, r_nat * kernel_size))
            )
            self.scaling_nat = self.lora_alpha / self.r_nat
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A_nat'):
            if self.r_nat > 0:
                nn.init.normal_(self.lora_A_nat)
                nn.init.zeros_(self.lora_B_nat)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)

    def forward(self, x):
        if self.r_nat > 0:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B_nat @ self.lora_A_nat).view(self.conv.weight.shape) * self.scaling_nat,
                self.conv.bias
            )
        else:
            return self.conv._conv_forward(x, self.conv.weight, self.conv.bias)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict


# from .layers import LoRALayer


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        # if 'lora_B_nat' not in n:
        if 'lora_' not in n:
            p.requires_grad = False

        # if 'fc' not in n:
            # p.requires_grad = True

    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError