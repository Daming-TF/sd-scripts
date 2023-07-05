import os
import torch

from networks.lora import parse_block_lr_kwargs, LoRANetwork, LoRAModule


def create_lora_from_weights(multiplier, file, vae, text_encoder, unet,
                             weights_sd=None, module_class=None, **kwargs):

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        weights_sd = load_file(file)

    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            # print(lora_name, value.size(), dim)

    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    network = MyLoRANetwork(
        text_encoder, unet, multiplier=multiplier, modules_dim=modules_dim, modules_alpha=modules_alpha
    )

    # block lr
    down_lr_weight, mid_lr_weight, up_lr_weight = parse_block_lr_kwargs(kwargs)
    if up_lr_weight is not None or mid_lr_weight is not None or down_lr_weight is not None:
        network.set_block_lr_weight(up_lr_weight, mid_lr_weight, down_lr_weight)

    return network


class LoRAMergeModule(LoRAModule):
    def __init__(
            self,
            lora_name,
            org_module: torch.nn.Module,
            multiplier=1.0,
            lora_dim=4,
            alpha=1,
            dropout=None,
            rank_dropout=None,
            module_dropout=None,
    ):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down_A = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False, device=('cuda'))
            self.lora_up_A = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False, device=('cuda'))
            self.lora_down_B = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False, device=('cuda'))
            self.lora_up_B = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False, device=('cuda'))
        else:
            self.lora_down_A = torch.nn.Linear(in_dim, self.lora_dim, bias=False, device=('cuda'))
            self.lora_up_A = torch.nn.Linear(self.lora_dim, out_dim, bias=False, device=('cuda'))
            self.lora_down_B = torch.nn.Linear(in_dim, self.lora_dim, bias=False, device=('cuda'))
            self.lora_up_B = torch.nn.Linear(self.lora_dim, out_dim, bias=False, device=('cuda'))

        self.weights_ratios = None

    def forward(self, x):
        org_forwarded = self.org_forward(x)
        [r_a, r_b] = self.weights_ratios

        lx_a = self.lora_down_A(x)
        lx_b = self.lora_down_B(x)

        lx_a = self.lora_up_A(lx_a)
        lx_b = self.lora_up_B(lx_b)

        return org_forwarded + (r_a * lx_a + r_b * lx_b) * self.multiplier * self.scale

    def load_weights(self, down_weight_a, up_weight_a, down_weight_b, up_weight_b, ratios):
        self.weights_ratios = ratios
        # device = self.lora_down_A.weight.device
        self.lora_down_A.weight.data.copy_(down_weight_a)
        self.lora_up_A.weight.data.copy_(up_weight_a)
        self.lora_down_B.weight.data.copy_(down_weight_b)
        self.lora_up_B.weight.data.copy_(up_weight_b)


class MyLoRANetwork(LoRANetwork):
    def __init__(
            self,
            text_encoder,
            unet,
            multiplier=1.0,
            modules_dim=None,
            modules_alpha=None,
            module_class=LoRAMergeModule,
    ):
        super().__init__(text_encoder, unet, multiplier=multiplier, modules_dim=modules_dim,
                         modules_alpha=modules_alpha, module_class=module_class)

    def load_weights(self, lora_models, ratios):
        # get lora's ckpt
        state_dict_list = []
        from safetensors.torch import load_file
        for ckpt_path in lora_models:
            state_dict_list.append(load_file(ckpt_path))

        for lora in self.unet_loras:
            down_key = lora.lora_name + '.lora_down.weight'
            up_key = down_key.replace("lora_down", "lora_up")

            # load loRA
            [lora_sd_a, lora_sd_b] = state_dict_list
            down_weight_a = lora_sd_a[down_key]  # {128,320,1,1}
            up_weight_a = lora_sd_a[up_key]  # {320,128,1,1}
            down_weight_b = lora_sd_b[down_key]  # {128,320,1,1}
            up_weight_b = lora_sd_b[up_key]  # {320,128,1,1}

            lora.load_weights(down_weight_a, up_weight_a, down_weight_b, up_weight_b, ratios)
            lora.apply_to()





