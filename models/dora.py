import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRALinear(nn.Module):

    def __init__(self, linear):
        super().__init__()
        rank = 64
        alpha = 16.0
        eps = 5e-06
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.eps = eps
        self.weight_base = nn.Parameter(
            linear.weight.detach().clone(), requires_grad=False
        )
        if linear.bias is None:
            self.register_parameter("bias_base", None)
        else:
            self.bias_base = nn.Parameter(
                linear.bias.detach().clone(), requires_grad=False
            )
        device = linear.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, device=device))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        with torch.no_grad():
            magnitude = self.weight_base.norm(dim=1).clamp_min(eps)
        self.magnitude = nn.Parameter(magnitude.to(device))

    @property
    def weight(self):
        return self.adapted_weight()

    @property
    def bias(self):
        return self.bias_base

    def adapted_weight(self):
        update = torch.matmul(self.lora_b, self.lora_a) * self.scaling
        direction = self.weight_base + update
        direction_norm = direction.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return direction / direction_norm * self.magnitude.view(-1, 1)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}, bias={self.bias is not None}"


def inject_dora_linear_layers(module):
    replaced = []

    def replace_children(parent, prefix=""):
        for child_name, child in list(parent.named_children()):
            full_name = child_name if prefix == "" else f"{prefix}.{child_name}"
            if isinstance(child, DoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, child_name, DoRALinear(child))
                replaced.append(full_name)
            else:
                replace_children(child, full_name)

    replace_children(module)
    return replaced


def freeze_non_dora_parameters(module):
    for parameter in module.parameters():
        parameter.requires_grad = False
    for dora_module in iter_dora_modules(module):
        dora_module.lora_a.requires_grad = True
        dora_module.lora_b.requires_grad = True
        dora_module.magnitude.requires_grad = True


def iter_dora_modules(module):
    for child in module.modules():
        if isinstance(child, DoRALinear):
            yield child


def dora_state_dict(module):
    return {
        name: tensor.detach().cpu()
        for (name, tensor) in module.state_dict().items()
        if name.endswith(("lora_a", "lora_b", "magnitude"))
    }


def dora_parameter_report(module, replaced_modules=None):
    total_params = sum((parameter.numel() for parameter in module.parameters()))
    trainable_params = sum(
        (
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )
    dora_params = sum(
        (
            dora_module.lora_a.numel()
            + dora_module.lora_b.numel()
            + dora_module.magnitude.numel()
            for dora_module in iter_dora_modules(module)
        )
    )
    return {
        "num_dora_modules": sum((1 for _ in iter_dora_modules(module))),
        "replaced_modules": [] if replaced_modules is None else list(replaced_modules),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "dora_params": int(dora_params),
        "trainable_ratio": float(trainable_params / max(total_params, 1)),
    }
