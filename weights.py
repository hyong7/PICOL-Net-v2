from pathlib import Path
import torch

from exp.finetune_target_dora import config_from_checkpoint
from models.dora import inject_dora_linear_layers
from models.picolnet import PicolNet
from models.memory_v2 import AnchorPool, MemoryV2


def load_model(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = PicolNet(config_from_checkpoint(checkpoint)).to(device)
    if "dora" in checkpoint:
        inject_dora_linear_layers(model)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def load_memory(checkpoint, device="cpu"):
    state = checkpoint["memory"]
    memory = MemoryV2(
        dim=state["config"]["dim"],
        gradient_dim=state["config"]["gradient_dim"],
        device=device,
    )

    def pool(values):
        return AnchorPool(
            **{
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in values.items()
            }
        )

    memory.anchor = pool(state["anchor"])
    memory.experience = pool(checkpoint["experience"])
    memory.current_vehicle = state["current_vehicle"]
    memory.bias_state = state.get("bias_state", {})
    memory.diag.update(state.get("diag", {}))
    return memory
