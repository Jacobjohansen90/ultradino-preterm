import torch


from ultradino_finetune.models.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
)
import ultradino_finetune.models.dinov2.utils.param_groups as param_groups_utils

import logging

logger = logging.getLogger("ultradino_finetune.load")

CONFIG_SHARED = {
    # Defaults from dinov2/configs/ssl_default_config.yaml
    #
    # (dinov2.models.build_model_from_cfg() picks these from the "student."
    # section)
    "drop_path_rate": 0.3,
    "init_values": 1.0e-05,  # Called 'layerscale' in the config files
    "drop_path_uniform": True,
    "ffn_layer": "mlp",
    "block_chunks": 0,
    "qkv_bias": True,
    "proj_bias": True,
    "ffn_bias": True,
    "num_register_tokens": 0,
    "interpolate_antialias": False,
    "interpolate_offset": 0.1,
    # From dinov2/models/__init__.py
    "img_size": 224,
    # Defaults in dinov2/models/vision_transformer.py
    "patch_size": 16,
    "depth": 12,
    "mlp_ratio": 4,
    # dinov2/configs/train/vits16_dinus_13m.yaml
    # dinov2/configs/train/vitb16_dinus_13m.yaml
    # dinov2/configs/train/ablations/vits16_dinus_2m_150e.yaml
    # dinov2/configs/train/ablations/vitb16_dinus_2m.yaml
    #
    # (dinov2.models.build_model_from_cfg() picks these from the "student."
    # section)
    "block_chunks": 0,
    "in_chans": 1,
    "num_register_tokens": 4,
}
CONFIG_SMALL = {
    # dinov2.models.vision_transformer:vit_small()
    "embed_dim": 384,
    "num_heads": 6,
}
CONFIG_BASE = {
    # dinov2.models.vision_transformer:vit_base()
    "embed_dim": 768,
    "num_heads": 12,
}


def get_model_config(model_type):
    if model_type == "vits16":
        return CONFIG_SHARED | CONFIG_SMALL
    elif model_type == "vitb16":
        return CONFIG_SHARED | CONFIG_BASE
    else:
        raise RuntimeError(f'Unknown model type "{model_type}"')


def is_ignored_parameter(name):
    if name.startswith("student."):
        return True
    elif name.startswith("teacher.dino_head"):
        return True
    elif name == "dino_loss.center":
        return True
    elif name == "ibot_patch_loss.center":
        return True
    return False


def load_pretrained_weights(model, path, device="cpu", keep_weights=None):
    state_dict = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    state_dict = state_dict["model"]
    state_dict = {
        k.replace("teacher.", ""): v
        for k, v in state_dict.items()
        if not is_ignored_parameter(k)
    }
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    if keep_weights:
        current_state_dict = model.state_dict()

        for key in keep_weights:
            state_dict[key] = current_state_dict[key]

    model.load_state_dict(state_dict)


def load_pretrained_ultradino(model_type, weights_path, device="cpu", **kwargs):
    """
    Load a pretrained DinoVisionTransformer model.

    Args:
        model_type (str): The type of the model, e.g., "vits16" or "vitb16".
        weights_path (str): Path to the pretrained weights file.
        device (str): Device to load the model onto, e.g., "cpu" or "cuda".
    """
    conf = get_model_config(model_type)

    conf = conf | kwargs

    model = DinoVisionTransformer(**conf)

    load_pretrained_weights(model, weights_path, device=device)

    return model


def load_from_scratch(model_type, device="cpu", **kwargs):
    """
    Load a DinoVisionTransformer model from scratch with random weights.

    Args:
        model_type (str): The type of the model, e.g., "vits16" or "vitb16".
        device (str): Device to load the model onto, e.g., "cpu" or "cuda".
    """
    conf = get_model_config(model_type)

    conf = conf | kwargs

    model = DinoVisionTransformer(**conf)

    return model.to(device)


def get_param_groups_with_decay(
    model, lr_decay_rate: float = 0.65, patch_embed_lr_mult: float = 0.2
):
    params_groups = param_groups_utils.get_params_groups_with_decay(
        model=model,
        lr_decay_rate=lr_decay_rate,
        patch_embed_lr_mult=patch_embed_lr_mult,
    )
    fused_params_groups = param_groups_utils.fuse_params_groups(params_groups)
    logger.info("fusing param groups")

    for g in fused_params_groups:
        g["foreach"] = True
    return fused_params_groups
