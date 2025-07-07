# ipadapter_patch.py

import torch
from diffusers import StableDiffusionControlNetPipeline
from huggingface_hub import snapshot_download

def apply_ipadapter(
    pipe: StableDiffusionControlNetPipeline,
    ip_ckpt: str = "ip-adapter_sd15.bin",
    encoder_repo: str = "h94/IP-Adapter",
    encoder_subfolder: str = "models/image_encoder",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    scale: float = 1.0,
) -> StableDiffusionControlNetPipeline:
    """
    Apply IPAdapter integration to an existing StableDiffusionControlNetPipeline.

    Args:
        pipe (StableDiffusionControlNetPipeline): Your initialized SD+ControlNet pipeline.
        ip_ckpt (str): Name of the IP-Adapter checkpoint file.
        encoder_repo (str): Hugging Face repo for the encoder.
        encoder_subfolder (str): Subfolder inside the repo where the encoder weights live.
        device (torch.device): Torch device for loading.
        scale (float): Default strength for style conditioning.

    Returns:
        Patched StableDiffusionControlNetPipeline with style conditioning enabled.
    """
    # Download encoder weights
    encoder_path = snapshot_download(
        repo_id=encoder_repo,
        repo_type="model",
        allow_patterns=[f"{encoder_subfolder}/*"]
    ) + f"/{encoder_subfolder}"

pipe.load_ip_adapter(
    pretrained_model_name_or_path_or_dict="h94/IP-Adapter",
    subfolder="models",
    weight_name=ip_ckpt
)



    pipe.set_ip_adapter_scale(scale)
    pipe.to(device)
    return pipe
