import torch
from sam_audio import SAMAudio, SAMAudioProcessor


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sam_audio(
    model_id="facebook/sam-audio-base", dtype=torch.bfloat16, freeze=True
):
    """Loads the SAM Audio base model and processor."""
    device = get_device()
    processor = SAMAudioProcessor.from_pretrained(model_id)
    model = SAMAudio.from_pretrained(model_id).to(device=device, dtype=dtype)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model, processor


def get_frozen_vae(base_model):
    """Extracts and explicitly freezes the DAC-VAE codec."""
    vae = base_model.audio_codec
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae
