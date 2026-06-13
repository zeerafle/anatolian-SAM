import torch


def expand_to_256(latent_128):
    """Duplicates 128-channel latents to 256 for SAM-Audio DiT conditioning."""
    return torch.cat([latent_128, latent_128], dim=-1)


def slice_to_128(latent_256):
    """Slices a 256-channel latent back down to 128 channels for VAE decoding."""
    return latent_256[:, :, :128]
