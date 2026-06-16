import json
import os

import torch
from torch.utils.data import Dataset


class TurkishMusicLatentsDataset(Dataset):
    def __init__(self, jsonl_path, data_base_path):
        super().__init__()
        self.data = []
        self.data_base_path = data_base_path

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Metadata file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(
            f"Loaded {len(self.data)} audio tuples from {os.path.basename(jsonl_path)}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load mixture and target latents from disk
        mixture_filename = os.path.split(self.data[idx]["mixture_path"])[-1]
        mixture_path = os.path.join(
            self.data_base_path, mixture_filename.replace(".wav", ".pt")
        )
        target_filename = os.path.split(self.data[idx]["target_path"])[-1]
        target_path = os.path.join(
            self.data_base_path, target_filename.replace(".wav", ".pt")
        )

        mixture_latent = torch.load(mixture_path)
        target_latent = torch.load(target_path)

        return {
            "mixture_latent": mixture_latent,
            "target_latent": target_latent,
            "mixture_filename": mixture_filename,
            "target_filename": target_filename,
            "prompt": self.data[idx]["prompt"],
        }


def pad_collate_fn(batch):
    """Dynamically pads text features/masks and stacks latents."""
    mixture_latents = torch.stack([item["mixture_latent"] for item in batch])
    target_latents = torch.stack([item["target_latent"] for item in batch])

    target_filenames = [item["target_filename"] for item in batch]
    prompts = [item["prompt"] for item in batch]

    return {
        "mixture_latent": mixture_latents,
        "target_latent": target_latents,
        "target_filename": target_filenames,
        "prompt": prompts,
    }
