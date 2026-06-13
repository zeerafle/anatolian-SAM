import json
import os

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


class TurkishMusicLatentsDataset(Dataset):
    def __init__(self, jsonl_path, data_base_path, text_encoder):
        super().__init__()
        self.data = []
        self.data_base_path = data_base_path
        self.text_encoder = text_encoder

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

        # encode the prompt using the text encoder
        with torch.no_grad():
            features, mask = self.text_encoder(self.data[idx]["prompt"])

        return {
            "mixture_latent": mixture_latent,
            "target_latent": target_latent,
            "text_features": features,
            "text_mask": mask,
            "mixture_filename": mixture_filename,
            "target_filename": target_filename,
        }


def pad_collate_fn(batch):
    """Dynamically pads text features/masks and stacks latents."""
    mixture_latents = torch.stack([item["mixture_latent"] for item in batch])
    target_latents = torch.stack([item["target_latent"] for item in batch])

    text_feats = [item["text_features"].squeeze(0) for item in batch]
    text_masks = [item["text_mask"].squeeze(0) for item in batch]

    padded_text_feats = pad_sequence(text_feats, batch_first=True, padding_value=0.0)
    padded_text_masks = pad_sequence(text_masks, batch_first=True, padding_value=0)

    return {
        "mixture_latent": mixture_latents,
        "target_latent": target_latents,
        "text_features": padded_text_feats,
        "text_mask": padded_text_masks,
    }


def get_train_val_loaders(dataset, batch_size=4, train_ratio=0.8, seed=42):
    """Splits dataset and returns pre-configured DataLoaders."""
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn
    )

    return train_loader, test_loader
