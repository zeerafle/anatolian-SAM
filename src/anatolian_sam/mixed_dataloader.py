import json
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class TurkishMusicMixedAudioDataset(Dataset):
    """
    A PyTorch Dataset for loading raw .wav files for zero-shot testing
    of SAM-Audio. Handles on-the-fly resampling and mono conversion.
    """
    def __init__(self, jsonl_path, data_base_path, target_sr=16000):
        super().__init__()
        self.data = []
        self.target_sr = target_sr
        self.data_base_path = data_base_path

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Metadata file not found: {jsonl_path}")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} audio tuples from {os.path.basename(jsonl_path)}")

    def __len__(self):
        return len(self.data)

    def _load_and_process_audio(self, audio_path):
        # Load audio; returns [Channels, Time]
        wav, orig_sr = torchaudio.load(audio_path)

        # Resample if necessary
        if orig_sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, orig_sr, self.target_sr)

        # Convert to Mono
        wav = wav.mean(0, keepdim=True)
        return wav.float()

    def __getitem__(self, idx):
        row = self.data[idx]

        mixture_filename = os.path.split(row["mixture_path"])[-1]
        mixture_path = os.path.join(self.data_base_path, mixture_filename)
        target_filename = os.path.split(row["target_path"])[-1]
        target_path = os.path.join(self.data_base_path, target_filename)

        mixture_wav = self._load_and_process_audio(mixture_path)
        target_wav = self._load_and_process_audio(target_path)

        return {
            "prompt": row["prompt"],
            "mixture_wav": mixture_wav,
            "target_wav": target_wav,
            "mixture_filename": mixture_filename,
            "target_filename": target_filename
        }

def get_audio_dataloader(jsonl_path, data_base_path, target_sr, batch_size=4, shuffle=False, num_workers=2):
    """
    Factory function to initialize the Dataset and DataLoader.
    """
    dataset = TurkishMusicMixedAudioDataset(jsonl_path, data_base_path, target_sr)

    def custom_collate(batch):
        prompts = [item["prompt"] for item in batch]
        mixture_filenames = [item["mixture_filename"] for item in batch]
        target_filenames = [item["target_filename"] for item in batch]

        # Extract wavs. Original shape is [1, T].
        # pad_sequence expects [T, C], so we transpose before padding.
        mix_wavs_t = [item["mixture_wav"].transpose(0, 1) for item in batch]
        tgt_wavs_t = [item["target_wav"].transpose(0, 1) for item in batch]

        # Pad sequences to the longest in the batch, then transpose back to [B, 1, T]
        mixture_padded = torch.nn.utils.rnn.pad_sequence(mix_wavs_t, batch_first=True).transpose(1, 2)
        target_padded = torch.nn.utils.rnn.pad_sequence(tgt_wavs_t, batch_first=True).transpose(1, 2)

        return {
            "prompts": prompts,
            "mixture_wavs": mixture_padded, # Now a proper tensor of shape [B, 1, T]
            "target_wavs": target_padded,
            "mixture_filenames": mixture_filenames,
            "target_filenames": target_filenames
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, # Set to False for reproducible zero-shot evaluation
        num_workers=num_workers,
        collate_fn=custom_collate
    )

    return dataset, loader
