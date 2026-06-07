import torch
from sam_audio import SAMAudio, SAMAudioProcessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_ID = "facebook/sam-audio-base"  # base model

processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
base_model = SAMAudio.from_pretrained(MODEL_ID)
base_moel = base_model.to(device=device, dtype=torch.float16)

base_model.eval()  # Freeze model
base_model.text_encoder.requires_grad_(False)
