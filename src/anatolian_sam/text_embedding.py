from anatolian_sam.load_sam import base_model

with torch.no_grad():
    text_features, text_mask = base_model.text_encoder(['zurna'])

print("Text features:", text_features.shape)
