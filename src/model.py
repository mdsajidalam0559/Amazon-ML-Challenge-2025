import torch
import torch.nn as nn
from transformers import CLIPModel
from .config import CFG

class ContentOnlyFinetuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CFG.model_name)
        embed_dim = self.clip.config.projection_dim

        # Placeholder embedding for missing images
        self.missing_image_embedding = nn.Parameter(torch.randn(1, embed_dim))

        # Trainable projection head for the image embeddings
        self.image_projection_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        final_input_size = embed_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(final_input_size),
            nn.Linear(final_input_size, embed_dim), # Intermediate layer
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 1) # Final output layer
        )

    def forward(self, batch):
        text_features = self.clip.get_text_features(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        image_emb_raw, valid_mask = batch['image_emb'], batch['valid_mask'].unsqueeze(-1)
        image_features_raw = image_emb_raw * valid_mask + self.missing_image_embedding * (1 - valid_mask)

        image_features = self.image_projection_head(image_features_raw)

        final_vector = torch.cat([image_features, text_features], dim=1)
        prediction = self.head(final_vector)
        return prediction.squeeze(-1)
