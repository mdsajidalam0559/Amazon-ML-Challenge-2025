import torch
from torch.utils.data import Dataset
import numpy as np

class ContentOnlyDataset(Dataset):
    def __init__(self, df, tokenizer, image_embeddings, valid_mask, indices=None):
        if indices is not None:
            self.df = df.iloc[indices].reset_index(drop=True)
            self.image_embeddings = image_embeddings[indices]
            self.valid_mask = valid_mask[indices]
        else:
            self.df = df
            self.image_embeddings = image_embeddings
            self.valid_mask = valid_mask
            
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = str(row['catalog_content']) # Ensure string
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        image_emb = self.image_embeddings[idx]
        is_valid_image = self.valid_mask[idx]
        
        item = {
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze(),
            'image_emb': torch.tensor(image_emb, dtype=torch.float32),
            'valid_mask': torch.tensor(is_valid_image, dtype=torch.float32),
        }
        
        if 'price' in row:
            price = row['price']
            log_price = np.log1p(price)
            item['target'] = torch.tensor(log_price, dtype=torch.float32)
            
        return item
