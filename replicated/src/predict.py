
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import CLIPTokenizerFast

from .config import CFG
from .model import ContentOnlyFinetuneModel
from .dataset import ContentOnlyDataset

def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(batch)
            all_preds.append(predictions.cpu().numpy())
    return np.concatenate(all_preds)

def main():
    print("Loading test data...")
    if not os.path.exists(CFG.TEST_CSV):
        raise FileNotFoundError(f"Test CSV not found at {CFG.TEST_CSV}")
        
    df_test = pd.read_csv(CFG.TEST_CSV)
    
    if not os.path.exists(CFG.TEST_IMAGE_EMB_PATH):
         raise FileNotFoundError(f"Test embeddings not found at {CFG.TEST_IMAGE_EMB_PATH}. Run prepare_data.py --test --embed first.")
         
    test_image_embeddings = np.load(CFG.TEST_IMAGE_EMB_PATH)
    test_valid_mask = np.load(CFG.TEST_VALID_MASK_PATH)
    
    tokenizer = CLIPTokenizerFast.from_pretrained(CFG.model_name)
    
    # Dataset and Loader
    # Test dataset typically has different structure or we reuse ContentOnlyDataset 
    # ContentOnlyDataset handles standard catalog_content, image_link etc.
    # We pass all indices.
    
    print(f"Creating test dataset with {len(df_test)} samples.")
    test_dataset = ContentOnlyDataset(df_test, tokenizer, test_image_embeddings, test_valid_mask, indices=None) # Use all
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    print("Loading model...")
    model = ContentOnlyFinetuneModel().to(CFG.device)
    
    if not os.path.exists(CFG.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {CFG.MODEL_SAVE_PATH}")
        
    state_dict = torch.load(CFG.MODEL_SAVE_PATH, map_location=CFG.device)
    model.load_state_dict(state_dict)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    print("Running inference...")
    log_predictions = get_predictions(model, test_loader, CFG.device)
    
    # Convert back to price
    final_predictions = np.expm1(log_predictions)
    
    # Submission
    submission_df = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': final_predictions
    })
    
    submission_df.to_csv(CFG.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {CFG.SUBMISSION_PATH}")
    print(submission_df.head())

if __name__ == "__main__":
    main()
