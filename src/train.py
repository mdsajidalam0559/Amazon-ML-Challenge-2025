
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import CFG
from .model import ContentOnlyFinetuneModel
from .dataset import ContentOnlyDataset
from .utils import calculate_smape

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch['target'])
        if torch.isnan(loss): 
            continue
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            predictions = model(batch)
            loss = criterion(predictions, batch['target'])
            total_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(batch['target'].cpu().numpy())
    all_preds, all_targets = np.concatenate(all_preds), np.concatenate(all_targets)
    true_prices, pred_prices = np.expm1(all_targets), np.expm1(all_preds)
    val_smape = calculate_smape(true_prices, pred_prices)
    return total_loss / len(loader), val_smape

def main():
    # Load Data
    print("Loading data...")
    if not os.path.exists(CFG.TRAIN_CSV):
        raise FileNotFoundError(f"Train CSV not found at {CFG.TRAIN_CSV}")
        
    df_main = pd.read_csv(CFG.TRAIN_CSV)
    
    if not os.path.exists(CFG.IMAGE_EMB_PATH):
         raise FileNotFoundError(f"Image embeddings not found at {CFG.IMAGE_EMB_PATH}. Run prepare_data.py first.")
         
    image_embeddings = np.load(CFG.IMAGE_EMB_PATH)
    valid_mask = np.load(CFG.VALID_MASK_PATH)
    
    from transformers import CLIPTokenizerFast
    tokenizer = CLIPTokenizerFast.from_pretrained(CFG.model_name)
    
    # Dataset and Loader
    all_indices = range(len(df_main))
    full_train_dataset = ContentOnlyDataset(df_main, tokenizer, image_embeddings, valid_mask, all_indices)
    full_train_loader = DataLoader(full_train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    # Model
    print("Initializing model...")
    model = ContentOnlyFinetuneModel().to(CFG.device)
    criterion = nn.MSELoss()
    
    param_groups = [
        {'params': model.clip.text_model.parameters(), 'lr': CFG.encoder_lr},
        {'params': list(model.head.parameters()) +
                   list(model.clip.text_projection.parameters()) +
                   list(model.image_projection_head.parameters()) +
                   [model.missing_image_embedding], 'lr': CFG.decoder_lr}
    ]
    optimizer = AdamW(param_groups, weight_decay=CFG.weight_decay)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[CFG.encoder_lr, CFG.decoder_lr],
        epochs=CFG.epochs,
        steps_per_epoch=len(full_train_loader),
        anneal_strategy='linear',
        pct_start=0.3
    )

    print("Starting training...")
    for epoch in range(CFG.epochs):
        print(f"\nEpoch {epoch+1}/{CFG.epochs}")
        train_loss = train_one_epoch(model, full_train_loader, optimizer, scheduler, criterion, CFG.device)
        print(f"Train Loss = {train_loss:.4f}")
        
        # Save per epoch if needed, or just best/final
    
    # Save Final
    # Handle DataParallel separation if needed, though state_dict is usually fine if we access module
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), CFG.MODEL_SAVE_PATH)
    else:
        torch.save(model.state_dict(), CFG.MODEL_SAVE_PATH)
        
    print(f"Model saved to {CFG.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
