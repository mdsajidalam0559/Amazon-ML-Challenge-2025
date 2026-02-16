
import os
import requests
import traceback
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
import multiprocessing
from functools import partial
import urllib.request
from transformers import CLIPProcessor, CLIPModel
from .config import CFG

def download_image(image_link, savefolder):
    if(isinstance(image_link, str)):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if(not os.path.exists(image_save_path)):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)    
            except Exception as ex:
                pass
                # print('Warning: Not able to download - {}\n{}'.format(image_link, ex))
        else:
            return
    return

def download_images_parallel(image_links, download_folder):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    
    download_image_partial = partial(download_image, savefolder=download_folder)
    
    # Use fewer processes to avoid overwhelming the system/network
    with multiprocessing.Pool(16) as pool:
        list(tqdm(pool.imap(download_image_partial, image_links), total=len(image_links), desc="Downloading Images"))

def get_image_path(fname, input_images_dir):
    p = os.path.join(input_images_dir, fname)
    if os.path.exists(p):
        return p
    return None

def compute_embeddings(df, image_dir):
    device = CFG.device
    model_name = CFG.model_name
    print(f"Loading CLIP model {model_name} on {device}...")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    embedding_dim = clip_model.visual_projection.out_features
    
    n_rows = len(df)
    embeddings = np.zeros((n_rows, embedding_dim), dtype=np.float32)
    valid_mask = np.zeros(n_rows, dtype=bool)
    
    print("Generating embeddings...")
    clip_model.eval()
    
    # We iterate, but ideally we should batch this for speed.
    # For now, keeping it simple as per original notebook logic but cleaner.
    
    for i, row in tqdm(df.iterrows(), total=n_rows, desc="Encoding images"):
        image_link = row['image_link']
        fname = Path(image_link).name
        fpath = get_image_path(fname, image_dir)
        
        if not fpath:
            continue
            
        try:
            img = Image.open(fpath).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                embeddings[i] = emb.cpu().numpy()
                valid_mask[i] = True
        except Exception:
            pass

    return embeddings, valid_mask

if __name__ == "__main__":
    import pandas as pd
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--download", action="store_true", help="Download images")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings")
    parser.add_argument("--test", action="store_true", help="Process test data instead of train")
    args = parser.parse_args()
    
    target_csv = CFG.TEST_CSV if args.test else CFG.TRAIN_CSV
    target_emb_path = CFG.TEST_IMAGE_EMB_PATH if args.test else CFG.IMAGE_EMB_PATH
    target_mask_path = CFG.TEST_VALID_MASK_PATH if args.test else CFG.VALID_MASK_PATH
    
    if args.download:
        print(f"Reading {target_csv}...")
        df = pd.read_csv(target_csv)
        print(f"Found {len(df)} rows.")
        image_links = df['image_link'].tolist()
        download_images_parallel(image_links, CFG.IMAGES_DIR)
        
    if args.embed:
        print(f"Reading {target_csv}...")
        df = pd.read_csv(target_csv)
        embeddings, valid_mask = compute_embeddings(df, CFG.IMAGES_DIR)
        
        print(f"Saving embeddings to {target_emb_path}")
        np.save(target_emb_path, embeddings)
        np.save(target_mask_path, valid_mask)
        print("Done.")
