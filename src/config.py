import os
import torch

class CFG:
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "openai/clip-vit-large-patch14" 
    
    # Base paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "student_resource", "dataset")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    
    # Input Paths
    # Assuming train.csv is in dataset folder, though it was missing in the listing.
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv") 
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    
    # Generated Artifacts
    IMAGE_EMB_PATH = os.path.join(OUTPUT_DIR, "image_embeddings.npy")
    VALID_MASK_PATH = os.path.join(OUTPUT_DIR, "valid_mask.npy")
    
    TEST_IMAGE_EMB_PATH = os.path.join(OUTPUT_DIR, "test_image_embeddings.npy")
    TEST_VALID_MASK_PATH = os.path.join(OUTPUT_DIR, "test_valid_mask.npy")
    
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")
    
    # Training
    epochs = 10
    batch_size = 64
    encoder_lr = 2e-5
    decoder_lr = 1e-3
    weight_decay = 1e-5
    
    # Model Save
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "clip_img_proj_best.pth")
    
    # Image Download
    IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
os.makedirs(CFG.MODEL_DIR, exist_ok=True)
os.makedirs(CFG.IMAGES_DIR, exist_ok=True)
