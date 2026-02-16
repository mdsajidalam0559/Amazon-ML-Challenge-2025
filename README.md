# Amazon ML Challenge 2025 - Smart Product Pricing

A multimodal machine learning solution for predicting e-commerce product prices using both textual catalog descriptions and product images. Built on OpenAI's CLIP model, the system fuses text and vision embeddings through a custom regression head to produce price predictions.

## Problem

Given a product's catalog text (title, description, item pack quantity, specs) and an image URL, predict the product's price. The task is evaluated using SMAPE (Symmetric Mean Absolute Percentage Error), where lower is better (bounded between 0% and 200%).

Training and test sets each contain 75,000 products.

## Project Structure

```
src/
  config.py         - Hyperparameters and path configuration
  model.py          - CLIP-based multimodal regression model
  dataset.py        - PyTorch Dataset for text + image embeddings
  train.py          - Training loop
  predict.py        - Inference and submission generation
  prepare_data.py   - Image downloading and CLIP embedding extraction
  utils.py          - SMAPE metric calculation
student_resource/
  dataset/
    train.csv       - Training data (75k samples with prices)
    test.csv        - Test data (75k samples, no prices)
    sample_test.csv - Small sample test input
    sample_test_out.csv - Example output format
  README.md         - Challenge problem statement
  Documentation_template.md - Submission documentation template
output/             - Generated embeddings and submission files
models/             - Saved model checkpoints
requirements.txt    - Python dependencies
```

## Approach

The model uses CLIP (clip-vit-large-patch14) as a backbone for both text and image understanding.

**Text pipeline:** Product catalog text is tokenized and encoded through CLIP's text encoder to produce a 768-dimensional embedding.

**Image pipeline:** Product images are downloaded and pre-encoded into 768-dimensional CLIP embeddings. A learned placeholder embedding handles missing or invalid images. A trainable projection head refines the image embeddings before fusion.

**Fusion and prediction:** Text and image embeddings are concatenated into a 1536-dimensional vector, then passed through a regression head (LayerNorm, Linear, GELU, Dropout, Linear) that outputs a single price prediction.

**Training details:**
- Prices are log-transformed (log1p) during training and converted back (expm1) at inference
- Loss function: MSE in log-space
- Optimizer: AdamW with differential learning rates (2e-5 for encoder, 1e-3 for head)
- Scheduler: OneCycleLR
- Batch size: 64, Epochs: 10

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Download images and generate embeddings

```bash
python -m src.prepare_data --download --embed
```

This downloads product images from the URLs in the dataset and extracts CLIP image embeddings, saving them as numpy arrays in the output directory.

### 2. Train the model

```bash
python -m src.train
```

Trains for 10 epochs and saves the best checkpoint to `models/clip_img_proj_best.pth`.

### 3. Generate predictions

```bash
python -m src.predict
```

Loads the trained model, runs inference on the test set, and writes `output/submission.csv`.

## Output Format

The submission file is a CSV with two columns:

```
sample_id,price
217392,62.08
209156,17.19
262333,96.50
```

Each row corresponds to a test sample. Prices must be positive floats.

## Dependencies

- Python 3.13
- PyTorch
- Transformers (Hugging Face)
- torchvision
- pandas, numpy, scikit-learn
- Pillow, requests, tqdm

See [requirements.txt](requirements.txt) for the full list.

## Evaluation

SMAPE formula:

```
SMAPE = (1/n) * sum(|predicted - actual| / ((|actual| + |predicted|) / 2)) * 100
```

## Constraints

- Only the provided training data may be used. External price lookup is prohibited.
- Final model must be MIT or Apache 2.0 licensed and under 8 billion parameters.
