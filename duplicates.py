import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from DeezyMatch import Predictor, utils

# ==== CONFIG ====
THRESHOLD = 0.99      # similarity threshold for grouping
BATCH_SIZE = 512       # batch size for encoding
MASTER_FILE_PATH = "data/raw/master file.csv"
OUTPUT_PATH = "data/output/duplicate_groups.xlsx"

# ==== STEP 1: Load data ====
print("Loading master file...")
df = pd.read_csv(MASTER_FILE_PATH)
df['sku'] = df['sku'].astype(str)
df['product_name_en'] = df['product_name'].fillna("")
df['product_name_ar'] = df['product_name_ar'].fillna("")
print(f"Loaded {len(df)} records")

# Combine Arabic + English into single unified column for matching
df['combined_name'] = df['product_name_en'].astype(str) + " " + df['product_name_ar'].astype(str)
df['combined_name'] = df['combined_name'].apply(utils.minimal_clean_text)

# ==== STEP 2: Initialize Predictor ====
print("Loading DeezyMatch model...")
predictor = Predictor()

# ==== STEP 3: Encode all product names ====
print("Encoding product names...")
embeddings = predictor.encode_batch(df['combined_name'].tolist())
embeddings = F.normalize(embeddings, dim=1)  # normalize for cosine similarity

# ==== STEP 4: Compute similarity and group duplicates ====
print("Computing similarities and grouping duplicates...")
groups = []
visited = set()

for i in tqdm(range(len(df)), desc="Finding duplicates"):
    if i in visited:
        continue
    # Compute similarity of item i to all others
    sim = torch.matmul(embeddings[i], embeddings.T)
    sim[i] = -1  # ignore self
    similar_idx = (sim >= THRESHOLD).nonzero(as_tuple=True)[0].tolist()

    if similar_idx:
        group_indices = [i] + similar_idx
        visited.update(group_indices)

        group_df = df.iloc[group_indices][['sku', 'product_name_en', 'product_name_ar']].copy()
        group_df['group_id'] = len(groups) + 1
        group_df['similarity_ref'] = df.iloc[i]['combined_name']
        groups.append(group_df)

# ==== STEP 5: Combine and save results ====
if groups:
    result_df = pd.concat(groups, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result_df.to_excel(OUTPUT_PATH, index=False)
    print(f"✅ Found {len(groups)} duplicate groups. Saved to {OUTPUT_PATH}")
else:
    print("No duplicates found above threshold.")

predictor.cleanup()
