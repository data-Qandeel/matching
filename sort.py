import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, leaves_list
from DeezyMatch import Predictor, utils
import config


def group_master_by_similarity(
    csv_path: str,
    threshold: float = 0.85,
    batch_size: int = 512,
    output_path: str = "./data/processed/master_groups.xlsx"
):
    """
    Groups products in the master file based on DeezyMatch embedding similarity.
    Ensures each product belongs to exactly one group (no duplicates).
    Then reorders groups so that similar groups appear closer to each other.
    """

    print("Loading master file...")
    df = pd.read_csv(csv_path)
    df = df.fillna("")
    df['sku'] = df['sku'].astype(str)

    if not {'product_name', 'product_name_ar'}.issubset(df.columns):
        raise ValueError("Master file must contain 'product_name' and 'product_name_ar' columns.")

    print(f"Loaded {len(df)} products")

    # Prepare texts (prefer Arabic, fallback to English, else empty)
    texts = []
    for en, ar in zip(df['product_name'], df['product_name_ar']):
        if ar.strip():
            texts.append(ar)
        elif en.strip():
            texts.append(en)
        else:
            texts.append("")

    print("Initializing DeezyMatch predictor...")
    predictor = Predictor()

    print("Encoding all product names...")
    embeddings = predictor._encode_in_batches(texts, batch_size, "Encoding master items", show_progress=True)
    embeddings = F.normalize(embeddings, dim=1)

    print("Computing cosine similarity matrix...")
    sim_matrix = cosine_similarity(embeddings.cpu().numpy())

    print(f"Grouping products with similarity ≥ {threshold}")
    n = len(df)
    visited = np.zeros(n, dtype=bool)
    groups = []
    centroids = []  # store group embeddings for reordering
    group_id = 1

    for i in tqdm(range(n)):
        if visited[i]:
            continue

        similar_idx = np.where(sim_matrix[i] >= threshold)[0]
        if i not in similar_idx:
            similar_idx = np.append(similar_idx, i)

        # Keep only unvisited
        similar_idx = [idx for idx in similar_idx if not visited[idx]]
        visited[similar_idx] = True

        group = df.iloc[similar_idx][['sku', 'product_name', 'product_name_ar']].copy()
        group['group_id'] = group_id
        groups.append(group)

        # Compute centroid embedding for this group
        group_embed = embeddings[similar_idx].mean(dim=0).cpu().numpy()
        centroids.append(group_embed)

        group_id += 1

    # Concatenate groups
    result = pd.concat(groups, ignore_index=True)

    # --- Sanity Check ---
    if len(result) != len(df):
        print(f"⚠️ WARNING: Mismatch in row count! input={len(df)}, output={len(result)}")
    else:
        print(f"✅ Row counts match: {len(df)} input → {len(result)} output")

    print(f"Created {group_id - 1} groups. Now reordering groups...")

    # --- Reorder groups using hierarchical clustering ---
    centroids = np.vstack(centroids)
    Z = linkage(centroids, method="average", metric="cosine")
    ordered_indices = leaves_list(Z)

    reordered_groups = []
    new_group_id = 1
    for idx in ordered_indices:
        group = groups[idx].copy()
        group['group_id'] = new_group_id
        reordered_groups.append(group)
        new_group_id += 1

    result = pd.concat(reordered_groups, ignore_index=True)

    print(f"✅ Groups reordered by similarity")

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_excel(output_path, index=False)
    print(f"Saved grouped results to {output_path}")

    return result


if __name__ == "__main__":
    csv_path = os.path.join(config.DATA_DIR, "raw", "master file.csv")
    group_master_by_similarity(csv_path, threshold=0.88)
