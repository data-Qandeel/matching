import pandas as pd
from DeezyMatch import Predictor, utils

# Load master data
print("Loading master catalog data...")
master_file = pd.read_csv('data/raw/master_file_activated.csv')
master_file['sku'] = master_file['sku'].astype(str)
print(f"Master catalog loaded with {master_file.shape[0]} products")

# Load input data
print("Loading input data...")
file_path = './data/test/EAGLE Pharma.xlsx'
df = pd.read_excel(file_path)
print(f'Dataset size: {df.shape[0]} entries')

# Define column names for matching
seller_item_name = 'item_name'
seller_item_price = 'price'

# Prepare queries and candidates
print("Preparing queries and candidates...")
queries = df[seller_item_name].apply(utils.minimal_clean_text)
query_prices = df[seller_item_price]

# Combine Arabic and English candidates
candidates = pd.concat([
    master_file['product_name'],
    master_file['product_name_ar']
]).apply(utils.minimal_clean_text)
candidate_prices = pd.concat([master_file['price'], master_file['price']])

print(f"Prepared {len(queries)} queries and {len(candidates)} candidates")

# Initialize predictor
print("Initializing DeezyMatch predictor...")
p = Predictor()

# Perform matching for all entries
print("Matching entries...")
scores = p.rank_candidates(
    queries=queries,
    candidates=candidates,
    batch_size=1024,
    top_k=3,
    show_progress=True,
    query_prices=query_prices,
    candidate_prices=candidate_prices,
    price_weight=0.20,
    price_bandwidth=1.0
)
print(f"Matching completed")

# Add candidates to dataframe
k = 2
print(f"Adding top {k} candidates to dataframe...")
df_results = utils.add_top_k_candidates_to_df(
    df, scores, master_file, k=k,
    include_cols=['sku', 'pred', 'sim']
)

# Process results
print("Processing results...")
review, no_review = utils.segment_for_review(
    df_results, quantile=0.10, sim1_threshold=0.92
)


auto_match_percent = round(no_review.shape[0] / df_results.shape[0] * 100, 2)
review_percent = round(review.shape[0] / df_results.shape[0] * 100, 2)
print(f"Results: {auto_match_percent}% ({no_review.shape[0]} entries) can be automatically matched")
print(f"Results: {review_percent}% ({review.shape[0]} entries) need manual review")

# save results
utils.save_dataframe_to_xlsx(no_review, file_path, 'no_review')
utils.save_dataframe_to_xlsx(review, file_path, 'review')