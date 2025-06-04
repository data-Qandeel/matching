import os
import tempfile
import pandas as pd
import gradio as gr
from DeezyMatch import Predictor, utils

# Configuration parameters
CONFIG = {
    'master_file': 'data/raw/master_file_activated.csv',
    'model': {
        'batch_size': 256,
        'top_k': 5,
        'price_weight': 0.20,
        'price_bandwidth': 1.0
    },
    'review': {
        'quantile': 0.10,
        'sim1_threshold': 0.92
    },
    'defaults': {
        'item_name_col': 'item_name',
        'price_col': 'price',
        'k_value': 2,
        'use_price': True
    }
}

# File handling functions
def read_master_file():
    """Read the master catalog file"""
    master_path = os.path.join(os.path.dirname(__file__), CONFIG['master_file'])
    if not os.path.exists(master_path):
        raise ValueError(f"Master file not found at {master_path}")
    return pd.read_csv(master_path)

def get_excel_sheets(file):
    """Get sheet names from Excel file"""
    if file is None or not file.name.endswith(('.xlsx', '.xls')):
        return []
    return pd.ExcelFile(file.name).sheet_names

def read_file(file_path, sheet_name=None):
    """Read CSV or Excel file based on extension"""
    if isinstance(file_path, str):
        return pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path, sheet_name=sheet_name)
    elif file_path.name.endswith('.csv'):
        return pd.read_csv(file_path.name)
    elif file_path.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path.name, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel file.")

def get_file_columns(file, sheet_name=None):
    """Get column names from uploaded file"""
    if file is None:
        return [], []
    df = read_file(file, sheet_name)
    return df.columns.tolist(), df.columns.tolist()

# Main processing function
def process_files(test_file, sheet_name, k_value, item_name_col, price_col, use_price):
    """Process the files using DeezyMatch model"""
    try:
        # Read files
        df_master = read_master_file()
        df_test = read_file(test_file, sheet_name)
        if 'sku' in df_master.columns:
            df_master['sku'] = df_master['sku'].astype(str)

        # Prepare data
        queries = df_test[item_name_col].apply(utils.minimal_clean_text)
        candidates = pd.concat([
            df_master['product_name'],
            df_master['product_name_ar']
        ]).apply(utils.minimal_clean_text)

        # Handle price matching
        query_prices = None
        candidate_prices = None
        if use_price and price_col:
            if price_col in df_test.columns and 'price' in df_master.columns:
                query_prices = df_test[price_col]
                candidate_prices = pd.concat([df_master['price'], df_master['price']])
            else:
                print("Warning: Price columns not found, continuing without price matching")
                use_price = False

        # Process matches
        p = Predictor()
        scores = p.rank_candidates(
            queries=queries,
            candidates=candidates,
            batch_size=CONFIG['model']['batch_size'],
            top_k=CONFIG['model']['top_k'],
            show_progress=True,
            query_prices=query_prices if use_price else None,
            candidate_prices=candidate_prices if use_price else None,
            price_weight=CONFIG['model']['price_weight'] if use_price else 0.0,
            price_bandwidth=CONFIG['model']['price_bandwidth']
        )

        # Process results
        df_results = utils.add_top_k_candidates_to_df(
            df_test, scores, df_master, k=k_value,
            include_cols=['sku', 'pred', 'sim']
        )

        review, no_review = utils.segment_for_review(
            df_results, 
            quantile=CONFIG['review']['quantile'], 
            sim1_threshold=CONFIG['review']['sim1_threshold']
        )

        # Prepare output
        review_ratio = round(no_review.shape[0] / df_results.shape[0], 4)
        dataset_info = f"Dataset size: {df_test.shape[0]}\n"
        stats_info = (
            f"No Review: {no_review.shape[0]} items ({review_ratio * 100:.2f}%)\n"
            f"Needs Review: {review.shape[0]} items ({(1-review_ratio) * 100:.2f}%)"
        )

        # Save results
        original_name = os.path.splitext(test_file.name)[0]
        result_filename = f"{original_name}_results.xlsx"
        output_path = os.path.join(tempfile.gettempdir(), result_filename)

        # Use the original sheet name or 'Sheet1' if it's a CSV file
        original_sheet_name = sheet_name if sheet_name else 'Sheet1'

        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_test.to_excel(writer, sheet_name=original_sheet_name, index=False)
            no_review.to_excel(writer, sheet_name='No_Review', index=False)
            review.to_excel(writer, sheet_name='Review', index=False)

        return dataset_info, stats_info, no_review, review, output_path

    except Exception as e:
        return f"Error: {str(e)}", "", pd.DataFrame(), pd.DataFrame(), None

# UI Components
def create_ui():
    """Create the Gradio interface"""
    with gr.Blocks(title="DeezyMatch Product Matching") as app:
        gr.Markdown("# DeezyMatch Product Matching Tool")
        gr.Markdown("Upload a test file to match against the master catalog")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"Using master catalog: {CONFIG['master_file']}")
                test_file = gr.File(label="Upload Test File (CSV/Excel)")
                
                sheet_selector = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Excel Sheet",
                    visible=False
                )
                
                k_value = gr.Slider(
                    minimum=1, 
                    maximum=CONFIG['model']['top_k'],
                    value=CONFIG['defaults']['k_value'],
                    step=1,
                    label="Number of predictions"
                )
                
                use_price = gr.Checkbox(
                    label="Use Price Matching",
                    value=CONFIG['defaults']['use_price']
                )
                
                item_name_col = gr.Dropdown(
                    choices=[],
                    value=CONFIG['defaults']['item_name_col'],
                    label="Item Name Column",
                    allow_custom_value=True,
                    interactive=True
                )
                price_col = gr.Dropdown(
                    choices=[],
                    value=CONFIG['defaults']['price_col'],
                    label="Price Column",
                    allow_custom_value=True,
                    interactive=True,
                    visible=True
                )
                
                process_btn = gr.Button("Process Files", variant="primary")

        with gr.Row():
            with gr.Column():
                dataset_info = gr.Textbox(label="Dataset Information", lines=3)
                stats_info = gr.Textbox(label="Matching Statistics", lines=3)
        
        with gr.Tabs():
            with gr.TabItem("No Review Preview"):
                no_review_preview = gr.DataFrame()
            with gr.TabItem("Review Preview"):
                review_preview = gr.DataFrame()
        
        output_file = gr.File(label="Download Results")

        # Event handlers
        def update_sheet_selector(file):
            if file and file.name.endswith(('.xlsx', '.xls')):
                sheets = get_excel_sheets(file)
                return gr.update(choices=sheets, value=sheets[0] if sheets else None, visible=True)
            return gr.update(choices=[], value=None, visible=False)
        
        def update_columns(file, sheet):
            if file is None:
                return gr.update(choices=[]), gr.update(choices=[])
            
            cols, _ = get_file_columns(file, sheet)
            
            # Set smart defaults for columns
            default_item = (
                CONFIG['defaults']['item_name_col'] if CONFIG['defaults']['item_name_col'] in cols 
                else cols[0] if cols else None
            )
            
            default_price = (
                CONFIG['defaults']['price_col'] if CONFIG['defaults']['price_col'] in cols
                else cols[1] if len(cols) > 1 else cols[0] if cols else None
            )
            
            return (
                gr.update(choices=cols, value=default_item),
                gr.update(choices=cols, value=default_price)
            )

        # Event handler bindings
        test_file.change(
            fn=update_sheet_selector,
            inputs=[test_file],
            outputs=[sheet_selector]
        )
        
        sheet_selector.change(
            fn=update_columns,
            inputs=[test_file, sheet_selector],
            outputs=[item_name_col, price_col]
        )
        
        use_price.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_price],
            outputs=[price_col]
        )
        
        process_btn.click(
            fn=process_files,
            inputs=[test_file, sheet_selector, k_value, item_name_col, price_col, use_price],
            outputs=[dataset_info, stats_info, no_review_preview, review_preview, output_file]
        )
        
        return app

if __name__ == "__main__":
    app = create_ui()
    app.launch()