# sales_app.py
# A Streamlit app for analyzing water sports equipment store sales data
# This app allows complex queries across multiple datasets with visualizations

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
# Initialize OpenAI client with your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set page configuration
st.set_page_config(
    page_title="Water Sports Sales Analytics",
    page_icon="🏄‍♂️",
    layout="wide"
)

# Define paths to data files
DATA_DIR = "data/sales_data"
DATA_FILES = {
    "products": f"{DATA_DIR}/products.csv",
    "customers": f"{DATA_DIR}/customers.csv",
    "orders": f"{DATA_DIR}/orders.csv",
    "order_items": f"{DATA_DIR}/order_items.csv",
    "inventory": f"{DATA_DIR}/inventory.csv",
    "warehouses": f"{DATA_DIR}/warehouses.csv",
    "marketing_campaigns": f"{DATA_DIR}/marketing_campaigns.csv"
}

# Dictionary of datasets with their descriptions
DATASETS = {
    "products": "Products catalog with pricing and categories",
    "customers": "Customer information including loyalty tiers",
    "orders": "Order header information with totals and status",
    "order_items": "Line items for each order with quantities and prices",
    "inventory": "Current inventory levels across warehouses",
    "warehouses": "Warehouse information and locations",
    "marketing_campaigns": "Marketing campaign details and performance"
}

@st.cache_data
def load_all_data():
    """
    Load all sales data files into pandas DataFrames and cache the results.
    
    Returns:
        dict: Dictionary of DataFrames with the keys matching DATA_FILES
    """
    data = {}
    for name, path in DATA_FILES.items():
        try:
            data[name] = pd.read_csv(path)
            # Convert date columns to datetime
            for col in data[name].columns:
                if 'date' in col.lower() or col.lower().endswith('_at'):
                    try:
                        data[name][col] = pd.to_datetime(data[name][col])
                    except:
                        pass
        except Exception as e:
            st.error(f"Error loading {name} data: {e}")
    return data

@st.cache_data
def load_sales_data():
    """
    Load all sales data CSV files from the data/sales_data folder into pandas DataFrames 
    and return a dictionary of DataFrames.
    
    Returns:
        dict: Dictionary with dataset names as keys and pandas DataFrames as values
    """
    data_dir = "data/sales_data"
    dfs = {}
    
    # Load each CSV file from the dictionary of datasets
    for dataset in DATASETS.keys():
        file_path = f"{data_dir}/{dataset}.csv"
        try:
            dfs[dataset] = pd.read_csv(file_path)
        except Exception as e:
            st.warning(f"Could not load {dataset}.csv: {e}")
    
    return dfs

@st.cache_data
def generate_analysis_code(dfs, question, include_viz=True):
    """
    Use OpenAI to generate Python code that answers the user's question 
    using the available sales data.
    
    Parameters:
        dfs (dict): Dictionary of pandas DataFrames
        question (str): The user's natural language question
        include_viz (bool): Whether to include visualization code
    
    Returns:
        str: Generated Python code to answer the question
        str: System message sent to the LLM
        str: User message sent to the LLM
    """
    # Create schema information for each DataFrame
    schema_info = ""
    sample_data = ""
    
    for name, df in dfs.items():
        schema_info += f"\n## {name} DataFrame:\n"
        schema_info += f"Columns: {list(df.columns)}\n"
        schema_info += f"Shape: {df.shape}\n"
        
        # Add sample data for the first 3 rows
        sample_data += f"\n## First 3 rows of {name}:\n"
        sample_data += df.head(3).to_string() + "\n"
    
    # System message with more detailed instructions
    system_msg = """You are an expert data analyst specializing in e-commerce analytics.
Your task is to write Python code to answer questions about sales data from a water sports equipment store.

CRITICAL REQUIREMENTS:
1. Return ONLY valid, executable Python code. No explanations, markdown, or text outside of code comments.
2. Code must be 100% complete and ready to run without any modifications.
3. Always include all necessary merges, calculations, and processing steps.
4. Never provide just an outline or skeleton - implement every step described in comments.
5. Make sure parentheses, brackets, and braces are always properly closed.
6. Always include 'result = ' assignment for the output and 'fig = ' for visualizations.
7. Include detailed comments explaining your approach.

Technical requirements:
- Assume DataFrames are already loaded with names: products, customers, orders, order_items, inventory, warehouses, marketing_campaigns
- When joining data, use appropriate merge operations and specify relationships clearly
- Round numerical results to 2 decimal places for readability
- For visualizations, use plotly.express and store the figure in a variable called 'fig'
- For categorical data, use appropriate chart types (bar, pie, etc.)
- For time series, consider line charts or area charts
- Add proper titles, labels, and meaningful color schemes to visualizations"""

    # User message with schema, sample data and the question
    user_msg = f"""# Sales Data Schema:
{schema_info}

# Sample Data:
{sample_data}

# Data Relationships:
- orders.customer_id → customers.customer_id
- order_items.order_id → orders.order_id  
- order_items.product_id → products.product_id
- inventory.product_id → products.product_id
- inventory.warehouse_id → warehouses.warehouse_id

Question: {question}

IMPORTANT: Return ONLY complete, executable Python code with no surrounding text or markdown."""

    if not include_viz:
        user_msg += "\nDo NOT include visualization code, just provide the analysis results."
    
    # Call the chat completion endpoint on the client with higher tokens limit
    response = client.chat.completions.create(
        model="gpt-4",  # Using GPT-4 for complex analysis
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.1,
        max_tokens=2000,  # Increased token limit for full implementations
    )
    
    # Extract the code snippet from the response (strip markdown blocks if present)
    code_snippet = response.choices[0].message.content.strip()
    
    # Basic cleanup of potential markdown formatting
    if code_snippet.startswith("```python"):
        code_snippet = code_snippet[len("```python"):].strip()
    elif code_snippet.startswith("```"):
        code_snippet = code_snippet[3:].strip()
    
    if code_snippet.endswith("```"):
        code_snippet = code_snippet[:-3].strip()
    
    return code_snippet, system_msg, user_msg

def run_analysis(dfs, code_snippet):
    """
    Safely execute the generated code and return the result and any visualization.
    
    Parameters:
        dfs (dict): Dictionary of pandas DataFrames
        code_snippet (str): The code to execute
    
    Returns:
        tuple: (result, fig) where result is the analysis result and fig is any visualization
    """
    # Create a scope with all necessary libraries and data
    globals_dict = {
        'pd': pd, 
        'plt': plt, 
        'px': px, 
        'sns': sns,
        'np': __import__('numpy')
    }
    locals_dict = {**dfs}
    
    # Show the code that will be executed
    st.subheader("Generated Analysis Code:")
    st.code(code_snippet, language="python")
    
    # Execute the snippet in the local scope
    try:
        exec(code_snippet, globals_dict, locals_dict)
        
        # Look for result and visualization
        result = locals_dict.get("result", None)
        fig = locals_dict.get("fig", None)
        
        # If result is None, check for a DataFrame variable as fallback
        if result is None:
            for var_name, var_value in locals_dict.items():
                if isinstance(var_value, pd.DataFrame) and var_name not in dfs:
                    result = var_value
                    st.info(f"No 'result' variable found, using '{var_name}' as the result.")
                    break
        
        return result, fig
    except Exception as e:
        st.error(f"Error executing code: {e}")
        import traceback
        st.error(f"Error details:\n{traceback.format_exc()}")
        return None, None

def create_dataframes_info(data):
    """
    Create a string representation of available DataFrames and their schemas.
    
    Parameters:
    data (dict): Dictionary of DataFrames
    
    Returns:
    str: String describing the DataFrames
    """
    info = []
    for name, df in data.items():
        info.append(f"{name} DataFrame: {len(df)} rows x {len(df.columns)} columns")
        info.append("Columns: " + ", ".join(df.columns.tolist()))
        info.append("Sample data:")
        info.append(df.head(2).to_string())
        info.append("\n")
    
    return "\n".join(info)

# Main app logic
def main():
    st.title("📊 Water Sports Sales Analytics")
    st.write("""
    Ask natural language questions about sales data and get insights with visualizations.
    This app can analyze data across multiple tables like products, customers, orders, and marketing campaigns.
    """)
    
    # Load all data
    data = load_all_data()
    
    # Create a string representation of the DataFrames for the prompt
    dataframes_info = create_dataframes_info(data)
    
    # Show data dictionaries in an expander
    with st.expander("📚 Data Dictionary"):
        tabs = st.tabs(list(data.keys()))
        for tab, name in zip(tabs, data.keys()):
            with tab:
                st.dataframe(data[name].head())
                st.text(f"Shape: {data[name].shape}")
    
    # Load sales data
    dfs = load_sales_data()
    
    # Show available datasets
    with st.expander("Available Datasets"):
        for dataset, description in DATASETS.items():
            st.write(f"**{dataset}**: {description}")
            if dataset in dfs:
                st.write(f"Columns: {list(dfs[dataset].columns)}")
                st.dataframe(dfs[dataset].head(3))
            else:
                st.write("Dataset not loaded")
            st.markdown("---")
    
    # Example questions for users to copy-paste
    st.subheader("Example Questions (copy and paste any of these):")
    examples = [
        "What were our total sales and profits for the last 3 months?",
        "Which product categories have the highest profit margins?",
        "Who are our most loyal customers based on order frequency and total spend?",
        "Show me the seasonal sales patterns for different product categories.",
        "What products are frequently purchased together?",
    ]
    
    # Display examples as plain text in smaller font
    for example in examples:
        st.markdown(f"<span style='font-size: 0.8em;'>{example}</span>", unsafe_allow_html=True)
    
    # User input with default question
    question = st.text_input(
        "Enter your sales analysis question:", 
        value="What are our top 5 best-selling products by revenue?",
        key="question"
    )
    
    # Toggle for visualization
    include_viz = st.checkbox("Include visualization (when appropriate)", value=True)
    
    # Regular analyze button
    if st.button("Analyze"):
        if not question:
            st.error("Please enter a question.")
        else:
            with st.spinner("Generating analysis..."):
                # Generate the analysis code
                code_snippet, system_msg, user_msg = generate_analysis_code(dfs, question, include_viz)
                
                # Execute the code
                result, fig = run_analysis(dfs, code_snippet)
                
                # Display results
                if result is not None:
                    st.subheader("Analysis Result:")
                    if isinstance(result, pd.DataFrame):
                        st.dataframe(result)
                    else:
                        st.write(result)
                
                # Display visualization if available
                if fig is not None:
                    st.subheader("Visualization:")
                    st.plotly_chart(fig)
                
                # Show the full prompt sent to the LLM with nice formatting
                with st.expander("Show prompt sent to LLM"):
                    st.markdown("### System Message (Instructions to AI)")
                    st.markdown(f"```\n{system_msg}\n```")
                    
                    st.markdown("### User Message (Your Question & Data)")
                    st.markdown(f"```\n{user_msg}\n```")

# Run the app
if __name__ == "__main__":
    main() 