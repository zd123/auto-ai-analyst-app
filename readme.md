# AI-Powered Sales Analytics App

## Overview
This Streamlit application allows users to ask natural language questions about water sports equipment sales data and receive detailed analyses with visualizations. The app leverages the power of GPT-4 to translate natural language questions into executable Python code that analyzes multiple datasets.

## How to Run the App

### Prerequisites
- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone or download this repository**

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key**
   Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the app** in your web browser at http://localhost:8501

## How the AI Works

### 1. Prompt Engineering
The app uses careful prompt engineering to instruct the GPT-4 model. The prompt consists of:

- **System Message**: Contains detailed instructions for the AI to act as a data analyst, specifying the output should be only valid Python code with proper comments.

- **User Message**: Contains the schema information for all datasets (column names, data types), sample data, relationship mappings between tables, and the user's natural language question.

Here's how the app formats the prompt:
```python
# System message with detailed instructions
system_msg = """You are an expert data analyst specializing in e-commerce analytics.
Your task is to write Python code to answer questions about sales data...
"""

# User message with schema, sample data and the question
user_msg = f"""# Sales Data Schema:
{schema_info}

# Sample Data:
{sample_data}

# Data Relationships:
- orders.customer_id → customers.customer_id
...

Question: {question}
"""
```

### 2. AI Code Generation
When a user submits a question, the app:
1. Constructs the detailed prompt with all necessary context
2. Sends the prompt to OpenAI's API
3. Receives the generated Python code
4. Cleans up the code (removing Markdown formatting if present)

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ],
    temperature=0.1,
    max_tokens=2000,
)
```

### 3. Safe Code Execution
The app then executes the generated code in a controlled environment:

1. Creates a safe execution scope with necessary libraries (pandas, matplotlib, etc.)
2. Executes the code using Python's `exec()` function
3. Captures the results and any visualizations
4. Handles exceptions gracefully

```python
def run_analysis(dfs, code_snippet):
    # Create a scope with necessary libraries
    globals_dict = {'pd': pd, 'plt': plt, 'px': px, 'sns': sns, 'np': __import__('numpy')}
    locals_dict = {**dfs}
    
    # Execute the code safely
    try:
        exec(code_snippet, globals_dict, locals_dict)
        # Get results and visualizations
        result = locals_dict.get("result", None)
        fig = locals_dict.get("fig", None)
        return result, fig
    except Exception as e:
        # Handle errors
        st.error(f"Error executing code: {e}")
        return None, None
```

## Educational Purposes
This app demonstrates several powerful concepts:
1. Using LLMs to generate executable code from natural language
2. Safe execution of AI-generated code
3. Creating interactive data analysis workflows
4. Dynamic visualization generation

The code is extensively commented to serve as a teaching tool for those interested in building AI-powered data analysis applications.

## Limitations
- The app requires an OpenAI API key
- Complex questions might result in code that takes longer to execute
- The AI might occasionally generate code that doesn't perfectly address the question



## Data Structure
The app uses several CSV files located in the `data/sales_data/` directory:
- products.csv - Product catalog with pricing and categories
- customers.csv - Customer information including loyalty tiers
- orders.csv - Order header information with totals and status
- order_items.csv - Line items for each order with quantities and prices
- inventory.csv - Current inventory levels across warehouses
- warehouses.csv - Warehouse information and locations
- marketing_campaigns.csv - Marketing campaign details and performance


## License
Free.99

