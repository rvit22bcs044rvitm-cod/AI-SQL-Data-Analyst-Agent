import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains.sql_database.query import create_sql_query_chain

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Data Analyst Pro", layout="wide")
st.title("📊 AI SQL Data Analyst Agent")
st.markdown("#### Robust CSV → SQL → Visual Insights (Kaggle Ready)")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Priority: Secrets (for deployment) or User Input (for local testing)
    api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter Groq API Key", type="password")
    selected_model = st.selectbox("Model Selection", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"])
    st.divider()
    st.info("Built with: LangChain, Groq (Llama 3), SQLite, & Seaborn")

# --- 3. DATA PREPROCESSING LAYER (The 'Kaggle' Fixer) ---
def clean_dataframe(df):
    """Automatically makes a messy CSV safe for SQL."""
    # 1. Fill numeric NaN with 0, categorical with 'Unknown'
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('Unknown')
    
    # 2. Rename columns: Remove special chars, replace spaces with underscores
    new_cols = []
    for col in df.columns:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', col) # Remove non-alphanumeric
        clean_name = re.sub(r'_+', '_', clean_name).strip('_') # Remove double underscores
        new_cols.append(clean_name)
    df.columns = new_cols
    return df

def create_db_from_csv(df):
    conn = sqlite3.connect("analyst_data.db")
    df.to_sql("data_table", conn, index=False, if_exists='replace')
    return SQLDatabase.from_uri("sqlite:///analyst_data.db")

def sanitize_sql(raw_query):
    """Isolates the SELECT statement from LLM chatter."""
    clean = raw_query.replace("```sql", "").replace("```", "")
    if "SQLQuery:" in clean:
        clean = clean.split("SQLQuery:")[-1]
    start = clean.upper().find("SELECT")
    if start != -1:
        clean = clean[start:]
    return clean.strip().split(';')[0]

# --- 4. MAIN APPLICATION FLOW ---
uploaded_file = st.file_uploader("Upload a CSV file (Kaggle or Custom)", type=["csv"])

if uploaded_file and api_key:
    # Processing
    raw_df = pd.read_csv(uploaded_file)
    df = clean_dataframe(raw_df)
    db = create_db_from_csv(df)
    
    st.success("✅ Data Processed & SQL Table Created!")
    with st.expander("👀 Preview Cleaned Data"):
        st.dataframe(df.head(10), use_container_width=True)

    # Initialize LLM
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(model_name=selected_model, temperature=0)
    sql_chain = create_sql_query_chain(llm, db)

    # Interaction
    query = st.text_input("💬 Query your data (e.g., 'What are the top 5 records by sales?')")

    if query:
        with st.spinner("Analyzing data..."):
            try:
                # SQL Generation
                raw_sql = sql_chain.invoke({"question": query})
                clean_sql = sanitize_sql(raw_sql)
                
                # Execution
                result = db.run(clean_sql)
                
                # Layout
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("🤖 Generated SQL")
                    st.code(clean_sql, language="sql")
                    st.subheader("📝 Answer")
                    st.write(result)
                
                with col2:
                    st.subheader("📊 Visual Insight")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    # Plot the first two available columns for a quick trend view
                    sns.barplot(data=df.head(10), x=df.columns[0], y=df.columns[1], palette="magma", ax=ax)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Logic Error: {e}")
                st.warning("Try rephrasing your question or check the column names in the preview.")

elif not api_key:
    st.warning("Please provide a Groq API Key in the sidebar.")
