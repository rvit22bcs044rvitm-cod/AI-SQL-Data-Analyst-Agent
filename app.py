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
st.markdown("#### Robust CSV → SQL → Visual Insights")

# --- 2. SIDEBAR (Secret Management) ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Checks Streamlit Secrets first, then falls back to manual input
    api_key = st.secrets.get("GROQ_API_KEY") or st.text_input("Enter Groq API Key", type="password")
    selected_model = st.selectbox("Model Selection", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"])
    st.divider()
    st.info("Built with: LangChain, Groq (Llama 3.1), SQLite, & Seaborn")

# --- 3. DATA PREPROCESSING (The Kaggle Fixer) ---
def clean_dataframe(df):
    """Automatically makes a messy CSV safe for SQL."""
    # Fill numeric NaN with 0, categorical with 'Unknown'
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna('Unknown')
    
    # Rename columns: Remove special chars, replace spaces with underscores
    new_cols = []
    for col in df.columns:
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', col)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
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
    # Processing the file
    raw_df = pd.read_csv(uploaded_file)
    df = clean_dataframe(raw_df)
    db = create_db_from_csv(df)
    
    st.success("✅ Data Processed & SQL Table Created!")
    with st.expander("👀 Preview Cleaned Data"):
        st.dataframe(df.head(10), use_container_width=True)

    # Initialize LLM Agent
    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(model_name=selected_model, temperature=0)
    sql_chain = create_sql_query_chain(llm, db)

    # Interaction
    query = st.text_input("💬 Query your data (e.g., 'What are the top 5 records by sales?')")

    if query:
        with st.spinner("Analyzing data..."):
            try:
                # 1. SQL Generation
                raw_sql = sql_chain.invoke({"question": query})
                clean_sql = sanitize_sql(raw_sql)
                
                # 2. Execution
                # We execute the query and convert the result to a DataFrame for plotting
                conn = sqlite3.connect("analyst_data.db")
                result_df = pd.read_sql_query(clean_sql, conn)
                
                # 3. Display Results
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader("🤖 Generated SQL")
                    st.code(clean_sql, language="sql")
                    st.subheader("📝 Answer")
                    st.dataframe(result_df, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Visual Insight")
                    if not result_df.empty:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        
                        # DYNAMIC PLOTTING LOGIC
                        if len(result_df.columns) >= 2:
                            # Plot the first two columns of the RESULT (not the whole CSV)
                            sns.barplot(data=result_df, x=result_df.columns[0], y=result_df.columns[1], palette="magma", ax=ax)
                        else:
                            # Single column result: just count frequencies
                            sns.countplot(data=result_df, x=result_df.columns[0], palette="magma", ax=ax)
                        
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    else:
                        st.info("No data found to plot for this specific query.")
                    
            except Exception as e:
                st.error(f"Error: {e}")

elif not api_key:
    st.warning("Please provide a Groq API Key (check Sidebar or Streamlit Secrets).")
