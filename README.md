# AI SQL Data Analyst Agent 🤖
### CSV → SQL → Visual Insights

An advanced AI-powered agent built to democratize data analysis. This tool allows anyone to interact with complex datasets using natural language. It automatically handles the conversion of raw CSV files into a structured SQL database, generates executable queries, and visualizes the results.

## 🌟 Key Features
* **Kaggle-Ready Pipeline**: Includes an automated data cleaning layer that sanitizes messy headers, handles special characters, and manages missing values using Regex and Pandas.
* **Real-Time Text-to-SQL**: Leverages the **Groq Llama 3.1** model for near-instant translation of English questions into optimized SQL code.
* **Interactive Visualization**: Generates dynamic charts using **Seaborn** to provide immediate visual feedback on data trends.

## 🏗️ System Architecture
1. **Frontend**: Streamlit Dashboard
2. **Orchestrator**: LangChain (SQL Chain)
3. **Brain**: Groq API (Llama 3.1 8B / 70B)
4. **Storage**: SQLite (In-memory/Local)
5. **Data Engine**: Pandas & Regex

## 🛠️ Tech Stack
* **Language**: Python 3.12
* **Libraries**: `langchain-groq`, `langchain-classic`, `pandas`, `sqlite3`, `matplotlib`, `seaborn`

## 🚦 How to Run Locally
1. Clone the repo: `git clone <your-repo-link>`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
