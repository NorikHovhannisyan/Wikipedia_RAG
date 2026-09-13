# Wikipedia RAG (Retrieval-Augmented Generation) System

A complete RAG pipeline built with Python that scrapes data from Wikipedia, processes and embeds text chunks into a Vector Database, and queries an LLM using Groq API to answer user questions.

## 📌 Features
- **Data Scraping & Preprocessing:** Extracts Python Wikipedia article using BeautifulSoup and cleans text with Pandas.
- **Vector Embeddings:** Generates 384-dimensional dense vectors using `SentenceTransformer` (`all-MiniLM-L6-v2`).
- **Vector Storage:** Stores document vectors and metadata in `ChromaDB` for fast similarity search.
- **LLM Integration:** Uses Groq API (`openai/gpt-oss-20b`) to provide accurate, context-based answers in Armenian.

## 🛠️ Project Structure
```text
├── Data_Scraping.py        # Web scraping logic
├── Data_Preprocessing.py   # Text cleaning and filtering
├── Gen_Embedding.py        # Sentence transformer embedding generation
├── Save_to_Vector.py       # ChromaDB vector initialization and persistent storage
├── RAG_app.py              # Main RAG execution file with Groq LLM
├── requirements.txt        # Core project dependencies
└── .env                    # API Keys (not tracked by git)
```
## 🚀 Getting Started
### 1. Clone the repository
```
git clone [https://github.com/NorikHovhannisyan/Wikipedia_RAG.git](https://github.com/NorikHovhannisyan/Wikipedia_RAG.git)
cd Wikipedia_RAG
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
### 3. Environment Setup
Create a .env file in the root directory and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```
## 4. Run the Application
```
python RAG_app.py
```
