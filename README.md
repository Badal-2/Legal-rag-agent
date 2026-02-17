## 🚀 Features

- Upload PDF legal documents
- Ask questions in natural language
- Get accurate answers with source citations
- Extract key clauses automatically
- Generate document summaries

## 🛠️ Tech Stack

- **LLM:** Google Gemini 2.5 Flash
- **Embeddings:** Google Gemini embedding-001
- **Vector DB:** ChromaDB
- **Framework:** LangChain
- **Backend:** FastAPI
- **Frontend:** HTML + CSS + JavaScript

## 📋 RAG Workflow
```
PHASE 1 - Document Storage:
PDF → Extract Text → Chunk → Gemini Embeddings → ChromaDB

PHASE 2 - Question Answering:
Question → Gemini Embeddings → Semantic Search → Top K Chunks → LLM → Answer 




## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/legal-rag-agent.git
cd legal-rag-agent
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables
```bash
# Create .env file
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Run the application
```bash
python main.py
```

### 6. Open in browser
```
http://localhost:8000
```

## 📁 Project Structure
```
legal_rag_agent/
├── tools/
│   ├── pdf_extractor.py    # Extract text from PDF
│   ├── chunking_tool.py    # Split text into chunks
│   └── vector_store.py     # ChromaDB operations
├── agent.py                # RAG Agent (LangChain + Gemini)
├── main.py                 # FastAPI backend
├── index.html              # Frontend UI
├── requirements.txt        # Dependencies
└── .env                    # API keys (not committed)
```

## 🔑 Environment Variables
```env
GOOGLE_API_KEY=your_key_here
```

## 📝 License
MIT License
