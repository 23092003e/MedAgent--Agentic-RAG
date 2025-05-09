# MedAgent: AI-Powered Medical Information Assistant

MedAgent is an intelligent RAG (Retrieval-Augmented Generation) system designed to provide reliable medical information based on trusted medical literature sources.

## 🌟 Key Features

- **Smart Information Retrieval**: Semantic search using FAISS vector store
- **Self-reflection**: Automatic verification and reliability checking of responses
- **Multi-format Document Processing**: Support for PDF and Word documents
- **User-friendly Interface**: Optimized UI with Streamlit
- **Parallel Processing**: Enhanced performance with concurrent operations
- **Strict Validation**: Pydantic-powered configuration validation

## 🚀 Getting Started

### System Requirements

- Python 3.8+
- Ollama
- Minimum 8GB RAM
- Storage: 10GB free space

### Installation

1. Clone repository:
```bash
git clone https://github.com/23092003e/MedAgent--Agentic-RAG.git
cd MedAgent--Agentic-RAG
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and start Ollama:
```bash
ollama serve
ollama pull medllama2
```

### Configuration

Create `.env` file in the root directory:
```env
DATA_PATH=data/raw
DB_FAISS_PATH=vectorstore/database_faiss
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=3
LLM_MODEL_NAME=medllama2
LLM_TEMPERATURE=0.5
MAX_WORKERS=4
```

### Data Preparation

1. Place medical documents (PDF/DOCX) in the `data/raw` directory

2. Create vector store:
```bash
python -m backend.rag.prepare_db
```

### Running the Application

```bash
streamlit run frontend/medibot.py
```

## 📁 Project Structure

```
medagent/
├── backend/
│   ├── rag/
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── retrieval_qa.py
│   │   ├── self_reflection.py
│   │   └── vector_store.py
│   └── config.py
├── frontend/
│   └── medibot.py
├── data/
│   └── raw/
├── vectorstore/
│   └── database_faiss/
├── requirements.txt
└── README.md
```

## 🔧 Customization

### Change Embedding Model
Update `EMBEDDING_MODEL_NAME` in `.env`:
```env
EMBEDDING_MODEL_NAME=your-preferred-model
```

### Adjust Chunking
```env
CHUNK_SIZE=1000  # Chunk size
CHUNK_OVERLAP=200  # Overlap between chunks
```

### Customize Retrieval
```env
RETRIEVAL_K=3  # Number of documents to retrieve
```

## 🛡️ Safety Notes

- MedAgent provides information for reference only
- Not a substitute for professional medical advice
- Always verify information with healthcare professionals
- Check sources and confidence scores for each response

## 🤝 Contributing

We welcome contributions! Here's how:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

- Email: hoangvanmanh2309@gmail.com

## 🙏 Acknowledgments

- LangChain team
- Ollama team
- Streamlit team
- Sentence Transformers team

---
**Disclaimer**: MedAgent is an information assistance tool only. Please consult healthcare professionals for medical decisions.



