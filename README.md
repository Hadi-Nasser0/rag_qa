# RAG (Retrieval-Augmented Generation) System

A Python-based RAG system that uses LangChain to create a question-answering system over PDF documents. The system processes PDF files, creates vector embeddings, and uses OpenAI's GPT model to provide intelligent answers based on the document content.

## Features

- **PDF Processing**: Load and split PDF documents into manageable chunks
- **Vector Embeddings**: Create embeddings using sentence transformers
- **Vector Storage**: Store embeddings in FAISS for efficient similarity search
- **Question Answering**: Use OpenAI's GPT model to answer questions based on document context
- **Interactive Interface**: Jupyter notebook with widgets for interactive questioning
- **Flexible Architecture**: Supports both legacy and modern LangChain APIs

## Project Structure

```
rag_qa/
├── rag.ipynb              # Main Jupyter notebook with the RAG implementation
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (OpenAI API key)
├── .gitignore            # Git ignore file
├── cv.pdf               # Sample PDF document
└── faiss_index/         # FAISS vector storage directory
    ├── index.faiss      # FAISS index file
    └── index.pkl        # Pickled metadata
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag_qa
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the `rag_qa` directory
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `rag.ipynb` and run the cells in order.

3. The notebook will:
   - Load and process the PDF document
   - Create vector embeddings
   - Set up the RAG chain
   - Provide an interactive widget for asking questions

### Key Components

#### 1. Document Processing

#### 2. Vector Store Creation

#### 3. RAG Chain Setup

#### 4. Interactive Questioning
The notebook includes an interactive widget that allows you to:
- Type questions in a text input field
- Click "Ask" to get answers
- See responses displayed in real-time

## Dependencies

- **langchain**: Core LangChain framework
- **langchain-community**: Community LangChain integrations
- **langchain-core**: Core LangChain components
- **langchain-huggingface**: HuggingFace integrations
- **langchain-openai**: OpenAI integrations
- **langchain-classic**: Legacy LangChain API for backward compatibility
- **numpy**: Numerical computing
- **pypdf**: PDF processing
- **pymupdf**: Advanced PDF processing
- **faiss-cpu**: Vector similarity search
- **transformers**: HuggingFace transformers
- **torch**: Deep learning framework
- **sentence-transformers**: Sentence embeddings
- **tqdm**: Progress bars
- **ipywidgets**: Interactive widgets for Jupyter
- **python-dotenv**: Environment variable management

## Configuration

### Environment Variables

Create a `.env` file in the `rag_qa` directory:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Configuration

The system uses:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `gpt-4o-mini` (OpenAI)
- **Chunk Size**: 500 characters with 50 character overlap
- **Search Type**: Similarity search with k=3

### Main Functions

- `load_and_split_single_pdf(pdf_path)`: Loads and splits a PDF document
- `create_vector_store(chunks)`: Creates FAISS vector store from document chunks
- `create_rag_chain(vector_store, llm)`: Creates the RAG chain for question answering

## Acknowledgments

- LangChain framework for the RAG implementation
- OpenAI for the GPT model
- HuggingFace for the transformer models
- FAISS for vector similarity search
