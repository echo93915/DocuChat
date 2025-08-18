# DocuChat - PDF Q&A with RAG

A modern web application that enables users to upload PDF documents and ask natural language questions about their content. Built with Retrieval-Augmented Generation (RAG) technology for accurate, context-aware responses with multi-tier AI fallback system.

## Features

- **PDF Document Upload**: Support for PDF file processing with validation
- **Intelligent Text Processing**: Advanced text extraction and chunking for optimal retrieval
- **Semantic Search**: Vector-based similarity search using FAISS or ChromaDB
- **Multi-Tier AI System**: Google Gemini (primary) with OpenAI GPT fallback for robust responses
- **Intelligent Fallback**: Automatic failover between Gemini → OpenAI Direct → OpenAI SDK → Mock
- **Source Attribution**: All answers include references to relevant document sections
- **Interactive Web Interface**: Modern Streamlit-based UI with real-time feedback
- **Chat History**: Persistent conversation history with timestamps
- **Performance Metrics**: Processing time and retrieval statistics

## Architecture

### RAG Pipeline

```
PDF Upload → Text Extraction → Chunking → Embeddings → Vector Storage
     ↓
User Query → Embedding → Similarity Search → Context Retrieval → Answer Generation
```

### AI Fallback System

The application implements a robust multi-tier fallback system for maximum reliability:

```
1. Google Gemini (Primary)
   ↓ (if fails)
2. OpenAI Direct API
   ↓ (if fails)
3. OpenAI SDK Client
   ↓ (if fails)
4. Mock Response (for testing)
```

This ensures the application continues to function even if individual AI services experience issues.

### Unified LLM Architecture

The application features a clean, unified LLM architecture that consolidates all AI providers into a single, well-organized module:

- **Single Interface**: All LLM functionality accessible through one unified module
- **Provider Classes**: Organized provider implementations (Gemini, OpenAI Direct, OpenAI SDK, Mock)
- **Automatic Dimension Detection**: Vector store automatically adapts to different embedding dimensions (768 for Gemini, 1536 for OpenAI)
- **Backward Compatibility**: Existing import paths continue to work seamlessly
- **Maintainable**: Easier to understand, debug, and extend with new providers

### Technology Stack

- **Backend**: Python 3.10+
- **AI Models**:
  - **Primary**: Google Gemini 1.5 Flash (chat), text-embedding-004 (embeddings)
  - **Fallback**: OpenAI GPT-4o-mini (chat), text-embedding-3-small (embeddings)
- **Vector Storage**: FAISS (default) or ChromaDB with automatic embedding dimension detection
- **PDF Processing**: pypdf
- **Web Interface**: Streamlit
- **Configuration**: Pydantic settings with environment variable support

## Installation

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key (primary)
- OpenAI API key (fallback)

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/echo93915/DocuChat.git
   cd DocuChat
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   Create a `.env` file in the project root:

   ```bash
   touch .env
   ```

   Edit `.env` and add your API keys:

   ```env
   # Primary AI Service (Google Gemini)
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_CHAT_MODEL=gemini-1.5-flash
   GEMINI_EMBEDDING_MODEL=models/text-embedding-004

   # Fallback AI Service (OpenAI)
   OPENAI_API_KEY=your_openai_api_key_here
   EMBEDDING_MODEL=text-embedding-3-small
   CHAT_MODEL=gpt-4o-mini

   # System Configuration
   VECTOR_STORE=faiss
   INDEX_DIR=./storage
   CHUNK_SIZE=1200
   CHUNK_OVERLAP=200
   TOP_K=4
   MAX_TOKENS_ANSWER=600
   TEMPERATURE=0.2
   ```

## Usage

### Running the Application

Start the Streamlit web interface:

```bash
streamlit run run_app.py
```

The application will be available at `http://localhost:8501`

### Using the Interface

1. **Upload Document**: Use the file uploader to select a PDF document
2. **Process Document**: Click "Ingest Document" to extract and index the content
3. **Ask Questions**: Enter questions about the document content
4. **Review Answers**: View AI-generated responses with source citations
5. **Explore History**: Review previous questions and answers in the chat history

## Configuration

The application can be configured through environment variables:

### AI Service Configuration

| Variable                 | Default                   | Description                            |
| ------------------------ | ------------------------- | -------------------------------------- |
| `GEMINI_API_KEY`         | -                         | Google Gemini API key (required)       |
| `GEMINI_CHAT_MODEL`      | gemini-1.5-flash          | Gemini chat model                      |
| `GEMINI_EMBEDDING_MODEL` | models/text-embedding-004 | Gemini embedding model                 |
| `OPENAI_API_KEY`         | -                         | OpenAI API key (required for fallback) |
| `EMBEDDING_MODEL`        | text-embedding-3-small    | OpenAI embedding model                 |
| `CHAT_MODEL`             | gpt-4o-mini               | OpenAI chat model                      |

### System Configuration

| Variable            | Default   | Description                          |
| ------------------- | --------- | ------------------------------------ |
| `VECTOR_STORE`      | faiss     | Vector store backend (faiss/chroma)  |
| `INDEX_DIR`         | ./storage | Directory for storing vector indices |
| `CHUNK_SIZE`        | 1200      | Text chunk size in characters        |
| `CHUNK_OVERLAP`     | 200       | Overlap between chunks               |
| `TOP_K`             | 4         | Number of chunks to retrieve         |
| `MAX_TOKENS_ANSWER` | 600       | Maximum tokens for answers           |
| `TEMPERATURE`       | 0.2       | Model temperature for responses      |

## Project Structure

```
DocuChat/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── .env                     # Environment configuration
├── run_app.py               # Application entry point
├── TODO.md                  # Development roadmap
├── data/                    # Sample documents
│   └── sample.pdf
├── storage/                 # Vector indices and metadata
└── src/                     # Source code
    ├── __init__.py
    ├── app_streamlit.py     # Streamlit web interface
    ├── settings.py          # Configuration management
    ├── types.py            # Data structures
    ├── llm.py              # Backward compatibility wrapper
    ├── llm_unified.py      # Unified multi-provider LLM interface
    ├── pdf_utils.py        # PDF processing utilities
    ├── vectorstore.py      # Vector storage abstraction
    └── rag.py              # RAG pipeline implementation
```

## Development

### Testing

The project includes comprehensive test suites for each component:

```bash
# Test individual phases
python test_phase1.py  # Core infrastructure
python test_phase2.py  # Document processing
python test_phase3.py  # RAG system
python test_phase4.py  # Streamlit UI
```

### Code Quality

- Type hints throughout the codebase
- Comprehensive error handling
- Modular, extensible architecture
- Detailed logging and monitoring

## API Reference

### Core Functions

#### Document Processing

```python
from src.pdf_utils import process_pdf
chunks, metadata = process_pdf("document.pdf")
```

#### Vector Operations

```python
from src.vectorstore import build_index, search
build_index(chunk_texts)
results = search("query", k=5)
```

#### RAG Pipeline

```python
from src.rag import answer_query
response = answer_query("What is this document about?")
```

#### LLM Operations

```python
from src.llm_unified import embed_texts, chat_complete
embeddings = embed_texts(["sample text"])
response = chat_complete("You are a helpful assistant", "Hello!")
```

## Troubleshooting

### Common Issues

**Installation Problems**

- Ensure Python 3.10+ is installed
- Try upgrading pip: `pip install --upgrade pip`
- Install dependencies one by one if batch install fails

**API Key Issues**

- Verify both Gemini and OpenAI API keys are correctly set in `.env`
- Check API keys have sufficient credits/quota
- Ensure no extra spaces in the environment file
- Test individual API services to isolate issues

**PDF Processing Errors**

- Verify PDF is not password-protected
- Ensure PDF contains extractable text (not scanned images)
- Check file size is reasonable (< 50MB recommended)

**Performance Issues**

- Reduce `CHUNK_SIZE` for faster processing
- Decrease `TOP_K` for quicker retrieval
- Use FAISS instead of ChromaDB for better performance

**Embedding Dimension Issues**

- The system automatically detects embedding dimensions (768 for Gemini, 1536 for OpenAI)
- If you encounter FAISS dimension errors, clear the storage directory: `rm -rf storage/*`
- The vector store will rebuild with the correct dimensions for your current provider

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is available under the MIT License. See LICENSE file for details.

## Acknowledgments

- Google for providing the Gemini AI platform and APIs
- OpenAI for providing the GPT and embedding models as fallback
- FAISS team for the efficient vector search library
- Streamlit for the excellent web framework
- pypdf contributors for PDF processing capabilities

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/echo93915/DocuChat) or open an issue.
