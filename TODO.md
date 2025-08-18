# DocuChat - PDF Q&A with RAG - TODO List

## Project Overview

Building a **PDF Q&A system with RAG (Retrieval-Augmented Generation)** that allows users to:

- Upload PDFs and create searchable embeddings
- Ask questions about the document content
- Get AI-generated answers with source citations
- View retrieved document snippets

## Current Progress ✅

- ✅ **Project structure** created (`src/`, `data/`, `storage/` directories)
- ✅ **Dependencies** defined in `requirements.txt`
- ✅ **Environment template** created (`.env.example`)

## Detailed TODO Breakdown

### **Phase 1: Core Infrastructure** 🏗️

#### 1. Configuration Management (`src/settings.py`) - ✅ **COMPLETED**

- [x] Create Pydantic settings class for environment variables
- [x] Add type validation for vector store options (faiss|chroma)
- [x] Implement default values and configuration loading
- [x] Expose module-level `settings = Settings()` instance

#### 2. OpenAI Integration (`src/llm.py`) - ✅ **COMPLETED**

- [x] Create `embed_texts(texts: list[str]) -> list[list[float]]` wrapper
- [x] Create `chat_complete(system: str, user: str, *, max_tokens: int, temperature: float) -> str` wrapper
- [x] Add tenacity retry logic with exponential backoff
- [x] Handle rate limiting and API errors gracefully
- [x] Respect model names from settings
- [x] Create mock implementation for testing (`llm_mock.py`)

#### 3. Data Types (`src/types.py`) - ✅ **COMPLETED**

- [x] Define Document chunk data structures
- [x] Create RetrievalResult classes
- [x] Define Answer response models
- [x] Add type hints for better code clarity
- [x] Add DocumentMetadata and IndexStats classes

### **Phase 2: Document Processing** 📄

#### 4. PDF Processing (`src/pdf_utils.py`) - PENDING

- [ ] Implement `extract_text(pdf_path: str) -> str` using pypdf.PdfReader
- [ ] Create `chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]`
- [ ] Add text cleaning and preprocessing
- [ ] Handle empty chunks and whitespace stripping

#### 5. Vector Storage (`src/vectorstore.py`) - PENDING

- [ ] Create unified interface for FAISS and Chroma backends
- [ ] Implement `build_index(chunks: list[str]) -> None`
  - [ ] Compute embeddings with `embed_texts`
  - [ ] Persist index to INDEX_DIR
  - [ ] Save ordered chunk texts mapping
- [ ] Implement `search(query: str, k: int) -> list[str]`
  - [ ] Embed query and perform ANN search
  - [ ] Return chunk strings
- [ ] Add FAISS metadata handling with chunks.txt

### **Phase 3: RAG Pipeline** 🧠

#### 6. RAG System (`src/rag.py`) - PENDING

- [ ] Implement `retrieve(query: str, top_k: int) -> list[str]`
- [ ] Create `answer_query(query: str) -> dict` function
  - [ ] Retrieve top-k chunks
  - [ ] Compose grounded prompt with context
  - [ ] Call chat_complete with proper parameters
  - [ ] Return dict with answer and sources
- [ ] Design effective system prompt for grounded responses

### **Phase 4: User Interface** 🖥️

#### 7. Streamlit App (`src/app_streamlit.py`) - PENDING

- [ ] Set up page title: "DocuChat — PDF Q&A with RAG"
- [ ] Create **Upload & Ingest** section:
  - [ ] PDF file uploader widget
  - [ ] "Ingest" button with processing logic
  - [ ] Success message with chunk count
- [ ] Create **Ask a Question** section:
  - [ ] Text input widget
  - [ ] "Ask" button with query processing
  - [ ] Chat-style answer display
  - [ ] Expandable "Sources" section
- [ ] Add UX polish:
  - [ ] Loading spinners during processing
  - [ ] Error handling for missing index
  - [ ] User guidance messages

### **Phase 5: Testing & Quality** 🧪

#### 8. Evaluation Tools (`src/eval_utils.py`) - PENDING

- [ ] Create `smoke_test()` function
- [ ] Build index for sample PDF
- [ ] Run canned queries for testing
- [ ] Print answer lengths and source counts
- [ ] Add performance benchmarking utilities

#### 9. Sample Data & Testing - PENDING

- [ ] Add sample PDF to `data/` directory
- [ ] Create comprehensive end-to-end tests
- [ ] Test error scenarios (empty PDF, API failures)
- [ ] Validate RAG pipeline accuracy

### **Phase 6: Documentation & Polish** ✨

#### 10. Documentation (`README.md`) - PENDING

- [ ] Write project overview with features
- [ ] Create architecture diagram of RAG flow
- [ ] Add setup instructions:
  - [ ] Virtual environment creation
  - [ ] Dependency installation
  - [ ] Environment configuration
- [ ] Document usage instructions
- [ ] Add troubleshooting section
- [ ] Include next steps and future enhancements

#### 11. Final Polish - PENDING

- [ ] Comprehensive error handling throughout
- [ ] UI/UX improvements and responsiveness
- [ ] Performance optimizations
- [ ] Code documentation and docstrings
- [ ] Final testing and validation

## Technical Specifications

### **Tech Stack**

- **Python 3.10+**
- **OpenAI API** (openai SDK v1.x)
  - Chat model: `gpt-4o-mini`
  - Embeddings: `text-embedding-3-small`
- **Vector store:** FAISS (default); optional Chroma
- **PDF parsing:** `pypdf`
- **UI:** Streamlit
- **Dependencies:** `pydantic`, `python-dotenv`, `numpy`, `pandas`, `tenacity`

### **Project Structure**

```
docuchat/
├── README.md
├── .env.example
├── requirements.txt
├── TODO.md
├── data/
│   └── sample.pdf
├── storage/                 # FAISS index / metadata
└── src/
    ├── __init__.py
    ├── settings.py          # Environment config
    ├── llm.py              # OpenAI API wrappers
    ├── pdf_utils.py        # PDF processing
    ├── vectorstore.py      # Vector storage
    ├── rag.py              # RAG pipeline
    ├── app_streamlit.py    # Streamlit UI
    ├── types.py            # Data structures
    └── eval_utils.py       # Testing utilities
```

### **RAG Flow**

1. **Upload PDF** → extract text → chunk
2. **Embed chunks** via OpenAI → write FAISS index + chunks.txt
3. **User question** → embed query → ANN search → top-k chunks
4. **Construct prompt** → call gpt-4o-mini → display answer + sources

### **Environment Configuration**

```bash
OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
VECTOR_STORE=faiss
INDEX_DIR=./storage
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=4
MAX_TOKENS_ANSWER=600
TEMPERATURE=0.2
```

### **Acceptance Criteria**

1. ✅ Upload PDF and see successful ingestion with chunk count
2. ✅ Ask questions and receive grounded answers
3. ✅ View expandable source snippets
4. ✅ Graceful error handling (API errors, missing index, empty PDF)
5. ✅ Modular, typed code with proper documentation
6. ✅ Complete setup documentation

### **Nice-to-haves**

- [ ] Token usage display in footer
- [ ] Quick evaluation with preset Q&A pairs
- [ ] Highlight matched terms in source snippets
- [ ] Multi-document support
- [ ] Page number citations
- [ ] Reranking for better retrieval

---

**Last Updated:** Phase 1 completed and tested
**Status:** ✅ Phase 1 (Core Infrastructure) COMPLETE - Ready for Phase 2
