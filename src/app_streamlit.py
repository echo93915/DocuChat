"""Streamlit web interface for DocuChat PDF Q&A application."""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional, List
import time
from datetime import datetime

# Import DocuChat components
from .pdf_utils import process_pdf, validate_pdf_file, PDFProcessingError
from .vectorstore import build_index, get_vector_store, VectorStoreError
from .rag import answer_query, get_suggested_questions, ask, RAGError
from .settings import settings
from .types import DocumentMetadata, IndexStats

# Configure page
st.set_page_config(
    page_title="DocuChat — PDF Q&A with RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .source-snippet {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 10px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
    }
    .stat-metric {
        text-align: center;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    if 'document_metadata' not in st.session_state:
        st.session_state.document_metadata = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'suggested_questions' not in st.session_state:
        st.session_state.suggested_questions = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None


def display_header():
    """Display the main header and description."""
    st.markdown('<h1 class="main-header">📄 DocuChat — PDF Q&A with RAG</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to DocuChat!</strong> Upload a PDF document and ask questions about its content. 
        The system uses Retrieval-Augmented Generation (RAG) to provide accurate, grounded answers with source citations.
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with system information and settings."""
    with st.sidebar:
        st.header("⚙️ System Info")
        
        # Configuration display
        st.subheader("Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Embedding Model", settings.embedding_model.split('-')[-1].upper())
            st.metric("Chunk Size", settings.chunk_size)
        with col2:
            st.metric("Chat Model", settings.chat_model.split('-')[-1].upper())
            st.metric("Top K", settings.top_k)
        
        # Vector store info
        if st.session_state.document_processed:
            try:
                store = get_vector_store()
                stats = store.get_stats()
                
                st.subheader("📊 Index Stats")
                st.metric("Total Chunks", stats.total_chunks)
                st.metric("Index Size", f"{stats.index_size_mb:.2f} MB")
                st.metric("Vector Store", stats.vector_store_type.upper())
                
            except Exception as e:
                st.error(f"Error loading stats: {e}")
        
        # Document metadata
        if st.session_state.document_metadata:
            metadata = st.session_state.document_metadata
            st.subheader("📄 Document Info")
            st.metric("Filename", metadata.filename)
            st.metric("File Size", f"{metadata.file_size / 1024:.1f} KB")
            if metadata.num_pages:
                st.metric("Pages", metadata.num_pages)
            st.metric("Total Chunks", metadata.num_chunks)
            st.metric("Avg Chunk Size", f"{metadata.avg_chunk_size:.0f} chars")
        
        # Quick actions
        st.subheader("🔧 Quick Actions")
        if st.button("🗑️ Clear Chat History", help="Clear all previous questions and answers"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("🔄 Reset Application", help="Clear all data and start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def upload_and_process_section():
    """Handle PDF upload and processing."""
    st.header("📤 Upload & Process Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to start asking questions about its content"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size = len(uploaded_file.getvalue())
        st.info(f"📄 **{uploaded_file.name}** ({file_size / 1024:.1f} KB)")
        
        # Process button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            process_button = st.button(
                "🔄 Ingest Document",
                type="primary",
                help="Process the PDF and create searchable embeddings",
                disabled=st.session_state.get('processing_status') == 'processing'
            )
        
        if process_button:
            # Clear any previous status
            st.session_state.processing_status = None
            process_document(uploaded_file)


def process_document(uploaded_file):
    """Process the uploaded PDF document."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Show progress immediately
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Validate PDF
            status_text.text("Validating PDF file...")
            progress_bar.progress(10)
            if not validate_pdf_file(tmp_file_path):
                st.error("Invalid PDF file. Please upload a valid PDF document.")
                return
            
            # Process PDF
            status_text.text("Extracting text from PDF...")
            progress_bar.progress(30)
            chunks, metadata = process_pdf(
                tmp_file_path,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap
            )
            
            if not chunks:
                st.error("No text could be extracted from the PDF. Please ensure the PDF contains readable text.")
                return
            
            # Build vector index
            status_text.text("Creating searchable index...")
            progress_bar.progress(60)
            chunk_texts = [chunk.text for chunk in chunks]
            build_index(chunk_texts)
            
            # Generate suggested questions
            status_text.text("Generating suggested questions...")
            progress_bar.progress(90)
            suggestions = get_suggested_questions(chunk_texts[:5], max_suggestions=5)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Update session state
            st.session_state.document_processed = True
            st.session_state.document_metadata = metadata
            st.session_state.suggested_questions = suggestions
            st.session_state.processing_status = 'success'
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show success message
            st.success(f"Document processed successfully! Created {len(chunks)} chunks from {metadata.filename}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except Exception as e:
        st.session_state.processing_status = 'error'
        st.error(f"Error processing document: {str(e)}")
        return


def question_answering_section():
    """Handle question input and answering."""
    st.header("❓ Ask Questions")
    
    if not st.session_state.document_processed:
        st.warning("📝 Please upload and process a document first to start asking questions.")
        return
    
    # Display suggested questions
    if st.session_state.suggested_questions:
        st.subheader("💡 Suggested Questions")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, suggestion in enumerate(st.session_state.suggested_questions):
            with cols[i % len(cols)]:
                if st.button(f"💬 {suggestion}", key=f"suggestion_{i}"):
                    process_question(suggestion)
    
    # Question input
    st.subheader("💭 Ask Your Own Question")
    
    question = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?",
        help="Ask any question about the uploaded document content"
    )
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        ask_button = st.button("🔍 Ask", type="primary", disabled=not question.strip())
    
    if ask_button and question.strip():
        process_question(question.strip())


def process_question(question: str):
    """Process a user question and display the answer."""
    try:
        with st.spinner(f"🤔 Thinking about: {question}"):
            start_time = time.time()
            
            # Get answer using RAG pipeline
            result = answer_query(
                question,
                top_k=settings.top_k,
                max_tokens=settings.max_tokens_answer,
                temperature=settings.temperature
            )
            
            processing_time = time.time() - start_time
            
            # Add to chat history
            chat_entry = {
                'timestamp': datetime.now(),
                'question': question,
                'answer': result['answer'],
                'sources': result['sources'],
                'processing_time': processing_time,
                'retrieval_count': result['retrieval_count']
            }
            
            st.session_state.chat_history.append(chat_entry)
            
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return
    
    st.rerun()


def display_chat_history():
    """Display the chat history with questions and answers."""
    if not st.session_state.chat_history:
        return
    
    st.header("💬 Chat History")
    
    # Display most recent first
    for i, entry in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            # Question
            st.markdown(f"**🙋 Question #{len(st.session_state.chat_history) - i}:**")
            st.markdown(f"*{entry['question']}*")
            
            # Answer
            st.markdown("**🤖 Answer:**")
            st.markdown(entry['answer'])
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ {entry['processing_time']:.2f}s")
            with col2:
                st.caption(f"📚 {entry['retrieval_count']} sources")
            with col3:
                st.caption(f"🕒 {entry['timestamp'].strftime('%H:%M:%S')}")
            
            # Sources (expandable)
            if entry['sources']:
                with st.expander(f"📖 View Sources ({len(entry['sources'])} found)", expanded=False):
                    for j, source in enumerate(entry['sources'], 1):
                        st.markdown(f"""
                        <div class="source-snippet">
                            <strong>Source {j}:</strong><br>
                            {source[:300]}{'...' if len(source) > 300 else ''}
                        </div>
                        """, unsafe_allow_html=True)
            
            st.divider()


def display_footer():
    """Display footer information."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🚀 **DocuChat** - PDF Q&A with RAG")
    with col2:
        st.caption("🤖 Powered by OpenAI & FAISS")
    with col3:
        st.caption("⚡ Built with Streamlit")


def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Display main interface
    display_header()
    display_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        upload_and_process_section()
    
    with col2:
        question_answering_section()
    
    # Chat history (full width)
    display_chat_history()
    
    # Footer
    display_footer()


if __name__ == "__main__":
    main()
