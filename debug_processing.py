"""Debug script to identify document processing issues."""

import sys
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.append('src')

def debug_pdf_processing():
    """Debug PDF processing step by step."""
    print("=== DEBUG: PDF Processing ===")
    
    try:
        from src.pdf_utils import validate_pdf_file, extract_text, chunk_text, process_pdf
        
        # Check if sample PDF exists
        sample_pdf = "data/sample.pdf"
        if not Path(sample_pdf).exists():
            print(f"ERROR: Sample PDF not found at {sample_pdf}")
            return False
        
        print(f"✅ Sample PDF found: {sample_pdf}")
        
        # Test validation
        print("Testing PDF validation...")
        start_time = time.time()
        is_valid = validate_pdf_file(sample_pdf)
        print(f"  PDF validation: {is_valid} ({time.time() - start_time:.2f}s)")
        
        if not is_valid:
            print("ERROR: PDF validation failed")
            return False
        
        # Test text extraction
        print("Testing text extraction...")
        start_time = time.time()
        text = extract_text(sample_pdf)
        print(f"  Text extracted: {len(text)} chars ({time.time() - start_time:.2f}s)")
        print(f"  Preview: {text[:100]}...")
        
        # Test chunking
        print("Testing text chunking...")
        start_time = time.time()
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        print(f"  Chunks created: {len(chunks)} ({time.time() - start_time:.2f}s)")
        
        # Test full pipeline
        print("Testing full PDF processing pipeline...")
        start_time = time.time()
        all_chunks, metadata = process_pdf(sample_pdf, chunk_size=200, overlap=50)
        print(f"  Full pipeline: {len(all_chunks)} chunks ({time.time() - start_time:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"ERROR in PDF processing: {e}")
        traceback.print_exc()
        return False

def debug_vector_store():
    """Debug vector store operations."""
    print("\n=== DEBUG: Vector Store ===")
    
    try:
        from src.vectorstore import build_index, get_vector_store
        
        # Test with simple chunks
        test_chunks = [
            "This is a test document about DocuChat.",
            "DocuChat is a PDF Q&A system.",
            "It uses RAG technology for answers."
        ]
        
        print("Testing vector store creation...")
        start_time = time.time()
        store = get_vector_store()
        print(f"  Vector store created ({time.time() - start_time:.2f}s)")
        
        print("Testing index building...")
        start_time = time.time()
        build_index(test_chunks)
        print(f"  Index built ({time.time() - start_time:.2f}s)")
        
        print("Testing search...")
        start_time = time.time()
        results = store.search("What is DocuChat?", k=2)
        print(f"  Search completed: {len(results)} results ({time.time() - start_time:.2f}s)")
        
        return True
        
    except Exception as e:
        print(f"ERROR in vector store: {e}")
        traceback.print_exc()
        return False

def debug_embeddings():
    """Debug embedding generation."""
    print("\n=== DEBUG: Embeddings ===")
    
    try:
        from src.llm_mock import embed_texts
        
        test_texts = ["Hello world", "This is a test"]
        
        print("Testing embedding generation...")
        start_time = time.time()
        embeddings = embed_texts(test_texts)
        print(f"  Embeddings generated: {len(embeddings)} ({time.time() - start_time:.2f}s)")
        
        if embeddings and embeddings[0]:
            print(f"  First embedding dimension: {len(embeddings[0])}")
        else:
            print("  WARNING: Empty embeddings returned")
        
        return True
        
    except Exception as e:
        print(f"ERROR in embeddings: {e}")
        traceback.print_exc()
        return False

def debug_full_pipeline():
    """Debug the complete pipeline that Streamlit uses."""
    print("\n=== DEBUG: Full Pipeline (Streamlit Simulation) ===")
    
    try:
        # Import what Streamlit uses
        from src.pdf_utils import process_pdf, validate_pdf_file
        from src.vectorstore import build_index
        from src.rag import get_suggested_questions
        
        sample_pdf = "data/sample.pdf"
        
        if not Path(sample_pdf).exists():
            print("ERROR: No sample PDF for full pipeline test")
            return False
        
        print("Step 1: Validate PDF...")
        start_time = time.time()
        if not validate_pdf_file(sample_pdf):
            print("ERROR: PDF validation failed")
            return False
        print(f"  ✅ PDF validated ({time.time() - start_time:.2f}s)")
        
        print("Step 2: Process PDF...")
        start_time = time.time()
        chunks, metadata = process_pdf(sample_pdf, chunk_size=1200, overlap=200)
        print(f"  ✅ PDF processed: {len(chunks)} chunks ({time.time() - start_time:.2f}s)")
        
        print("Step 3: Build vector index...")
        start_time = time.time()
        chunk_texts = [chunk.text for chunk in chunks]
        build_index(chunk_texts)
        print(f"  ✅ Index built ({time.time() - start_time:.2f}s)")
        
        print("Step 4: Generate suggestions...")
        start_time = time.time()
        suggestions = get_suggested_questions(chunk_texts[:5], max_suggestions=3)
        print(f"  ✅ Suggestions generated: {len(suggestions)} ({time.time() - start_time:.2f}s)")
        
        print("\n🎉 Full pipeline completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR in full pipeline: {e}")
        traceback.print_exc()
        return False

def check_environment():
    """Check environment and dependencies."""
    print("=== DEBUG: Environment Check ===")
    
    try:
        # Check key imports
        import numpy
        print(f"✅ numpy: {numpy.__version__}")
        
        import faiss
        print(f"✅ faiss: {faiss.__version__}")
        
        import pypdf
        print(f"✅ pypdf: {pypdf.__version__}")
        
        from src.settings import settings
        print(f"✅ Settings loaded")
        print(f"  - Chunk size: {settings.chunk_size}")
        print(f"  - Overlap: {settings.chunk_overlap}")
        print(f"  - Top K: {settings.top_k}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in environment: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 DocuChat Processing Debug Tool")
    print("=" * 50)
    
    # Run all debug tests
    tests = [
        ("Environment Check", check_environment),
        ("PDF Processing", debug_pdf_processing),
        ("Embeddings", debug_embeddings),
        ("Vector Store", debug_vector_store),
        ("Full Pipeline", debug_full_pipeline)
    ]
    
    passed = 0
    for name, test_func in tests:
        print(f"\n🧪 Running {name}...")
        try:
            if test_func():
                print(f"✅ {name} passed")
                passed += 1
            else:
                print(f"❌ {name} failed")
        except Exception as e:
            print(f"💥 {name} crashed: {e}")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed < len(tests):
        print("\n🚨 Issues found! Check the errors above.")
        print("💡 The processing might be hanging due to:")
        print("   - Large embedding operations without progress feedback")
        print("   - Blocking I/O operations")
        print("   - Memory issues with large documents")
        print("   - Network timeouts (if using real OpenAI API)")
    else:
        print("\n🎉 All tests passed! The issue might be:")
        print("   - Streamlit's session state handling")
        print("   - UI threading issues")
        print("   - Browser-specific problems")
