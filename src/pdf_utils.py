"""PDF processing utilities for text extraction and chunking."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

try:
    from pypdf import PdfReader
except ImportError:
    # Fallback for older versions
    from PyPDF2 import PdfReader

from .types import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file using pypdf.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a single string
        
    Raises:
        PDFProcessingError: If PDF cannot be read or processed
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFProcessingError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise PDFProcessingError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            
            if len(reader.pages) == 0:
                raise PDFProcessingError("PDF has no pages")
            
            text_parts = []
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        # Add page marker for potential future use
                        text_parts.append(f"[PAGE {page_num}]\n{page_text}")
                    logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            if not text_parts:
                raise PDFProcessingError("No text could be extracted from PDF")
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"Successfully extracted {len(full_text)} characters from {len(reader.pages)} pages")
            
            return full_text
            
    except PDFProcessingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing PDF {pdf_path}: {e}")
        raise PDFProcessingError(f"Failed to process PDF: {e}")


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page markers for chunking (keep the text, remove the markers)
    text = re.sub(r'\[PAGE \d+\]\s*', '', text)
    
    # Fix common PDF extraction issues
    text = text.replace('\x00', '')  # Remove null bytes
    text = text.replace('\ufeff', '')  # Remove BOM
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive newlines but preserve paragraph breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()


def chunk_text(
    text: str, 
    chunk_size: int = 1200, 
    overlap: int = 200,
    min_chunk_size: int = 100
) -> List[DocumentChunk]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk to be included
        
    Returns:
        List of DocumentChunk objects
        
    Raises:
        ValueError: If parameters are invalid
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    if min_chunk_size <= 0:
        raise ValueError("min_chunk_size must be positive")
    
    text = clean_text(text)
    if not text:
        logger.warning("No text to chunk")
        return []
    
    chunks = []
    start = 0
    chunk_id = 0
    
    logger.info(f"Chunking text of {len(text)} characters with chunk_size={chunk_size}, overlap={overlap}")
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If this is not the last chunk, try to find a good break point
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            break_window = min(100, chunk_size // 4)  # Look within 100 chars or 25% of chunk_size
            search_start = max(end - break_window, start + min_chunk_size)
            search_end = min(end + break_window, len(text))
            
            # Find the best break point (sentence ending)
            best_break = end
            for i in range(search_end - 1, search_start - 1, -1):
                if text[i] in '.!?':
                    # Check if it's not an abbreviation
                    if i < len(text) - 1 and text[i + 1].isspace():
                        best_break = i + 1
                        break
            
            end = best_break
        
        # Extract chunk text
        chunk_text = text[start:end].strip()
        
        # Skip chunks that are too small (unless it's the last chunk)
        if len(chunk_text) >= min_chunk_size or end >= len(text):
            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=chunk_id,
                start_char=start,
                end_char=end,
                metadata={
                    'chunk_size': len(chunk_text),
                    'original_length': len(text)
                }
            )
            chunks.append(chunk)
            chunk_id += 1
        
        # Move start position for next chunk (with overlap)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def get_document_metadata(pdf_path: str, chunks: List[DocumentChunk]) -> DocumentMetadata:
    """
    Extract metadata about the processed document.
    
    Args:
        pdf_path: Path to the original PDF file
        chunks: List of processed chunks
        
    Returns:
        DocumentMetadata object
    """
    pdf_path = Path(pdf_path)
    
    # Get file size
    file_size = pdf_path.stat().st_size if pdf_path.exists() else 0
    
    # Get number of pages
    num_pages = None
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            num_pages = len(reader.pages)
    except Exception as e:
        logger.warning(f"Could not get page count: {e}")
    
    # Calculate statistics
    total_characters = sum(len(chunk.text) for chunk in chunks)
    avg_chunk_size = total_characters / len(chunks) if chunks else 0
    
    return DocumentMetadata(
        filename=pdf_path.name,
        file_size=file_size,
        num_pages=num_pages,
        num_chunks=len(chunks),
        ingestion_timestamp=datetime.now(),
        total_characters=total_characters,
        avg_chunk_size=avg_chunk_size
    )


def process_pdf(
    pdf_path: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> Tuple[List[DocumentChunk], DocumentMetadata]:
    """
    Complete PDF processing pipeline: extract text, chunk, and get metadata.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        Tuple of (chunks, metadata)
        
    Raises:
        PDFProcessingError: If processing fails
    """
    try:
        logger.info(f"Starting PDF processing pipeline for: {pdf_path}")
        
        # Extract text
        text = extract_text(pdf_path)
        
        # Create chunks
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        
        # Get metadata
        metadata = get_document_metadata(pdf_path, chunks)
        
        logger.info(f"PDF processing complete: {len(chunks)} chunks, {metadata.total_characters} characters")
        
        return chunks, metadata
        
    except Exception as e:
        logger.error(f"PDF processing pipeline failed: {e}")
        raise PDFProcessingError(f"Failed to process PDF: {e}")


def validate_pdf_file(pdf_path: str) -> bool:
    """
    Validate that a PDF file can be processed.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if file can be processed, False otherwise
    """
    try:
        pdf_path = Path(pdf_path)
        
        # Check file exists and is readable
        if not pdf_path.exists():
            logger.error(f"PDF file does not exist: {pdf_path}")
            return False
        
        if not pdf_path.is_file():
            logger.error(f"Path is not a file: {pdf_path}")
            return False
        
        if not pdf_path.suffix.lower() == '.pdf':
            logger.error(f"File is not a PDF: {pdf_path}")
            return False
        
        # Try to read the PDF
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            if len(reader.pages) == 0:
                logger.error("PDF has no pages")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"PDF validation failed: {e}")
        return False