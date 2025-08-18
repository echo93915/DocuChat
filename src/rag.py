"""RAG (Retrieval-Augmented Generation) pipeline for DocuChat."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .settings import settings
from .types import DocumentChunk, QueryResponse, RetrievalResult
from .vectorstore import get_vector_store
from .llm import chat_complete  # Using real OpenAI API

logger = logging.getLogger(__name__)


class RAGError(Exception):
    """Custom exception for RAG pipeline errors."""
    pass


def retrieve(query: str, top_k: Optional[int] = None) -> List[str]:
    """
    Retrieve relevant document chunks for a query.
    
    Args:
        query: User query string
        top_k: Number of chunks to retrieve (uses settings default if None)
        
    Returns:
        List of relevant chunk texts
        
    Raises:
        RAGError: If retrieval fails
    """
    if not query.strip():
        raise RAGError("Query cannot be empty")
    
    k = top_k if top_k is not None else settings.top_k
    
    try:
        logger.info(f"Retrieving top {k} chunks for query: {query[:100]}...")
        
        # Get vector store and search
        store = get_vector_store()
        chunks = store.search(query, k=k)
        
        if not chunks:
            logger.warning("No chunks retrieved from vector store")
            return []
        
        logger.info(f"Successfully retrieved {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise RAGError(f"Failed to retrieve chunks: {e}")


def create_grounded_prompt(query: str, context_chunks: List[str]) -> tuple[str, str]:
    """
    Create a grounded prompt with context for the LLM.
    
    Args:
        query: User query
        context_chunks: Retrieved context chunks
        
    Returns:
        Tuple of (system_message, user_message)
    """
    system_message = """You are a precise assistant that answers questions based strictly on the provided context. 

IMPORTANT GUIDELINES:
- Answer ONLY using information from the provided context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
- Be accurate and concise
- Include specific details from the context when relevant
- Do not make assumptions or add information not present in the context
- If multiple relevant pieces of information exist, synthesize them clearly"""

    # Format context chunks
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n[Context {i}]\n{chunk.strip()}\n"
    
    user_message = f"""Question: {query}

Context:
{context_text}

Please answer the question based strictly on the provided context."""

    return system_message, user_message


def answer_query(
    query: str, 
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> Dict[str, Any]:
    """
    Complete RAG pipeline: retrieve context and generate answer.
    
    Args:
        query: User query string
        top_k: Number of chunks to retrieve
        max_tokens: Maximum tokens for answer generation
        temperature: Temperature for answer generation
        
    Returns:
        Dictionary containing answer, sources, and metadata
        
    Raises:
        RAGError: If the pipeline fails
    """
    if not query.strip():
        raise RAGError("Query cannot be empty")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting RAG pipeline for query: {query[:100]}...")
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = retrieve(query, top_k)
        
        if not retrieved_chunks:
            return {
                "answer": "I don't have any relevant information in my knowledge base to answer this question. Please make sure you have uploaded and processed a document first.",
                "sources": [],
                "query": query,
                "timestamp": start_time,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "model_used": settings.chat_model,
                "tokens_used": 0,
                "retrieval_count": 0
            }
        
        # Step 2: Create grounded prompt
        system_message, user_message = create_grounded_prompt(query, retrieved_chunks)
        
        # Step 3: Generate answer
        logger.info("Generating answer with LLM...")
        
        answer = chat_complete(
            system=system_message,
            user=user_message,
            max_tokens=max_tokens or settings.max_tokens_answer,
            temperature=temperature if temperature is not None else settings.temperature
        )
        
        if not answer:
            raise RAGError("Empty response from LLM")
        
        # Step 4: Prepare response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "answer": answer,
            "sources": retrieved_chunks,
            "query": query,
            "timestamp": start_time,
            "processing_time": processing_time,
            "model_used": settings.chat_model,
            "tokens_used": 0,  # Will be updated when real API is used
            "retrieval_count": len(retrieved_chunks)
        }
        
        logger.info(f"RAG pipeline completed in {processing_time:.2f}s")
        return response
        
    except RAGError:
        raise
    except Exception as e:
        logger.error(f"RAG pipeline failed: {e}")
        raise RAGError(f"Failed to process query: {e}")


def answer_query_with_details(
    query: str,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None
) -> QueryResponse:
    """
    Complete RAG pipeline returning detailed QueryResponse object.
    
    Args:
        query: User query string
        top_k: Number of chunks to retrieve
        max_tokens: Maximum tokens for answer generation
        temperature: Temperature for answer generation
        
    Returns:
        QueryResponse object with detailed information
        
    Raises:
        RAGError: If the pipeline fails
    """
    result = answer_query(query, top_k, max_tokens, temperature)
    
    # Convert sources to DocumentChunk objects (simplified)
    source_chunks = []
    retrieval_results = []
    
    for i, source_text in enumerate(result["sources"]):
        chunk = DocumentChunk(
            text=source_text,
            chunk_id=i,
            start_char=0,
            end_char=len(source_text),
            metadata={"retrieved_for": query}
        )
        source_chunks.append(chunk)
        
        # Create retrieval result (using mock similarity score)
        retrieval_result = RetrievalResult(
            chunk=chunk,
            similarity_score=1.0 - (i * 0.1),  # Mock decreasing similarity
            rank=i
        )
        retrieval_results.append(retrieval_result)
    
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        sources=source_chunks,
        retrieval_results=retrieval_results,
        timestamp=result["timestamp"],
        model_used=result["model_used"],
        tokens_used=result["tokens_used"],
        processing_time=result["processing_time"]
    )


def validate_query(query: str) -> bool:
    """
    Validate that a query is appropriate for processing.
    
    Args:
        query: User query to validate
        
    Returns:
        True if query is valid, False otherwise
    """
    if not query or not query.strip():
        return False
    
    # Check length constraints
    if len(query.strip()) < 3:
        logger.warning("Query too short")
        return False
    
    if len(query) > 1000:  # Reasonable upper limit
        logger.warning("Query too long")
        return False
    
    # Check for problematic content (basic)
    problematic_patterns = ['<script', '<?php', 'javascript:', 'data:']
    query_lower = query.lower()
    for pattern in problematic_patterns:
        if pattern in query_lower:
            logger.warning(f"Query contains problematic pattern: {pattern}")
            return False
    
    return True


def get_suggested_questions(context_chunks: List[str], max_suggestions: int = 5) -> List[str]:
    """
    Generate suggested questions based on the document content.
    
    Args:
        context_chunks: Sample chunks from the document
        max_suggestions: Maximum number of suggestions to generate
        
    Returns:
        List of suggested questions
    """
    if not context_chunks:
        return []
    
    # Basic question templates based on common document queries
    suggestions = []
    
    # Analyze content to generate relevant suggestions
    sample_text = " ".join(context_chunks[:3])  # Use first few chunks
    
    # Generic suggestions that work for most documents
    base_suggestions = [
        "What is this document about?",
        "What are the main topics covered?",
        "Can you summarize the key points?",
        "What are the most important details?",
        "What information is provided about [specific topic]?"
    ]
    
    # Try to make suggestions more specific based on content
    if any(word in sample_text.lower() for word in ['process', 'procedure', 'steps']):
        base_suggestions.append("What are the steps involved in this process?")
    
    if any(word in sample_text.lower() for word in ['policy', 'rule', 'regulation']):
        base_suggestions.append("What are the key policies or rules mentioned?")
    
    if any(word in sample_text.lower() for word in ['definition', 'term', 'meaning']):
        base_suggestions.append("How are key terms defined in this document?")
    
    return base_suggestions[:max_suggestions]


def explain_answer(query: str, answer: str, sources: List[str]) -> str:
    """
    Generate an explanation of how the answer was derived from sources.
    
    Args:
        query: Original query
        answer: Generated answer
        sources: Source chunks used
        
    Returns:
        Explanation text
    """
    explanation = f"This answer was generated by analyzing {len(sources)} relevant sections from the document(s). "
    
    if len(sources) == 1:
        explanation += "The answer is based on information from one specific section that directly addresses your question."
    else:
        explanation += f"The answer synthesizes information from {len(sources)} different sections to provide a comprehensive response."
    
    explanation += f"\n\nThe retrieval system found these sections most relevant to your query: '{query}'"
    
    return explanation


# Convenience function for simple usage
def ask(query: str) -> str:
    """
    Simple interface for asking questions. Returns just the answer text.
    
    Args:
        query: User question
        
    Returns:
        Answer text
    """
    try:
        if not validate_query(query):
            return "Please provide a valid question (at least 3 characters, no special formatting)."
        
        result = answer_query(query)
        return result["answer"]
        
    except Exception as e:
        logger.error(f"Simple query failed: {e}")
        return f"I'm sorry, I encountered an error while processing your question: {str(e)}"
