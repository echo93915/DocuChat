"""Google Gemini API integration for DocuChat."""

import logging
from typing import List
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .settings import settings

logger = logging.getLogger(__name__)

# Gemini API configuration will be loaded from settings


class GeminiError(Exception):
    """Custom exception for Gemini API errors."""
    pass


def initialize_gemini():
    """Initialize Gemini with API key."""
    if not GEMINI_AVAILABLE:
        raise GeminiError("Google Generative AI package not available")
    
    genai.configure(api_key=settings.gemini_api_key)
    logger.info("Gemini API initialized")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: logger.warning(
        f"Gemini API call failed, retrying in {retry_state.next_action.sleep} seconds... "
        f"(attempt {retry_state.attempt_number})"
    )
)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Google Gemini.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    if not GEMINI_AVAILABLE:
        raise GeminiError("Gemini not available")
    
    try:
        initialize_gemini()
        
        # Remove empty texts and track original indices
        non_empty_texts = []
        original_indices = []
        for i, text in enumerate(texts):
            if text.strip():
                non_empty_texts.append(text.strip())
                original_indices.append(i)
        
        if not non_empty_texts:
            logger.warning("All input texts are empty, returning empty embeddings")
            return [[] for _ in texts]
        
        logger.info(f"Generating embeddings for {len(non_empty_texts)} texts using Gemini")
        
        embeddings = [[] for _ in texts]
        
        # Generate embeddings one by one (Gemini doesn't support batch)
        for i, text in enumerate(non_empty_texts):
            try:
                result = genai.embed_content(
                    model=settings.gemini_embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                
                if result and 'embedding' in result:
                    original_idx = original_indices[i]
                    embeddings[original_idx] = result['embedding']
                else:
                    logger.warning(f"No embedding returned for text {i}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Failed to embed text {i}: {e}")
                continue
        
        logger.info(f"Successfully generated embeddings for {len([e for e in embeddings if e])} texts")
        return embeddings
        
    except Exception as e:
        logger.error(f"Gemini embedding error: {e}")
        raise GeminiError(f"Failed to generate embeddings: {e}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: logger.warning(
        f"Gemini API call failed, retrying in {retry_state.next_action.sleep} seconds... "
        f"(attempt {retry_state.attempt_number})"
    )
)
def chat_complete(
    system: str,
    user: str,
    *,
    max_tokens: int = None,
    temperature: float = None
) -> str:
    """
    Generate chat completion using Google Gemini.
    
    Args:
        system: System message
        user: User message
        max_tokens: Max tokens (optional)
        temperature: Temperature (optional)
        
    Returns:
        Generated response text
    """
    if not user.strip():
        raise ValueError("User message cannot be empty")
    
    if not GEMINI_AVAILABLE:
        raise GeminiError("Gemini not available")
    
    try:
        initialize_gemini()
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens or settings.max_tokens_answer,
            temperature=temperature if temperature is not None else settings.temperature,
        )
        
        model = genai.GenerativeModel(
            model_name=settings.gemini_chat_model,
            generation_config=generation_config
        )
        
        # Combine system and user messages
        if system.strip():
            prompt = f"System: {system.strip()}\n\nUser: {user.strip()}"
        else:
            prompt = user.strip()
        
        logger.info("Generating chat completion using Gemini")
        
        response = model.generate_content(prompt)
        
        if response and response.text:
            answer = response.text.strip()
            logger.info(f"Successfully generated chat completion with Gemini")
            return answer
        else:
            raise GeminiError("Empty response from Gemini")
        
    except Exception as e:
        logger.error(f"Gemini chat completion error: {e}")
        raise GeminiError(f"Failed to generate chat completion: {e}")


def validate_api_key() -> bool:
    """
    Validate Gemini API key.
    
    Returns:
        True if API key is valid
    """
    try:
        if not GEMINI_AVAILABLE:
            return False
        
        initialize_gemini()
        
        # Try to list models to test API key
        models = genai.list_models()
        return len(list(models)) > 0
        
    except Exception as e:
        logger.error(f"Gemini API key validation failed: {e}")
        return False
