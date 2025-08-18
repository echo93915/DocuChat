"""LLM interface - now redirects to unified implementation.

This module provides backward compatibility by importing from the new unified LLM module.
All functionality has been consolidated into llm_unified.py for better organization.
"""

# Import everything from the unified module for backward compatibility
from .llm_unified import (
    embed_texts,
    chat_complete,
    validate_api_key,
    validate_api_keys,
    get_token_usage,
    LLMError,
    OpenAIError,
    GeminiError
)

# Legacy compatibility
__all__ = [
    'embed_texts',
    'chat_complete', 
    'validate_api_key',
    'validate_api_keys',
    'get_token_usage',
    'LLMError',
    'OpenAIError',
    'GeminiError'
]