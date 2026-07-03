"""
Chatbot Module for Blu-Rag Chatbot

This module handles LLM initialization, memory management, and
conversation summarization for the Blu-Rag Chatbot application.

Features:
- Multi-provider LLM support (Groq, Gemini, Mistral, OpenRouter)
- Intelligent memory summarization
- Conversation context management

Author: Blu-Rag
Project: Blu-Rag Chatbot
"""

from typing import Optional, Tuple, List
from langchain.chat_models import init_chat_model
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage


# =============================================================================
# Memory Configuration
# =============================================================================

MEMORY_THRESHOLD = 40  # Messages before triggering summarization
KEEP_RECENT = 20  # Recent messages to keep uncompressed
SUMMARY_MAX_TOKENS = 300  # Maximum tokens for summary output


# =============================================================================
# Model Initialization Functions
# =============================================================================


def init_summary_model(api_key: str):
    """
    Initialize the memory summarization model using Groq.

    The summarizer condenses old conversation history to maintain
    context while managing token usage efficiently.

    Args:
        api_key: Groq API key for authentication

    Returns:
        Initialized chat model configured for summarization

    Raises:
        ValueError: If API key is empty or invalid
        Exception: If model initialization fails
    """
    if not api_key or not api_key.strip():
        raise ValueError("Summary API key cannot be empty")

    return init_chat_model(
        model="llama-3.3-70b-versatile",
        model_provider="groq",
        api_key=api_key.strip(),
        temperature=0.8,
        max_tokens=SUMMARY_MAX_TOKENS,
        streaming=True,
        model_kwargs={"top_p": 0.85},
        timeout=30,
    )


def init_llm(provider: str, api_key: str):
    """
    Initialize the main LLM based on selected provider.

    Supports multiple providers with consistent configuration
    for temperature, tokens, and streaming behavior.

    Args:
        provider: Provider name ('Groq', 'Gemini', 'Mistral', 'OpenRouter')
        api_key: API key for the selected provider

    Returns:
        Initialized chat model for the selected provider

    Raises:
        ValueError: If provider is unknown or API key is invalid
        Exception: If model initialization fails
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key cannot be empty")

    api_key = api_key.strip()

    # Common parameters for all providers
    common_params = {
        "temperature": 0.8,
        "max_tokens": 300,
        "streaming": True,
        "timeout": 30,
    }

    if provider == "Groq":
        return init_chat_model(
            model="llama-3.3-70b-versatile",
            model_provider="groq",
            api_key=api_key,
            top_p=0.85,
            **common_params,
        )

    elif provider == "Gemini":
        return init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
            api_key=api_key,
            top_p=0.85,
            **common_params,
        )

    elif provider == "Mistral":
        return init_chat_model(
            model="mistral-small-latest",
            model_provider="mistralai",
            api_key=api_key,
            top_p=0.85,
            **common_params,
        )

    elif provider == "OpenRouter":
        return ChatOpenRouter(
            model="openai/gpt-3.5-turbo",
            model_kwargs={"model_provider": "openrouter"},
            api_key=api_key,
            temperature=0.8,
            max_tokens=300,
            streaming=True,
            top_p=0.85,
            timeout=30,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: Groq, Gemini, Mistral, OpenRouter"
        )


# =============================================================================
# Memory Management Functions
# =============================================================================


def should_summarize(memory: List[BaseMessage], summary_index: int) -> bool:
    """
    Determine if memory should be summarized based on threshold.

    Args:
        memory: List of conversation messages
        summary_index: Index tracking summarized portion

    Returns:
        bool: True if summarization is needed
    """
    return len(memory) - summary_index > MEMORY_THRESHOLD


def summarize_memory(
    memory: List[BaseMessage], summary_index: int, summary_model
) -> Tuple[List[BaseMessage], int, str]:
    """
    Summarize old messages in conversation memory.

    Compresses older messages while preserving recent context.
    The summary replaces multiple messages with a condensed version.

    Args:
        memory: Current conversation memory
        summary_index: Current position in memory
        summary_model: Model to use for generating summary

    Returns:
        Tuple of (updated_memory, new_summary_index, summary_content)
    """
    # Extract messages to summarize (exclude system prompt and recent messages)
    start_idx = max(summary_index, 1)  # Ensure we don't summarize system prompt
    end_idx = len(memory) - KEEP_RECENT

    if start_idx >= end_idx:
        return memory, summary_index, ""

    convo = memory[start_idx:end_idx]

    if not convo:
        return memory, summary_index, ""

    # Generate summary
    try:
        message_summary = summary_model.invoke(convo).content
    except Exception as e:
        # Fallback: return original memory if summarization fails
        return memory, summary_index, ""

    # Create updated memory with summary
    new_memory = memory.copy()
    new_memory[start_idx] = SystemMessage(
        content=f"[SUMMARY of earlier conversation]: {message_summary}"
    )
    del new_memory[start_idx + 1 : start_idx + len(convo)]

    return new_memory, start_idx + 1, message_summary


def add_user_message(memory: List[BaseMessage], user_input: str) -> List[BaseMessage]:
    """
    Add user message to memory.

    Args:
        memory: Current conversation memory
        user_input: User's message text

    Returns:
        Updated memory with new user message
    """
    new_memory = memory.copy()
    new_memory.append(HumanMessage(content=user_input))
    return new_memory


def add_ai_message(memory: List[BaseMessage], ai_response: str) -> List[BaseMessage]:
    """
    Add AI response to memory.

    Args:
        memory: Current conversation memory
        ai_response: AI's response text

    Returns:
        Updated memory with new AI message
    """
    new_memory = memory.copy()
    new_memory.append(AIMessage(content=ai_response))
    return new_memory


def create_system_prompt(character_dict: dict) -> SystemMessage:
    """
    Create system prompt from character attributes.

    Args:
        character_dict: Dictionary of character attributes

    Returns:
        SystemMessage with formatted character prompt
    """
    from character import character_prompt

    prompt_text = character_prompt.format(**character_dict)
    return SystemMessage(content=prompt_text)


# =============================================================================
# Provider Information
# =============================================================================


def get_model_info(provider: str) -> dict:
    """
    Get model information for display purposes.

    Args:
        provider: Provider name

    Returns:
        Dictionary with model name and provider website
    """
    model_map = {
        "Groq": {
            "model": "Llama-3.3-70B",
            "website": "https://console.groq.com",
            "description": "Fast inference with Llama models",
        },
        "Gemini": {
            "model": "Gemini-2.5-Flash",
            "website": "https://aistudio.google.com",
            "description": "Google's advanced multimodal AI",
        },
        "Mistral": {
            "model": "Mistral-Small-Latest",
            "website": "https://console.mistral.ai",
            "description": "Efficient and capable language models",
        },
        "OpenRouter": {
            "model": "GPT-3.5-Turbo",
            "website": "https://openrouter.ai",
            "description": "Unified API for multiple models",
        },
    }
    return model_map.get(
        provider, {"model": "Unknown", "website": "", "description": "Unknown provider"}
    )


def get_supported_providers() -> list:
    """
    Get list of supported providers.

    Returns:
        List of provider names
    """
    return ["Groq", "Gemini", "Mistral", "OpenRouter"]
