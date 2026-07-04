"""
Chatbot Module for Blu-Rag Chatbot

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

MEMORY_THRESHOLD = 40  # Messages before triggering summarization
KEEP_RECENT = 20  # Recent messages to keep uncompressed
SUMMARY_MAX_TOKENS = 300  # Maximum tokens for summary output


def init_summary_model(api_key: str):

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


def should_summarize(memory: List[BaseMessage], summary_index: int) -> bool:
    return len(memory) - summary_index > MEMORY_THRESHOLD


def summarize_memory(
    memory: List[BaseMessage], summary_index: int, summary_model
) -> Tuple[List[BaseMessage], int, str]:
    start_idx = max(summary_index, 1)
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
    new_memory = memory.copy()
    new_memory.append(HumanMessage(content=user_input))
    return new_memory


def add_ai_message(memory: List[BaseMessage], ai_response: str) -> List[BaseMessage]:
    new_memory = memory.copy()
    new_memory.append(AIMessage(content=ai_response))
    return new_memory


def create_system_prompt(character_dict: dict) -> SystemMessage:
    from character import character_prompt

    prompt_text = character_prompt.format(**character_dict)
    return SystemMessage(content=prompt_text)


def get_model_info(provider: str) -> dict:
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
    return ["Groq", "Gemini", "Mistral", "OpenRouter"]
