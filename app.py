"""
Blu-Rag Chatbot - Roleplay AI with Optimized Short-Term Memory

Main Streamlit application providing the user interface for
creating characters and chatting with AI roleplay partners.

Features:
- Multi-chat support
- Custom character creation
- Multiple LLM providers
- Intelligent memory management
- Responsive dark-themed UI with proper markdown formatting

Author: Blu-Rag
Project: Blu-Rag Chatbot
Version: 1.1.0
"""

import streamlit as st
from langchain_core.messages import SystemMessage
from character import Character, character_prompt
from chatbot import (
    init_llm,
    init_summary_model,
    should_summarize,
    summarize_memory,
    add_user_message,
    add_ai_message,
    get_model_info,
    get_supported_providers,
    create_system_prompt,
)

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Blu-Rag Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Blu-Rag/blu-rag-chatbot",
        "Report a Bug": "https://github.com/Blu-Rag/blu-rag-chatbot/issues",
        "About": """
        # Blu-Rag Chatbot 🤖
        
        Roleplay AI with Optimized Short-Term Memory
        
        Created by **Blu-Rag**
        
        [GitHub Repository](https://github.com/Blu-Rag/blu-rag-chatbot)
        """,
    },
)

# =============================================================================
# Custom CSS Styling
# =============================================================================

st.markdown(
    """
<style>
    /* Main App Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
        padding: 18px 25px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #2d4a6f;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        font-size: 26px;
        font-weight: 700;
        color: #00d4ff;
        margin: 0;
        letter-spacing: 0.5px;
    }
    
    .header-subtitle {
        font-size: 13px;
        color: #8b9bb4;
        margin: 6px 0 0 0;
        font-weight: 400;
    }
    
    /* Online Indicator */
    .online-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 12px;
        color: #00ff88;
        background-color: rgba(0, 255, 136, 0.1);
        padding: 6px 12px;
        border-radius: 20px;
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    
    .online-dot {
        width: 8px;
        height: 8px;
        background-color: #00ff88;
        border-radius: 50%;
        animation: pulse 2s infinite;
        box-shadow: 0 0 8px rgba(0, 255, 136, 0.6);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.2); }
    }
    
    /* Chat Message Styling - Using st.chat_message */
    .stChatMessage {
        background-color: transparent;
        padding: 8px 0;
    }
    
    /* User message avatar and content */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, rgba(30, 58, 95, 0.4) 0%, rgba(21, 42, 69, 0.4) 100%);
        border-radius: 12px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border: 1px solid #2d4a6f;
    }
    
    /* AI message avatar and content */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: linear-gradient(135deg, rgba(45, 45, 68, 0.4) 0%, rgba(31, 31, 46, 0.4) 100%);
        border-radius: 12px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border: 1px solid #3d3d5c;
    }
    
    /* Chat message content - ensure proper formatting */
    .stChatMessageContent {
        color: #e0e0e0;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Markdown elements inside chat */
    .stChatMessageContent p {
        margin-bottom: 0.8em;
    }
    
    .stChatMessageContent p:last-child {
        margin-bottom: 0;
    }
    
    .stChatMessageContent strong {
        color: #00d4ff;
        font-weight: 600;
    }
    
    .stChatMessageContent em {
        color: #8b9bb4;
        font-style: italic;
    }
    
    .stChatMessageContent code {
        background-color: #1a1a2e;
        padding: 2px 6px;
        border-radius: 4px;
        color: #00ff88;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    
    .stChatMessageContent pre {
        background-color: #1a1a2e;
        padding: 12px;
        border-radius: 8px;
        overflow-x: auto;
        border: 1px solid #3d3d5c;
        margin: 8px 0;
    }
    
    .stChatMessageContent pre code {
        background: none;
        padding: 0;
        color: #e0e0e0;
    }
    
    /* Avatar styling */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #1e3a5f;
        border: 2px solid #00d4ff;
    }
    
    /* Character Card */
    .character-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #121220 100%);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #3d3d5c;
        font-size: 13px;
        line-height: 1.6;
    }
    
    .character-card strong {
        color: #00d4ff;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #2d2d44;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #00d4ff;
        font-size: 16px;
        margin-bottom: 12px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a5f 0%, #152a45 100%);
        color: #ffffff;
        border: 1px solid #2d4a6f;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2d4a6f 0%, #1e3a5f 100%);
        border-color: #00d4ff;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stNumberInput > div > div > input {
        background-color: #1a1a2e;
        color: #ffffff;
        border: 1px solid #3d3d5c;
        border-radius: 8px;
        padding: 10px 12px;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1);
    }
    
    /* Empty Chat State */
    .empty-chat {
        text-align: center;
        color: #8b9bb4;
        padding: 60px 20px;
        background-color: #161b22;
        border-radius: 12px;
        border: 1px solid #2d2d44;
    }
    
    .empty-chat-icon {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.7;
    }
    
    /* Memory Indicator */
    .memory-badge {
        display: inline-block;
        background-color: rgba(0, 212, 255, 0.1);
        color: #00d4ff;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #161b22;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2d4a6f;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #3d5a7f;
    }
    
    /* Chat input container */
    .stChatInputContainer {
        background-color: #161b22;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #2d2d44;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Session State Initialization
# =============================================================================


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "chats": {"Chat #1": []},
        "current_chat": "Chat #1",
        "character": Character(),
        "llm": None,
        "summary_model": None,
        "memory": {},
        "summary_index": {},
        "provider_configured": False,
        "character_applied": False,
        "current_provider": "Groq",
        "summary_key": "",
        "api_key": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# =============================================================================
# Header Component
# =============================================================================


def render_header():
    """Render the application header."""
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(
            """
        <div class="header-container">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="header-title">🤖 Blu-Rag Chatbot</h1>
                    <p class="header-subtitle">Roleplay AI with Optimized Short-Term Memory</p>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-top: 15px;">
            <div class="online-indicator">
                <span class="online-dot"></span>
                <span>Online</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# =============================================================================
# Sidebar Component
# =============================================================================


def render_sidebar():
    """Render the sidebar with chats and settings."""
    with st.sidebar:
        # Chats Section
        st.markdown("### 💬 Chats")

        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            new_chat_num = len(st.session_state.chats) + 1
            new_chat_name = f"Chat #{new_chat_num}"
            st.session_state.chats[new_chat_name] = []
            st.session_state.current_chat = new_chat_name
            st.session_state.memory[new_chat_name] = []
            st.session_state.summary_index[new_chat_name] = 1
            st.rerun()

        st.markdown("---")

        # Chat List
        for chat_name in st.session_state.chats:
            is_active = chat_name == st.session_state.current_chat
            if st.button(
                f"{'💬' if is_active else '📝'} {chat_name}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                key=f"chat_{chat_name}",
            ):
                st.session_state.current_chat = chat_name
                st.rerun()

        st.markdown("---")

        # Settings Section
        st.markdown("### ⚙️ Settings")

        # Provider Selection
        providers = get_supported_providers()
        provider = st.selectbox(
            "Provider",
            providers,
            index=providers.index(st.session_state.current_provider),
            key="provider_select",
        )
        st.session_state.current_provider = provider

        # Model Info
        model_info = get_model_info(provider)
        st.info(f"📦 Model: {model_info['model']}")
        st.caption(model_info["description"])

        # API Keys
        st.markdown("#### 🔑 API Keys")

        summary_key = st.text_input(
            "Memory Summarizer (Groq)",
            type="password",
            value=st.session_state.summary_key,
            key="summary_key_input",
            help="Required for memory summarization. Get key from console.groq.com",
        )
        st.session_state.summary_key = summary_key

        api_key = st.text_input(
            f"Main API Key ({provider})",
            type="password",
            value=st.session_state.api_key,
            key="api_key_input",
            help=f"API key for {provider}. Visit {model_info['website']}",
        )
        st.session_state.api_key = api_key

        # Configure Button
        if st.button(
            "🔧 Configure Provider", use_container_width=True, key="configure_btn"
        ):
            if not summary_key.strip():
                st.error("Please enter the Memory Summarizer API key!")
            elif not api_key.strip():
                st.error(f"Please enter the {provider} API key!")
            else:
                try:
                    with st.spinner("Configuring provider..."):
                        st.session_state.summary_model = init_summary_model(summary_key)
                        st.session_state.llm = init_llm(provider, api_key)
                        st.session_state.provider_configured = True
                    st.success("✅ Provider configured successfully!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Memory Stats
        current_chat = st.session_state.current_chat
        if current_chat in st.session_state.memory:
            mem_len = len(st.session_state.memory[current_chat])
            st.markdown("---")
            st.markdown(f"#### 🧠 Memory Status")
            st.markdown(
                f"<span class='memory-badge'>{mem_len} messages</span>",
                unsafe_allow_html=True,
            )

            if mem_len > 30:
                st.warning(f"⚠️ Memory will summarize at {40} messages")


# =============================================================================
# Chat Component
# =============================================================================


def render_chat():
    """Render the main chat interface using st.chat_message."""
    current_chat = st.session_state.current_chat

    st.markdown(f"#### 💬 {current_chat}")

    if st.session_state.provider_configured:
        model_info = get_model_info(st.session_state.current_provider)
        st.caption(
            f"Provider: {st.session_state.current_provider} | Model: {model_info['model']}"
        )

    # Chat Container
    chat_container = st.container(height=500)

    with chat_container:
        if (
            current_chat in st.session_state.chats
            and st.session_state.chats[current_chat]
        ):
            # Display messages using st.chat_message for proper formatting
            for msg in st.session_state.chats[current_chat]:
                if msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(msg["content"])
                else:
                    char_name = st.session_state.character.get_display_name()
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])
        else:
            st.markdown(
                """
            <div class="empty-chat">
                <div class="empty-chat-icon">👋</div>
                <p><strong>Welcome to Blu-Rag Chatbot!</strong></p>
                <p style="font-size: 13px; margin-top: 8px;">
                    Configure a provider, create your character, and start roleplaying!
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Chat Input using st.chat_input
    if prompt := st.chat_input("Type your message..."):
        if not st.session_state.provider_configured:
            st.error("⚠️ Please configure the provider first in the sidebar!")
        elif not st.session_state.character_applied:
            st.error("⚠️ Please apply a character first!")
        else:
            handle_message(prompt)

    # Reset Button
    if st.button("🚪 Reset Chat", use_container_width=True, key="exit_btn"):
        st.session_state.chats = {"Chat #1": []}
        st.session_state.current_chat = "Chat #1"
        st.session_state.memory = {}
        st.session_state.summary_index = {}
        st.session_state.character_applied = False
        st.rerun()


def handle_message(user_input: str):
    """Handle user message and get AI response."""
    current_chat = st.session_state.current_chat

    # Add to chat history
    st.session_state.chats[current_chat].append({"role": "user", "content": user_input})

    # Initialize memory if needed
    if (
        current_chat not in st.session_state.memory
        or not st.session_state.memory[current_chat]
    ):
        system_msg = create_system_prompt(st.session_state.character.to_dict())
        st.session_state.memory[current_chat] = [system_msg]
        st.session_state.summary_index[current_chat] = 1

    memory = st.session_state.memory[current_chat]
    summary_idx = st.session_state.summary_index[current_chat]

    # Summarize if needed
    if should_summarize(memory, summary_idx):
        with st.spinner("🧠 Summarizing memory..."):
            memory, summary_idx, _ = summarize_memory(
                memory, summary_idx, st.session_state.summary_model
            )
            st.session_state.memory[current_chat] = memory
            st.session_state.summary_index[current_chat] = summary_idx

    # Add user message
    memory = add_user_message(memory, user_input)
    st.session_state.memory[current_chat] = memory

    # Get response
    char_name = st.session_state.character.get_display_name()
    with st.spinner(f"🤖 {char_name} is typing..."):
        try:
            response = st.session_state.llm.invoke(memory)
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
            return

    # Add AI message
    memory = add_ai_message(memory, response.content)
    st.session_state.memory[current_chat] = memory
    st.session_state.chats[current_chat].append({
        "role": "ai",
        "content": response.content,
    })

    st.rerun()


# =============================================================================
# Character Component
# =============================================================================


def render_character():
    """Render the character creation panel."""
    st.markdown("### 👤 Character")

    with st.form("character_form", clear_on_submit=False):
        char_name = st.text_input(
            "Name",
            value=st.session_state.character.name,
            placeholder="Enter character name",
        )

        col_age, col_gender = st.columns(2)
        with col_age:
            char_age = st.number_input(
                "Age",
                min_value=1,
                max_value=150,
                value=st.session_state.character.age,
                help="Character's age",
            )
        with col_gender:
            char_gender = st.text_input(
                "Gender",
                value=st.session_state.character.gender,
                placeholder="e.g., Male, Female, Non-binary",
            )

        char_personality = st.text_area(
            "Personality",
            value=st.session_state.character.personality,
            height=80,
            placeholder="Describe personality traits...",
        )

        char_traits = st.text_input(
            "Traits",
            value=st.session_state.character.traits,
            placeholder="e.g., brave, witty, mysterious",
        )

        char_likes = st.text_input(
            "Likes",
            value=st.session_state.character.likes,
            placeholder="Things the character enjoys",
        )

        char_dislikes = st.text_input(
            "Dislikes",
            value=st.session_state.character.dislikes,
            placeholder="Things the character dislikes",
        )

        char_background = st.text_area(
            "Background",
            value=st.session_state.character.background,
            height=80,
            placeholder="Character's backstory...",
        )

        char_speaking_style = st.text_area(
            "Speaking Style",
            value=st.session_state.character.speaking_style,
            height=60,
            placeholder="How the character speaks...",
        )

        if st.form_submit_button("✅ Apply Character", use_container_width=True):
            if not char_name.strip():
                st.error("Character name is required!")
            elif not char_personality.strip():
                st.error("Personality is required!")
            else:
                st.session_state.character = Character(
                    name=char_name.strip(),
                    age=char_age,
                    gender=char_gender.strip(),
                    personality=char_personality.strip(),
                    traits=char_traits.strip(),
                    likes=char_likes.strip(),
                    dislikes=char_dislikes.strip(),
                    background=char_background.strip(),
                    speaking_style=char_speaking_style.strip(),
                )
                st.session_state.character_applied = True

                # Reset memory with new character
                system_msg = create_system_prompt(st.session_state.character.to_dict())
                for chat_name in st.session_state.chats:
                    st.session_state.memory[chat_name] = [system_msg]
                    st.session_state.summary_index[chat_name] = 1

                st.success(f"✅ Character '{char_name}' applied!")
                st.rerun()

    if st.button("🔄 Reset Character", use_container_width=True, key="reset_char_btn"):
        st.session_state.character.reset()
        st.session_state.character_applied = False
        st.success("Character reset!")
        st.rerun()

    # Character Preview
    if st.session_state.character_applied:
        st.markdown("---")
        st.markdown("#### 📋 Character Preview")
        char = st.session_state.character
        st.markdown(
            f"""
        <div class="character-card">
            <strong>Name:</strong> {char.name}<br>
            <strong>Age:</strong> {char.age}<br>
            <strong>Gender:</strong> {char.gender}<br>
            <strong>Personality:</strong> {char.personality}<br>
            <strong>Traits:</strong> {char.traits}<br>
            <strong>Likes:</strong> {char.likes}<br>
            <strong>Dislikes:</strong> {char.dislikes}<br>
            <strong>Background:</strong> {char.background}<br>
            <strong>Speaking Style:</strong> {char.speaking_style}
        </div>
        """,
            unsafe_allow_html=True,
        )


# =============================================================================
# Main Application
# =============================================================================


def main():
    """Main application entry point."""
    render_header()
    render_sidebar()

    col_chat, col_char = st.columns([2, 1])

    with col_chat:
        render_chat()

    with col_char:
        render_character()


if __name__ == "__main__":
    main()
