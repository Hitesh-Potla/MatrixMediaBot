import os
import time
import requests
import uuid
import streamlit as st

# Docker exposes the API through Nginx at port 8080. Override this with
# CHAT_API_URL=http://localhost:8000/v1/chat for direct-Uvicorn development.
API_URL = os.getenv("CHAT_API_URL", "http://localhost:8080/api/chat/chat")
AUTH_START_URL = API_URL.rsplit("/", 1)[0] + "/auth/start-session"

st.set_page_config(
    page_title="Matrix Media AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern design
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
        color: #e2e8f0;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 2.5rem 1.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Message styling */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1.5rem;
    }
    
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1.5rem;
    }
    
    .message-bubble {
        max-width: 70%;
        padding: 1rem 1.25rem;
        border-radius: 1.125rem;
        line-height: 1.6;
        word-wrap: break-word;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        border-radius: 1.125rem 0.25rem 1.125rem 1.125rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    .assistant-bubble {
        background: #1e293b;
        color: #e2e8f0;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 0.25rem 1.125rem 1.125rem 1.125rem;
    }
    
    .message-bubble p {
        margin: 0;
    }
    
    .message-bubble strong {
        color: inherit;
    }
    
    /* Loading state */
    .loading-dots {
        display: inline-flex;
        gap: 0.25rem;
    }
    
    .loading-dot {
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        background: #60a5fa;
        animation: pulse 1.5s infinite;
    }
    
    .loading-dot:nth-child(2) {
        animation-delay: 0.3s;
    }
    
    .loading-dot:nth-child(3) {
        animation-delay: 0.6s;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.1); }
    }
    
    /* Response time badge */
    .response-badge {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.75rem;
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: rgba(148, 163, 184, 0.1);
        border-radius: 0.5rem;
    }
    
    /* Input area */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, #0f172a 100%);
        border-top: 1px solid rgba(148, 163, 184, 0.1);
        padding: 1.5rem 1rem;
        z-index: 100;
    }
    
    .input-wrapper {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Sidebar */
    .sidebar {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .sidebar .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .sidebar .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        transform: translateY(-2px);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(148, 163, 184, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(96, 165, 250, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(96, 165, 250, 0.5);
    }
    
    /* Error styling */
    .stAlert {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        color: #fca5a5 !important;
    }
    
    /* Chat message container padding adjustment */
    .stChatMessage {
        padding: 0 !important;
    }
    
    .stChatMessage > div {
        padding: 0 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(148, 163, 184, 0.05) !important;
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 0.5rem !important;
    }
    
    /* Bottom padding for chat */
    .chat-bottom-padding {
        height: 200px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🤖 Matrix Media AI Assistant</div>
    <div class="header-subtitle">Ask anything about our services, solutions, careers, and company information</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.conversation_id = ""
    st.session_state.authenticated = False
    st.session_state.contact = ""
    st.session_state.name = ""

if not st.session_state.authenticated:
    st.markdown("""
    <div class="chat-container">
        <div class="assistant-message">
            <div class="message-bubble assistant-bubble">
                Please verify your contact details to start chatting.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("auth_form"):
            name = st.text_input("Name")
            contact = st.text_input("Email or Phone Number")
            submit = st.form_submit_button("Start Session", use_container_width=True)
            
            if submit:
                if not name or not contact:
                    st.error("Please enter both Name and Contact details.")
                else:
                    with st.spinner("Starting session..."):
                        try:
                            resp = requests.post(AUTH_START_URL, json={"name": name, "contact": contact}, timeout=10)
                            if resp.status_code == 200:
                                data = resp.json()
                                st.session_state.authenticated = True
                                st.session_state.contact = contact
                                st.session_state.name = name
                                st.session_state.conversation_id = data.get("conversation_id", "")
                                st.rerun()
                            else:
                                st.error(f"Error: {resp.text}")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")

else:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <div class="message-bubble user-bubble">
                        {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-bubble assistant-bubble">
                        {msg["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Add spacing for fixed input
    st.markdown('<div class="chat-bottom-padding"></div>', unsafe_allow_html=True)

    # Chat input and processing
    if prompt := st.chat_input("Type your question here...", key="chat_input"):
        
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        st.markdown(f"""
        <div class="user-message">
            <div class="message-bubble user-bubble">
                {prompt}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create placeholder for assistant response
        response_placeholder = st.empty()
        badge_placeholder = st.empty()
        
        with response_placeholder.container():
            st.markdown("""
            <div class="assistant-message">
                <div class="message-bubble assistant-bubble">
                    <div class="loading-dots">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                    <span style="margin-left: 0.5rem; color: #94a3b8;">Searching knowledge base...</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        start = time.time()
        
        try:
            response = requests.post(
                API_URL,
                json={"message": prompt, "conversation_id": st.session_state.conversation_id},
                headers={
                    "ngrok-skip-browser-warning": "true",
                    "User-Agent": "PostmanRuntime/7.32.3"
                },
                timeout=30
            )
            
            if response.status_code == 401:
                st.session_state.authenticated = False
                st.session_state.messages = []
                st.session_state.conversation_id = ""
                st.rerun()
                st.stop()
                
            elapsed = time.time() - start
            data = response.json()
            answer = data.get("answer", "Unable to retrieve answer")
            sources = data.get("sources", [])
            
            # Update response
            with response_placeholder.container():
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-bubble assistant-bubble">
                        {answer}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show response time badge
            with badge_placeholder.container():
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="response-badge">⏱️ Response time: {elapsed:.2f}s</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
            
            # Rerun to update UI
            st.rerun()
            
        except requests.exceptions.Timeout:
            with response_placeholder.container():
                st.markdown("""
                <div class="assistant-message">
                    <div class="message-bubble assistant-bubble">
                        <strong>⚠️ Request Timeout</strong><br>
                        The server took too long to respond. Please try again.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        except requests.exceptions.ConnectionError:
            with response_placeholder.container():
                st.markdown("""
                <div class="assistant-message">
                    <div class="message-bubble assistant-bubble">
                        <strong>❌ Connection Error</strong><br>
                        Unable to connect to the chatbot service. Please check if the server is running.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            with response_placeholder.container():
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="message-bubble assistant-bubble">
                        <strong>❌ Error</strong><br>
                        {str(e)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        st.session_state.conversation_id = ""
        st.session_state.authenticated = False
        st.session_state.contact = ""
        st.session_state.name = ""
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### About
    This AI assistant helps you find information about Matrix Media's services, solutions, and company information from our knowledge base.
    
    **Features:**
    - 🔍 Smart search across documents
    - 📄 Source attribution
    - ⏱️ Quick response times
    """)