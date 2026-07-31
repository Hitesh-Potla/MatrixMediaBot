import os
import time
import requests
import uuid
import streamlit as st

# Docker exposes the API through Nginx at port 8080. Override this with
# CHAT_API_URL=http://localhost:8000/v1/chat for direct-Uvicorn development.
API_URL = os.getenv("CHAT_API_URL", "http://localhost:8080/api/chat/chat")
AUTH_REQUEST_URL = API_URL.rsplit("/", 1)[0] + "/auth/request-otp"
AUTH_VERIFY_URL = API_URL.rsplit("/", 1)[0] + "/auth/verify-otp"

st.set_page_config(
    page_title="Matrix Media AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with modern navbar
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
    
    /* ===== NAVBAR STYLES ===== */
    .navbar-container {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .navbar-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .navbar-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .navbar-logo:hover {
        transform: translateY(-2px);
    }
    
    .logo-icon {
        font-size: 2rem;
        background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .logo-text {
        font-size: 1.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    
    .navbar-middle {
        display: flex;
        gap: 2rem;
        flex: 1;
        justify-content: center;
    }
    
    .nav-item {
        color: #cbd5e1;
        text-decoration: none;
        font-size: 0.95rem;
        font-weight: 500;
        padding: 0.5rem 0;
        border-bottom: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-item:hover {
        color: #60a5fa;
        border-bottom-color: #60a5fa;
    }
    
    .navbar-right {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    
    .user-badge {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 1rem;
        background: rgba(148, 163, 184, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 2rem;
        font-size: 0.9rem;
        color: #cbd5e1;
        transition: all 0.3s ease;
    }
    
    .user-badge:hover {
        background: rgba(96, 165, 250, 0.15);
        border-color: rgba(96, 165, 250, 0.5);
    }
    
    .user-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: white;
        font-size: 0.9rem;
    }
    
    .navbar-btn {
        padding: 0.6rem 1.2rem;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .navbar-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.4);
    }
    
    .navbar-btn.logout {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .navbar-btn.logout:hover {
        background: rgba(239, 68, 68, 0.3);
        box-shadow: 0 8px 16px rgba(239, 68, 68, 0.2);
    }
    
    /* ===== HEADER SECTION ===== */
    .header-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.3) 100%);
        padding: 3rem 1.5rem 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
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
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, #0f172a 100%);
        border-top: 1px solid rgba(148, 163, 184, 0.1);
        padding: 1.5rem 1rem;
        z-index: 99;
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
    
    /* Responsive navbar */
    @media (max-width: 768px) {
        .navbar-middle {
            display: none;
        }
        
        .message-bubble {
            max-width: 85%;
        }
    }
</style>
""", unsafe_allow_html=True)

# Modern Navbar
def render_navbar():
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col1:
        st.markdown("""
        <a href="#" class="navbar-logo" style="text-decoration: none;">
            <div class="logo-icon">🤖</div>
            <div class="logo-text">Matrix Media</div>
        </a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="navbar-middle">
            <div class="nav-item">📚 Services</div>
            <div class="nav-item">💼 Solutions</div>
            <div class="nav-item">👥 Careers</div>
            <div class="nav-item">📞 Contact</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.session_state.authenticated:
            col_a, col_b = st.columns(2)
            with col_a:
                user_initials = "".join([word[0].upper() for word in st.session_state.name.split()[:2]])
                st.markdown(f"""
                <div class="user-badge">
                    <div class="user-avatar">{user_initials}</div>
                    <span>{st.session_state.name.split()[0]}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                if st.button("Logout", key="navbar_logout"):
                    st.session_state.authenticated = False
                    st.session_state.otp_sent = False
                    st.session_state.messages = []
                    st.session_state.conversation_id = uuid.uuid4().hex
                    st.rerun()
        else:
            st.markdown("""
            <button class="navbar-btn">Sign In</button>
            """, unsafe_allow_html=True)

st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
render_navbar()
st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.conversation_id = uuid.uuid4().hex
    st.session_state.authenticated = False
    st.session_state.otp_sent = False
    st.session_state.contact = ""
    st.session_state.name = ""

if not st.session_state.authenticated:
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Welcome to Matrix Media Assistant</div>
        <div class="header-subtitle">Intelligent answers about our services, solutions, and opportunities</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.otp_sent:
            with st.form("auth_form"):
                st.markdown("### Verify Your Identity")
                name = st.text_input("Full Name", placeholder="John Doe")
                contact = st.text_input("Email or Phone", placeholder="john@example.com")
                submit = st.form_submit_button("Get OTP", use_container_width=True)
                
                if submit:
                    if not name or not contact:
                        st.error("Please enter both Name and Contact details.")
                    else:
                        with st.spinner("Sending OTP..."):
                            try:
                                resp = requests.post(AUTH_REQUEST_URL, json={"name": name, "contact": contact}, timeout=10)
                                if resp.status_code == 200:
                                    st.session_state.otp_sent = True
                                    st.session_state.contact = contact
                                    st.session_state.name = name
                                    st.success("✅ OTP sent! Check your email or phone.")
                                    st.rerun()
                                else:
                                    st.error(f"Error: {resp.text}")
                            except Exception as e:
                                st.error(f"Connection Error: {e}")
        else:
            with st.form("verify_form"):
                st.markdown("### Verify OTP")
                st.info(f"📧 OTP sent to **{st.session_state.contact}**")
                otp = st.text_input("Enter 6-digit OTP", placeholder="000000")
                verify = st.form_submit_button("Verify OTP", use_container_width=True)
                
                if verify:
                    if not otp:
                        st.error("Please enter the OTP.")
                    else:
                        with st.spinner("Verifying..."):
                            try:
                                resp = requests.post(AUTH_VERIFY_URL, json={"contact": st.session_state.contact, "otp": otp}, timeout=10)
                                if resp.status_code == 200:
                                    st.session_state.authenticated = True
                                    st.success("✅ Verified successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Invalid or expired OTP.")
                            except Exception as e:
                                st.error(f"Connection Error: {e}")
            
            if st.button("← Back to login", use_container_width=True):
                st.session_state.otp_sent = False
                st.rerun()

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
    st.markdown("### ⚙️ Settings")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        st.session_state.conversation_id = uuid.uuid4().hex
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About This Assistant
    AI-powered knowledge assistant for Matrix Media
    
    **Key Features:**
    - 🔍 Intelligent search across documents
    - 📄 Source attribution
    - ⏱️ Quick response times
    - 🔐 Secure authentication
    
    **Need Help?**
    Contact support@matrixmedia.com
    """)