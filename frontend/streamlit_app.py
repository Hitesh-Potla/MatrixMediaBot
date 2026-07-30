import os
import time
import requests
import streamlit as st

# Docker exposes the API through Nginx at port 8080. Override this with
# CHAT_API_URL=http://localhost:8000/v1/chat for direct-Uvicorn development.
API_URL = os.getenv("CHAT_API_URL", "http://localhost:8000/api/chat/chat")

st.set_page_config(
    page_title="Matrix Media AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Matrix Media AI Assistant")
st.caption("Ask anything about Matrix Media's services, solutions, careers, and company information.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):

            with st.expander("📄 Sources"):

                for src in msg["sources"]:
                    st.write(
                        f"**{src['document']}**  \n"
                        f"Page: {src['page']}  \n"
                        f"Similarity: {src['score']:.3f}"
                    )

if prompt := st.chat_input("Ask a question..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        placeholder.markdown("⏳ Thinking...")

        start = time.time()

        try:

            response = requests.post(
                API_URL,
                json={"message": prompt},
                timeout=30
            )

            elapsed = time.time() - start

            data = response.json()

            answer = data["answer"]

            placeholder.markdown(answer)

            st.caption(f"Response time: {elapsed:.2f}s")

            if data["sources"]:

                with st.expander("📄 Sources"):

                    for src in data["sources"]:

                        st.write(
                            f"**{src['document']}**  \n"
                            f"Page: {src['page']}  \n"
                            f"Similarity: {src['score']:.3f}"
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": data["sources"]
                }
            )

        except Exception as e:

            placeholder.error(
                f"Unable to connect to the chatbot.\n\n{e}"
            )

if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

