# app.py
import sys, os, re, random, time
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from chatbot import LegalChatbot
from utils import load_env

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="LawGlance – AI Legal Assistant", layout="wide", page_icon="⚖️")

load_env()

# -------------------- FORMATTER --------------------
def format_answer(text: str):
    """Clean and structure AI responses for readability."""
    if not text:
        return ""
    text = re.sub(r'\n+', '\n', text)
    text = text.replace("•", "-")
    text = re.sub(r'(?<=\.)\s+', '\n', text)
    text = re.sub(r'(\d+\))', r'\n\1', text)
    text = re.sub(r'([A-Z][a-z]+:)', r'\n**\1**', text)
    return text.strip()

# -------------------- CUSTOM STYLING --------------------
st.markdown("""
<style>
body { background-color: #0d1117; color: #f0f0f0; }

.main { max-width: 850px; margin: auto; padding-top: 1.5rem; }

.title-symbol {
    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 2.6rem;
    color: gold;
    margin-bottom: -0.4rem;
    animation: glow 2s infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 5px gold; }
    to { text-shadow: 0 0 15px gold; }
}

h1 {
    text-align: center;
    font-size: 2.3rem;
    color: #f5f5f5;
    margin-bottom: 0.2rem;
}
p.subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #ccc;
    margin-bottom: 1.5rem;
}

.stChatMessage {
    border-radius: 14px;
    padding: 14px 18px;
    margin: 10px auto;
    max-width: 75%;
    word-wrap: break-word;
}

.user-msg {
    background-color: #d8fcd8;
    color: #111;
    border-left: 5px solid #27ae60;
    box-shadow: 0px 2px 6px rgba(0,255,0,0.15);
}
.bot-msg {
    background-color: #1b1e26;
    color: #f5f5f5;
    border-left: 5px solid #9370db;
    box-shadow: 0px 2px 6px rgba(150,120,255,0.1);
}

.icon {
    font-size: 1.2rem;
    vertical-align: middle;
}
.suggestions button {
    background: linear-gradient(90deg, #3b82f6, #9333ea);
    color: white;
    border: none;
    padding: 8px 18px;
    border-radius: 10px;
    margin: 6px;
    font-size: 0.9rem;
    transition: 0.2s ease;
}
.suggestions button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #2563eb, #7e22ce);
}
.footer {
    text-align: center;
    font-size: 0.8rem;
    color: #888;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- INITIALIZE CHATBOT --------------------
@st.cache_resource
def get_chatbot():
    return LegalChatbot()

chatbot = get_chatbot()

# -------------------- SIDEBAR SETTINGS --------------------
st.sidebar.title("⚙️ Settings & Info")
mode = st.sidebar.selectbox("Response Mode", ["Normal", "Show Sources"])
top_k = st.sidebar.slider("Context Retrieval (top_k)", 3, 15, 5)
min_score = st.sidebar.slider("Similarity Threshold", 0.10, 0.90, 0.25, 0.01)

st.sidebar.markdown("### 🧠 Active Backend")
st.sidebar.write(f"Using: **{chatbot.backend.upper()}**")

st.sidebar.markdown("""
### 💡 How to Use:
- Type your legal question below  
- The AI searches your FAISS knowledge base  
- It answers using Groq/OpenAI/Flan  
""")

# -------------------- HEADER --------------------
st.markdown("<div class='title-symbol'>⚖️</div>", unsafe_allow_html=True)
st.markdown("<h1>LawGlance – AI Legal Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask any legal question below or choose a quick example 👇</p>", unsafe_allow_html=True)

# -------------------- SUGGESTED QUESTIONS --------------------
st.markdown("### 💬 Try Asking:")
cols = st.columns(3)
examples = [
    "How to file an FIR?",
    "What are fundamental rights?",
    "What is the process of getting bail?",
    "Explain Article 21 of the Indian Constitution",
    "What is defamation under IPC?",
    "How to register a cybercrime complaint?"
]

if "clicked_example" not in st.session_state:
    st.session_state.clicked_example = None

for i, example in enumerate(examples):
    with cols[i % 3]:
        if st.button(example):
            st.session_state.clicked_example = example

user_input = st.chat_input("Ask your legal question here...")
if st.session_state.clicked_example:
    user_input = st.session_state.clicked_example
    st.session_state.clicked_example = None

# -------------------- CHAT DISPLAY --------------------
if "history" not in st.session_state:
    st.session_state.history = []

if user_input:
    with st.spinner("💭 Thinking..."):
        answer = chatbot.ask(
            user_input,
            top_k=top_k,
            show_sources=(mode == "Show Sources"),
            min_score=min_score
        )
        answer = format_answer(answer)
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("bot", answer))

# Display conversation with improved formatting
for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f"<div class='stChatMessage user-msg'><span class='icon'>🧑‍⚖️</span> <b>You:</b><br>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='stChatMessage bot-msg'><span class='icon'>🤖</span> <b>LawGlance:</b><br>{msg}</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<div class='footer'>
⚠️ <i>This AI provides general legal information based on your knowledge base and public data.<br>
It does not constitute professional legal advice.</i><br><br>
Built with ❤️ using Streamlit, FAISS, Groq API, and Sentence Transformers.
</div>
""", unsafe_allow_html=True)
