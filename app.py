import sys
import os
import re
import html
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from chatbot import LegalChatbot
from utils import load_env

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="LawGlance – AI Legal Assistant", layout="wide", page_icon="⚖️")
load_env()

# ---------------- HELPERS ----------------
def render_bot_html(text: str) -> str:
    """Render chatbot answer in a styled HTML format (gold header + bullets)."""
    if not text:
        return ""

    text = html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # remove markdown bold

    lines = [ln.strip() for ln in text.splitlines()]
    html_parts = []
    i = 0

    # --- Heading ---
    if lines and len(lines[0]) > 0:
        first = lines[0]
        if len(first) <= 100:
            html_parts.append(f"<div class='bot-heading'><strong>{first}</strong></div>")
            i = 1

    # --- Process rest ---
    in_list = False
    for ln in lines[i:]:
        if ln == "":
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        if re.match(r"^[-•]\s+", ln):
            if not in_list:
                html_parts.append("<ul class='bot-bullets'>")
                in_list = True
            item = re.sub(r"^[-•]\s+", "", ln)
            html_parts.append(f"<li>{item}</li>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<p class='bot-para'>{ln}</p>")

    if in_list:
        html_parts.append("</ul>")

    inner = "\n".join(html_parts)
    html_card = f"""
    <div class="stChatMessage bot-msg">
      <span style="font-size:1.05rem;">🤖 <strong>LawGlance:</strong></span>
      <div style="margin-top:8px;">{inner}</div>
    </div>
    """
    return html_card


def render_user_html(text: str) -> str:
    return f"""
    <div class="stChatMessage user-msg">
        <span>🧑‍⚖️ <strong>You:</strong></span>
        <div style="margin-top:6px;">{html.escape(text)}</div>
    </div>
    """

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0d1117;
    color: #f0f0f0;
}
.main {
    max-width: 850px;
    margin: auto;
    padding-top: 1.2rem;
}

/* Gold law symbol */
.title-symbol {
    text-align: center;
    font-size: 3rem;
    color: gold;
    animation: glow 2s infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 5px gold; }
    to { text-shadow: 0 0 15px gold; }
}

h1 {
    text-align: center;
    font-size: 2.2rem;
    color: #ffffff;
    margin-top: 0.5rem;
}
p.subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #ccc;
    margin-bottom: 1.5rem;
}

/* Chat bubbles */
.stChatMessage {
    border-radius: 14px;
    padding: 14px 18px;
    margin: 10px auto;
    max-width: 75%;
}
.user-msg {
    background-color: #d8fcd8;
    color: #111;
    border-left: 5px solid #27ae60;
}
.bot-msg {
    background: linear-gradient(145deg, #1b1e26, #141418);
    color: #f5f5f5;
    border-left: 5px solid #9370db;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(147, 112, 219, 0.3);
}

/* Text styles */
.bot-heading {
    font-size: 1.05rem;
    margin-bottom: 6px;
    color: #ffd86b;
}
.bot-bullets {
    margin: 8px 0 8px 20px;
}
.bot-bullets li {
    margin: 6px 0;
    line-height: 1.55;
}
.bot-para {
    margin: 8px 0;
    line-height: 1.5;
    color: #d8d8d8;
}

.footer {
    text-align: center;
    font-size: 0.8rem;
    color: #888;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- INIT CHATBOT ----------------
@st.cache_resource
def get_chatbot():
    return LegalChatbot()

chatbot = get_chatbot()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Settings")

# 💡 How to Use
st.sidebar.markdown("### 💡 How to Use")
st.sidebar.info("""
1️⃣ Type your legal question or select a topic.  
2️⃣ The AI searches verified Indian laws and acts.  
3️⃣ Answers appear neatly in bullet points.  
💬 Try asking — *“How to file an FIR?”* or *“What is defamation under IPC?”*
""")

# 🧠 Active Backend
st.sidebar.markdown("### 🧠 Active Backend")
st.sidebar.write(f"Using: **{chatbot.backend.upper()}**")

# 🗑️ Clear Chat
if st.sidebar.button("🗑️ Clear Conversation"):
    st.session_state.history = []
    chatbot.chat_history = []
    st.rerun()

# 📚 Topics
st.sidebar.markdown("### 📚 Browse Legal Topics")
topics = {
    "Fundamental Rights": "What are Fundamental Rights?",
    "Criminal Law": "What is Section 302 IPC?",
    "Cyber Law": "How to register a cybercrime complaint?",
    "Women’s Rights": "What are women’s rights under Indian law?",
    "Consumer Protection": "What are consumer rights in India?"
}
topic_choice = st.sidebar.selectbox("Select a topic", ["Select"] + list(topics.keys()))
user_input = topics[topic_choice] if topic_choice != "Select" else None

# ---------------- HEADER ----------------
st.markdown("<div class='title-symbol'>⚖️</div>", unsafe_allow_html=True)
st.markdown("<h1>LawGlance – AI Legal Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask any legal question below or choose a quick example 👇</p>", unsafe_allow_html=True)

# ---------------- EXAMPLES ----------------
st.markdown("### 💬 Try Asking:")
cols = st.columns(3)
examples = [
    "How to file an FIR?",
    "What are fundamental rights?",
    "What is bail?",
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

chat_input = st.chat_input("Ask your legal question here...")
if st.session_state.clicked_example:
    user_input = st.session_state.clicked_example
    st.session_state.clicked_example = None
elif not user_input:
    user_input = chat_input

# ---------------- CHAT ----------------
if "history" not in st.session_state:
    st.session_state.history = []

if user_input:
    with st.spinner("💭 Thinking..."):
        answer = chatbot.ask(user_input, top_k=5, show_sources=False, min_score=0.25)
        st.session_state.history.append(("user", user_input))
        st.session_state.history.append(("bot", answer))

# ---------------- DISPLAY CHAT ----------------
for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(render_user_html(msg), unsafe_allow_html=True)
    else:
        st.markdown(render_bot_html(msg), unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
⚠️ This AI provides general legal information for awareness only.<br>
It does not replace professional legal consultation.<br><br>
Built with ❤️ using Streamlit, FAISS, Groq/OpenAI, and Sentence Transformers.
</div>
""", unsafe_allow_html=True)
