# PRISM – Advanced UI Version (Premium Dark Theme)

import numpy as np
import json
import os
from urllib import request
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="PRISM Digital Twin", layout="wide")

# Custom CSS for premium dark UI
st.markdown("""
<style>
body {
    background-color: #0d1117;
}

.stApp {
    background: linear-gradient(180deg, #0d1117, #111827);
    color: white;
}

h1 {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #00f5ff, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stTextInput>div>div>input {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
    padding: 12px;
}

.stButton>button {
    background: linear-gradient(90deg, #7c3aed, #00f5ff);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
}

.chat-box {
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
}

.user-msg {
    background: #2563eb;
    text-align: right;
}

.bot-msg {
    background: #1f2937;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Config
# -----------------------------
MEMORY_FILE = "memory_store.json"
EMBED_URL = "http://localhost:11434/api/embeddings"
LLM_URL = "http://localhost:11434/api/generate"

# -----------------------------
# Embedding
# -----------------------------
def get_embedding(text):
    payload = json.dumps({
        "model": "nomic-embed-text",
        "prompt": text
    }).encode("utf-8")

    req = request.Request(EMBED_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["embedding"]
    except:
        return np.random.rand(768).tolist()

# -----------------------------
# Memory
# -----------------------------
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        texts = ["I love robotics", "I build electronics"]
        embeddings = [get_embedding(t) for t in texts]
        save_memory(texts, embeddings)
        return texts, embeddings

    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
        return data["texts"], data["embeddings"]


def save_memory(texts, embeddings):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"texts": texts, "embeddings": embeddings}, f)

# -----------------------------
# Retrieval
# -----------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def retrieve(query, texts, embeddings, k=2):
    q_vec = np.array(get_embedding(query), dtype=np.float32)
    embeddings = np.array(embeddings, dtype=np.float32)
    scores = [cosine_similarity(q_vec, emb) for emb in embeddings]
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [texts[i] for i in top_k]

# -----------------------------
# LLM
# -----------------------------
def ask_llm(prompt):
    payload = json.dumps({
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = request.Request(LLM_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))["response"]
    except:
        return "Fallback response"

# -----------------------------
# UI Layout
# -----------------------------
st.markdown("<h1>PRISM Digital Twin</h1>", unsafe_allow_html=True)

texts, embeddings = load_memory()

# Session state
if "chat" not in st.session_state:
    st.session_state.chat = []

col1, col2 = st.columns([2, 1])

# -----------------------------
# Chat Section
# -----------------------------
with col1:
    user_input = st.text_input("Ask your AI Twin...")

    if user_input:
        memories = retrieve(user_input, texts, embeddings)
        context = "\n".join(memories)

        prompt = f"""
Memories:
{context}

Question:
{user_input}
"""

        response = ask_llm(prompt)
        st.session_state.chat.append((user_input, response))

    for q, a in st.session_state.chat[::-1]:
        st.markdown(f'<div class="chat-box user-msg">{q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-box bot-msg">{a}</div>', unsafe_allow_html=True)

# -----------------------------
# Sidebar (Features)
# -----------------------------
with col2:
    st.subheader("⚡ Quick Actions")

    new_memory = st.text_input("Add Memory")
    if st.button("Save Memory") and new_memory:
        texts.append(new_memory)
        embeddings.append(get_embedding(new_memory))
        save_memory(texts, embeddings)
        st.success("Saved!")

    if st.button("Clear Chat"):
        st.session_state.chat = []

    st.markdown("---")
    st.caption("PRISM v1.0 • Local AI Twin")
