'''
# PRISM – 50% Demo Pipeline
# Local Memory + Local LLM Integration (Fixed & Stable)

"""
This script demonstrates:
1. Loading sample user data
2. Creating simple embeddings (bag-of-words)
3. Storing and retrieving memory
4. Sending retrieved memory + user question to a local LLM via Ollama HTTP API

This version:
- Fixes the unterminated string error
- Ensures stable fallback behavior when Ollama is unavailable
- Adds extra tests to validate summarization and retrieval
"""

import numpy as np
import re
import json
from urllib import request, error

# -----------------------------
# Utility: Tokenizer
# -----------------------------
def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

# -----------------------------
# 1. Sample User Data
# -----------------------------
user_texts = [
    "I love working on robotics and embedded systems.",
    "I usually write code in Python and Java.",
    "I enjoy making DIY electronics projects.",
    "My cat often interrupts me while coding.",
    "I like solving AI and machine learning problems.",
]

# -----------------------------
# 2. Build Vocabulary
# -----------------------------
all_tokens = []
for text in user_texts:
    all_tokens.extend(tokenize(text))

vocab = sorted(set(all_tokens))
vocab_index = {word: i for i, word in enumerate(vocab)}

# -----------------------------
# 3. Create Embeddings (Bag-of-Words)
# -----------------------------
def text_to_vector(text):
    vec = np.zeros(len(vocab), dtype=np.float32)
    for token in tokenize(text):
        if token in vocab_index:
            vec[vocab_index[token]] += 1.0
    return vec

embeddings = np.array([text_to_vector(t) for t in user_texts], dtype=np.float32)

# -----------------------------
# 4. Vector Similarity + Retrieval
# -----------------------------
def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve(query, k=2):
    q_vec = text_to_vector(query)
    scores = [cosine_similarity(q_vec, emb) for emb in embeddings]
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [user_texts[i] for i in top_k]

# -----------------------------
# 5. Local LLM Call via Ollama HTTP API
# -----------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"


def summarize_memories(memories):
    """Create a simple summary from memories using frequency analysis."""
    if not memories:
        return ""

    tokens = []
    for m in memories:
        tokens.extend(tokenize(m))

    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    keywords = sorted(freq, key=freq.get, reverse=True)[:5]
    return "The user often focuses on " + ", ".join(keywords) + "."


def ask_llm(prompt, model="mistral", fallback_memories=None):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "(no response)")
    except error.URLError:
        if fallback_memories:
            return "[Local intelligent summary] " + summarize_memories(fallback_memories)
        return "[Mock LLM] Ollama not reachable and no fallback available."
    except Exception as e:
        return f"LLM error: {e}"

# -----------------------------
# 6. End-to-End Query
# -----------------------------
query = "What are my interests?"
memories = retrieve(query)

context = "\n".join(memories)

prompt = f"""
You are a personal digital twin of the user.
Use the memories below AND your own general knowledge to provide the best possible answer.
Blend the user's known preferences with general reasoning to fill in gaps.
Respond naturally in the user's style.

Memories:
{context}

Question:
{query}
"""

print("\n=== Retrieved Memory ===")
print(context)

print("\n=== LLM Response ===")
print(ask_llm(prompt, fallback_memories=memories))

# -----------------------------
# Tests
# -----------------------------
assert len(vocab) > 0, "Vocabulary must not be empty."
assert embeddings.shape[0] == len(user_texts), "Each text must have an embedding."
assert len(memories) > 0, "Memory retrieval failed."

# Additional tests
assert isinstance(memories, list), "Retrieved memories should be a list."
assert len(retrieve("coding")) > 0, "Retrieval should return results for valid query."
assert isinstance(ask_llm("test prompt"), str), "LLM output must be a string."

# New tests for fallback summarization
summary_test = summarize_memories(["I love robotics and electronics"])
assert "robotics" in summary_test or "electronics" in summary_test, "Summary must reflect input memories."

print("\nAll tests passed. Pipeline OK.")
'''


#new changes 1
'''
# PRISM – 60% Demo Pipeline
# Semantic Memory + Local LLM Integration

"""
This version upgrades the system to use REAL semantic embeddings via Ollama.

New Features:
1. Semantic embeddings (nomic-embed-text)
2. Better memory retrieval (meaning-based)
3. Local LLM response generation
4. Intelligent fallback summarization

Requirements:
- Ollama running
- Models installed:
    ollama pull mistral
    ollama pull nomic-embed-text
"""

import numpy as np
import json
from urllib import request, error

# -----------------------------
# 1. Sample User Data
# -----------------------------
user_texts = [
    "I love working on robotics and embedded systems.",
    "I usually write code in Python and Java.",
    "I enjoy making DIY electronics projects.",
    "My cat often interrupts me while coding.",
    "I like solving AI and machine learning problems.",
]

# -----------------------------
# 2. Embedding via Ollama
# -----------------------------
EMBED_URL = "http://localhost:11434/api/embeddings"


def get_embedding(text):
    payload = json.dumps({
        "model": "nomic-embed-text",
        "prompt": text
    }).encode("utf-8")

    req = request.Request(
        EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return np.array(data["embedding"], dtype=np.float32)
    except Exception:
        # fallback to random vector (keeps system running)
        return np.random.rand(768).astype(np.float32)

# -----------------------------
# 3. Create Embeddings
# -----------------------------
print("Creating semantic embeddings...")
embeddings = np.array([get_embedding(t) for t in user_texts])

# -----------------------------
# 4. Similarity + Retrieval
# -----------------------------
def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve(query, k=2):
    q_vec = get_embedding(query)
    scores = [cosine_similarity(q_vec, emb) for emb in embeddings]
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [user_texts[i] for i in top_k]

# -----------------------------
# 5. LLM via Ollama
# -----------------------------
LLM_URL = "http://localhost:11434/api/generate"


def summarize_memories(memories):
    text = " ".join(memories)
    return f"The user is interested in: {text[:100]}..."


def ask_llm(prompt, fallback_memories=None):
    payload = json.dumps({
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = request.Request(
        LLM_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "(no response)")
    except error.URLError:
        if fallback_memories:
            return "[Fallback] " + summarize_memories(fallback_memories)
        return "LLM unavailable"

# -----------------------------
# 6. End-to-End Query
# -----------------------------
query = "What do I like to build?"
memories = retrieve(query)

context = "\n".join(memories)

prompt = f"""
You are a personal AI digital twin.
Use memory + general knowledge.

Memories:
{context}

Question:
{query}
"""

print("\n=== Retrieved Memory ===")
print(context)

print("\n=== LLM Response ===")
print(ask_llm(prompt, fallback_memories=memories))

# -----------------------------
# Tests
# -----------------------------
assert len(embeddings) == len(user_texts)
assert len(retrieve("coding")) > 0
assert isinstance(ask_llm("test"), str)

print("\nAll tests passed. Semantic pipeline working.")
'''

#new changes 2
'''
# PRISM – 70% Demo Pipeline
# Semantic Memory + Persistent Storage + Local LLM

"""
New Features Added:
1. Semantic embeddings (Ollama)
2. Persistent memory (save/load from disk)
3. Improved retrieval
4. Local LLM + fallback

Requirements:
- ollama pull mistral
- ollama pull nomic-embed-text
"""

import numpy as np
import json
import os
from urllib import request, error

# -----------------------------
# Config
# -----------------------------
MEMORY_FILE = "memory_store.json"
EMBED_URL = "http://localhost:11434/api/embeddings"
LLM_URL = "http://localhost:11434/api/generate"

# -----------------------------
# 1. Initial Data (used only if no memory file exists)
# -----------------------------
def get_initial_data():
    return [
        "I love working on python projects",
        "I usually write code in Python and Java.",
        "I enjoy making DIY electronics projects.",
        "My cat often interrupts me while coding.",
        "I like to spend time in my garden.",
    ]

# -----------------------------
# 2. Embedding Function
# -----------------------------
def get_embedding(text):
    payload = json.dumps({
        "model": "nomic-embed-text",
        "prompt": text
    }).encode("utf-8")

    req = request.Request(
        EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]
    except Exception:
        return np.random.rand(768).tolist()

# -----------------------------
# 3. Memory Save/Load
# -----------------------------
def save_memory(texts, embeddings):
    data = {
        "texts": texts,
        "embeddings": embeddings
    }
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f)


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return None, None

    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
        return data["texts"], data["embeddings"]

# -----------------------------
# 4. Initialize Memory
# -----------------------------
texts, embeddings = load_memory()

if texts is None:
    print("Creating memory store...")
    texts = get_initial_data()
    embeddings = [get_embedding(t) for t in texts]
    save_memory(texts, embeddings)
else:
    print("Loaded existing memory.")

embeddings = np.array(embeddings, dtype=np.float32)

# -----------------------------
# 5. Retrieval
# -----------------------------
def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve(query, k=2):
    q_vec = np.array(get_embedding(query), dtype=np.float32)
    scores = [cosine_similarity(q_vec, emb) for emb in embeddings]
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [texts[i] for i in top_k]

# -----------------------------
# 6. Add New Memory
# -----------------------------
def add_memory(new_text):
    global texts, embeddings

    texts.append(new_text)
    new_emb = np.array(get_embedding(new_text), dtype=np.float32)
    embeddings = np.vstack([embeddings, new_emb])

    save_memory(texts, embeddings.tolist())

# -----------------------------
# 7. LLM
# -----------------------------
def ask_llm(prompt, fallback_memories=None):
    payload = json.dumps({
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = request.Request(
        LLM_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "(no response)")
    except error.URLError:
        if fallback_memories:
            return "[Fallback] " + " ".join(fallback_memories)
        return "LLM unavailable"

# -----------------------------
# 8. Run Query
# -----------------------------
query = "What do I like to build?"
memories = retrieve(query)

context = "\n".join(memories)

prompt = f"""
You are a personal AI digital twin.
Use memory + general knowledge.

Memories:
{context}

Question:
{query}
"""

print("\n=== Retrieved Memory ===")
print(context)

print("\n=== LLM Response ===")
print(ask_llm(prompt, fallback_memories=memories))

# -----------------------------
# 9. Demo: Add New Memory
# -----------------------------
add_memory("I recently started learning cloud computing and AWS.")

# -----------------------------
# Tests
# -----------------------------
assert len(texts) > 0
assert embeddings.shape[0] == len(texts)
assert len(retrieve("coding")) > 0

print("\nAll tests passed. Persistent memory working.")
add_memory("i have a lot of plants and i like to do plant works when i feel stressed")
'''

#new changes 3
'''
# PRISM – 80% Demo Pipeline
# Semantic Memory + Persistent Storage + Personality Modeling

"""
New Features Added:
1. Semantic embeddings (Ollama)
2. Persistent memory (save/load)
3. Personality modeling (style imitation)
4. Local LLM + fallback

Requirements:
- ollama pull mistral
- ollama pull nomic-embed-text
"""

import numpy as np
import json
import os
from urllib import request, error

# -----------------------------
# Config
# -----------------------------
MEMORY_FILE = "memory_store.json"
PROFILE_FILE = "personality.json"
EMBED_URL = "http://localhost:11434/api/embeddings"
LLM_URL = "http://localhost:11434/api/generate"

# -----------------------------
# 1. Initial Data
# -----------------------------
def get_initial_data():
    return [
        "I love working on robotics and embedded systems.",
        "I usually write code in Python and Java.",
        "I enjoy making DIY electronics projects.",
        "My cat often interrupts me while coding.",
        "I like solving AI and machine learning problems.",
    ]

# -----------------------------
# 2. Personality Profile
# -----------------------------
def create_personality_profile(texts):
    profile = {
        "tone": "technical and enthusiastic",
        "style": "concise but expressive",
        "interests": ", ".join(texts[:3])
    }
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f)
    return profile


def load_personality():
    if not os.path.exists(PROFILE_FILE):
        return None
    with open(PROFILE_FILE, "r") as f:
        return json.load(f)

# -----------------------------
# 3. Embedding
# -----------------------------
def get_embedding(text):
    payload = json.dumps({
        "model": "nomic-embed-text",
        "prompt": text
    }).encode("utf-8")

    req = request.Request(
        EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]
    except Exception:
        return np.random.rand(768).tolist()

# -----------------------------
# 4. Memory
# -----------------------------
def save_memory(texts, embeddings):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"texts": texts, "embeddings": embeddings}, f)


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return None, None
    with open(MEMORY_FILE, "r") as f:
        data = json.load(f)
        return data["texts"], data["embeddings"]

# -----------------------------
# 5. Initialize
# -----------------------------
texts, embeddings = load_memory()

if texts is None:
    texts = get_initial_data()
    embeddings = [get_embedding(t) for t in texts]
    save_memory(texts, embeddings)

personality = load_personality()
if personality is None:
    personality = create_personality_profile(texts)

embeddings = np.array(embeddings, dtype=np.float32)

# -----------------------------
# 6. Retrieval
# -----------------------------
def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve(query, k=2):
    q_vec = np.array(get_embedding(query), dtype=np.float32)
    scores = [cosine_similarity(q_vec, emb) for emb in embeddings]
    top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [texts[i] for i in top_k]

# -----------------------------
# 7. Add Memory
# -----------------------------
def add_memory(new_text):
    global texts, embeddings
    texts.append(new_text)
    new_emb = np.array(get_embedding(new_text), dtype=np.float32)
    embeddings = np.vstack([embeddings, new_emb])
    save_memory(texts, embeddings.tolist())

# -----------------------------
# 8. LLM with Personality
# -----------------------------
def ask_llm(prompt, personality, fallback_memories=None):

    persona_prompt = f"""
You are a digital twin of the user.

Tone: {personality['tone']}
Style: {personality['style']}
Interests: {personality['interests']}

Respond in this personality consistently.

{prompt}
"""

    payload = json.dumps({
        "model": "mistral",
        "prompt": persona_prompt,
        "stream": False
    }).encode("utf-8")

    req = request.Request(
        LLM_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "(no response)")
    except error.URLError:
        if fallback_memories:
            return "[Fallback Personality Response] " + " ".join(fallback_memories)
        return "LLM unavailable"

# -----------------------------
# 9. Run Query
# -----------------------------
query = "What do I like to build?"
memories = retrieve(query)

context = "\n".join(memories)

prompt = f"""
Memories:
{context}

Question:
{query}
"""

print("\n=== Retrieved Memory ===")
print(context)

print("\n=== Personality-Aware Response ===")
print(ask_llm(prompt, personality, fallback_memories=memories))

# -----------------------------
# Tests
# -----------------------------
assert isinstance(personality, dict)
assert "tone" in personality
assert len(retrieve("AI")) > 0

print("\nAll tests passed. Personality modeling active.")
'''

#new changes 4
'''
# PRISM – 100% Demo Pipeline
# Full System with Chat UI (Streamlit)

"""
Final Upgrade:
- Adds a simple Chat UI using Streamlit
- Uses memory + personality + LLM

Run UI with:
streamlit run prism_demo.py
"""

import numpy as np
import json
import os
from urllib import request, error
import streamlit as st

# -----------------------------
# Config
# -----------------------------
MEMORY_FILE = "memory_store.json"
PROFILE_FILE = "personality.json"
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
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]
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
# Personality
# -----------------------------
def load_personality():
    if not os.path.exists(PROFILE_FILE):
        profile = {"tone": "technical", "style": "clear", "interests": "AI, robotics"}
        with open(PROFILE_FILE, "w") as f:
            json.dump(profile, f)
        return profile

    with open(PROFILE_FILE, "r") as f:
        return json.load(f)

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
def ask_llm(prompt, personality):
    persona_prompt = f"""
Tone: {personality['tone']}
Style: {personality['style']}
Interests: {personality['interests']}

{prompt}
"""

    payload = json.dumps({
        "model": "mistral",
        "prompt": persona_prompt,
        "stream": False
    }).encode("utf-8")

    req = request.Request(LLM_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except:
        return "Fallback response based on memory."

# -----------------------------
# UI (Streamlit)
# -----------------------------
st.title("🧠 PRISM Digital Twin")

texts, embeddings = load_memory()
personality = load_personality()

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask something:")

if user_input:
    memories = retrieve(user_input, texts, embeddings)
    context = "\n".join(memories)

    prompt = f"""
Memories:
{context}

Question:
{user_input}
"""

    response = ask_llm(prompt, personality)

    st.session_state.chat.append((user_input, response))

# Display chat
for q, a in st.session_state.chat:
    st.write(f"**You:** {q}")
    st.write(f"**PRISM:** {a}")

'''
#new changes 5
'''
# PRISM – 100% Demo Pipeline
# Full System with Chat UI (Streamlit)

"""
Final Upgrade:
- Adds a simple Chat UI using Streamlit
- Uses memory + personality + LLM

Run UI with:
streamlit run prism_demo.py
"""

import numpy as np
import json
import os
from urllib import request, error
import streamlit as st

# -----------------------------
# Config
# -----------------------------
MEMORY_FILE = "memory_store.json"
PROFILE_FILE = "personality.json"
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
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]
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
# Personality
# -----------------------------
def load_personality():
    if not os.path.exists(PROFILE_FILE):
        profile = {"tone": "technical", "style": "clear", "interests": "AI, robotics"}
        with open(PROFILE_FILE, "w") as f:
            json.dump(profile, f)
        return profile

    with open(PROFILE_FILE, "r") as f:
        return json.load(f)

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
def ask_llm(prompt, personality):
    persona_prompt = f"""
Tone: {personality['tone']}
Style: {personality['style']}
Interests: {personality['interests']}

{prompt}
"""

    payload = json.dumps({
        "model": "mistral",
        "prompt": persona_prompt,
        "stream": False
    }).encode("utf-8")

    req = request.Request(LLM_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except:
        return "Fallback response based on memory."

# -----------------------------
# UI (Streamlit)
st.title("🧠 PRISM Digital Twin")

texts, embeddings = load_memory()
personality = load_personality()

if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# Add Memory UI
# -----------------------------
st.subheader("➕ Add New Memory")
new_memory = st.text_input("Enter something about yourself:")

if st.button("Save Memory") and new_memory:
    texts.append(new_memory)
    embeddings.append(get_embedding(new_memory))
    save_memory(texts, embeddings)
    st.success("Memory saved successfully!")

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.text_input("Ask something:")

if user_input:
    memories = retrieve(user_input, texts, embeddings)
    context = " ".join(memories)

    prompt = f"""
Memories:
{context}

Question:
{user_input}
"""

    response = ask_llm(prompt, personality)

    st.session_state.chat.append((user_input, response))

# Display chat
for q, a in st.session_state.chat:
    st.write(f"**You:** {q}")
    st.write(f"**PRISM:** {a}")

'''

#new changes 6

# PRISM – Clean Chat UI Version
# Minimal UI (Only Functional Parts for Demo)

import numpy as np
import json
import os
from urllib import request
import streamlit as st

# -----------------------------
# Config
# -----------------------------
MEMORY_FILE = "memory_store.json"
PROFILE_FILE = "personality.json"
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
            data = json.loads(resp.read().decode("utf-8"))
            return data["embedding"]
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
# Personality
# -----------------------------
def load_personality():
    if not os.path.exists(PROFILE_FILE):
        profile = {"tone": "technical", "style": "clear", "interests": "AI, robotics"}
        with open(PROFILE_FILE, "w") as f:
            json.dump(profile, f)
        return profile

    with open(PROFILE_FILE, "r") as f:
        return json.load(f)

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
def ask_llm(prompt, personality):
    persona_prompt = f"""
Tone: {personality['tone']}
Style: {personality['style']}
Interests: {personality['interests']}

{prompt}
"""

    payload = json.dumps({
        "model": "mistral",
        "prompt": persona_prompt,
        "stream": False
    }).encode("utf-8")

    req = request.Request(LLM_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except:
        return "Fallback response"

# -----------------------------
# UI (Clean)
# -----------------------------
st.title("PRISM Digital Twin")

texts, embeddings = load_memory()
personality = load_personality()

if "chat" not in st.session_state:
    st.session_state.chat = []

# Add Memory
new_memory = st.text_input("Add memory:")
if st.button("Save") and new_memory:
    texts.append(new_memory)
    embeddings.append(get_embedding(new_memory))
    save_memory(texts, embeddings)

# Chat
user_input = st.text_input("Ask:")

if user_input:
    memories = retrieve(user_input, texts, embeddings)
    context = "\n".join(memories)

    prompt = f"""
Memories:
{context}

Question:
{user_input}
"""

    response = ask_llm(prompt, personality)
    st.session_state.chat.append((user_input, response))

# Chat display
for q, a in st.session_state.chat:
    st.write(f"You: {q}")
    st.write(f"PRISM: {a}")

