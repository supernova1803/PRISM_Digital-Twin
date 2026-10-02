"""PRISM: a local digital twin chat application with editable, persistent memories."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from urllib import error, request

import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DB_FILE = "prism_memory.db"
LEGACY_MEMORY_FILE = "memory_store.json"
PERSONALITY_FILE = "personality.json"

DEFAULT_EMBED_URL = "http://localhost:11434/api/embeddings"
DEFAULT_LLM_URL = "http://localhost:11434/api/generate"
DEFAULT_TAGS_URL = "http://localhost:11434/api/tags"

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHAT_MODEL = "mistral"

HISTORY_TURNS = 6
SIMILARITY_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Custom Exceptions for Ollama & Retrieval Errors
# ---------------------------------------------------------------------------
class OllamaServiceError(RuntimeError):
    """Raised when Ollama is unreachable, times out, or returns a service failure."""
    pass


class OllamaModelNotFoundError(OllamaServiceError):
    """Raised when a required model is missing from the local Ollama instance."""
    pass


# ---------------------------------------------------------------------------
# Step 2: Embedding & Similarity
# ---------------------------------------------------------------------------
def get_embedding(text, model=DEFAULT_EMBED_MODEL, url=DEFAULT_EMBED_URL):
    """
    Return a normalized embedding vector from Ollama.
    Never uses random-vector fallbacks. Raises OllamaServiceError or OllamaModelNotFoundError.
    """
    clean_text = str(text).strip()
    if not clean_text:
        raise ValueError("Cannot generate embedding for empty text.")

    payload = json.dumps({"model": model, "prompt": clean_text}).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))

        if "embedding" not in data:
            raise ValueError("Ollama response did not contain an 'embedding' field.")

        vector = np.asarray(data["embedding"], dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("Ollama returned an empty or invalid embedding vector.")

        return vector.tolist()

    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 or "not found" in detail.lower():
            raise OllamaModelNotFoundError(
                f"Embedding model '{model}' was not found in Ollama.\n"
                f"Setup hint: Run `ollama pull {model}` in your terminal to install it."
            ) from exc
        raise OllamaServiceError(
            f"Ollama embedding request failed (HTTP {exc.code}): {detail or exc.reason}"
        ) from exc

    except (error.URLError, OSError) as exc:
        raise OllamaServiceError(
            f"Cannot reach Ollama at {url}.\n"
            f"Setup hint: Make sure Ollama is running (`ollama serve` or open the Ollama desktop app)."
        ) from exc

    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        raise OllamaServiceError(f"Could not parse embedding from Ollama: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, OllamaServiceError):
            raise
        raise OllamaServiceError(f"Unexpected error communicating with Ollama: {exc}") from exc


def cosine_similarity(a, b):
    """Compute cosine similarity between two 1D vectors with dimension checking."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Embeddings must be 1-dimensional vectors.")

    if a.size != b.size:
        raise ValueError(f"Embedding dimension mismatch: vector A has {a.size} dims, vector B has {b.size} dims.")

    if a.size == 0:
        return 0.0

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def retrieve(query, memories, conn=None, k=3, threshold=SIMILARITY_THRESHOLD,
             model=DEFAULT_EMBED_MODEL, url=DEFAULT_EMBED_URL):
    """
    Retrieve relevant memories for a query.
    - Handles empty memory list by immediately returning [].
    - Validates query embedding.
    - Checks embedding dimensions match before comparing.
    - Scores memories and only returns those with similarity >= threshold.
    - If nothing meets threshold, returns [].
    """
    if not memories:
        return []

    clean_query = str(query).strip()
    if not clean_query:
        return []

    query_vector = get_embedding(clean_query, model=model, url=url)

    scored = []
    for memory in memories:
        emb_data = memory["embedding"] if isinstance(memory, (dict, sqlite3.Row)) else memory[3]
        mem_text = memory["text"] if isinstance(memory, (dict, sqlite3.Row)) else memory[1]
        mem_id = memory["id"] if isinstance(memory, (dict, sqlite3.Row)) else memory[0]

        if not emb_data:
            # Generate missing embedding and persist if DB connection provided
            try:
                vector = get_embedding(mem_text, model=model, url=url)
                if conn is not None:
                    conn.execute("UPDATE memories SET embedding=? WHERE id=?", (json.dumps(vector), mem_id))
                    conn.commit()
            except OllamaServiceError:
                continue
        else:
            if isinstance(emb_data, str):
                try:
                    vector = json.loads(emb_data)
                except json.JSONDecodeError:
                    continue
            else:
                vector = emb_data

        if not isinstance(vector, (list, np.ndarray)) or len(query_vector) != len(vector):
            # Dimension mismatch or corrupted vector, skip to avoid crashes
            continue

        score = cosine_similarity(query_vector, vector)
        if score >= threshold:
            scored.append((score, mem_text, mem_id))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for score, text, _id in scored[:k]]


# ---------------------------------------------------------------------------
# Step 1: Database & Memory Persistence + Chat Persistence
# ---------------------------------------------------------------------------
def connect_db(db_file=DB_FILE):
    """Establish connection to SQLite memory database with schema initialization."""
    conn = sqlite3.connect(db_file, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            embedding TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            memories TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def initialize_memory_store(db_file=DB_FILE, legacy_file=LEGACY_MEMORY_FILE):
    """
    Initialize SQLite store and perform a one-time migration from legacy JSON
    without losing existing memories or precomputed embeddings.
    """
    conn = connect_db(db_file)
    imported = conn.execute("SELECT value FROM app_state WHERE key='legacy_imported'").fetchone()

    if imported is None:
        try:
            if os.path.exists(legacy_file):
                with open(legacy_file, "r", encoding="utf-8") as source:
                    legacy = json.load(source)
                texts = legacy.get("texts", []) if isinstance(legacy, dict) else []
                embeddings = legacy.get("embeddings", []) if isinstance(legacy, dict) else []
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

                rows = []
                for idx, text in enumerate(texts):
                    cleaned = str(text).strip()
                    if not cleaned:
                        continue
                    emb_str = None
                    if idx < len(embeddings) and embeddings[idx]:
                        emb_str = json.dumps(embeddings[idx])
                    rows.append((cleaned, now, emb_str))

                if rows:
                    conn.executemany(
                        "INSERT INTO memories(text, created_at, embedding) VALUES(?, ?, ?)",
                        rows
                    )

            conn.execute("INSERT OR REPLACE INTO app_state(key, value) VALUES('legacy_imported', 'true')")
            conn.commit()
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            conn.close()
            raise RuntimeError(f"Could not import existing {legacy_file}: {exc}") from exc

    return conn


def list_memories(conn, search_query=None):
    """Return all stored memories as a list of sqlite3.Row objects."""
    if search_query and search_query.strip():
        return conn.execute(
            "SELECT id, text, created_at, embedding FROM memories WHERE text LIKE ? ORDER BY id ASC",
            (f"%{search_query.strip()}%",)
        ).fetchall()
    return conn.execute("SELECT id, text, created_at, embedding FROM memories ORDER BY id ASC").fetchall()


def add_memory(conn, text, model=DEFAULT_EMBED_MODEL, url=DEFAULT_EMBED_URL):
    """Add a new memory with text, timestamp, and generated embedding."""
    clean_text = str(text).strip()
    if not clean_text:
        raise ValueError("Memory text cannot be empty.")

    vector = get_embedding(clean_text, model=model, url=url)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO memories(text, created_at, embedding) VALUES(?, ?, ?)",
        (clean_text, now, json.dumps(vector))
    )
    conn.commit()
    return cur.lastrowid


def update_memory(conn, memory_id, text, model=DEFAULT_EMBED_MODEL, url=DEFAULT_EMBED_URL):
    """Update memory text and regenerate its embedding in SQLite."""
    clean_text = str(text).strip()
    if not clean_text:
        raise ValueError("Memory text cannot be empty.")

    vector = get_embedding(clean_text, model=model, url=url)
    conn.execute(
        "UPDATE memories SET text=?, embedding=? WHERE id=?",
        (clean_text, json.dumps(vector), memory_id)
    )
    conn.commit()


def delete_memory(conn, memory_id):
    """Delete a memory and its embedding from SQLite."""
    conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    conn.commit()


def clear_all_memories(conn):
    """Delete all memories and embeddings from SQLite."""
    conn.execute("DELETE FROM memories")
    conn.commit()


# ---------------------------------------------------------------------------
# Persistent Chat Storage (SQLite)
# ---------------------------------------------------------------------------
def list_chat_messages(conn):
    """Load persistent chat conversation from SQLite."""
    rows = conn.execute(
        "SELECT id, role, content, memories, created_at FROM chat_messages ORDER BY id ASC"
    ).fetchall()
    messages = []
    for r in rows:
        mems = []
        if r["memories"]:
            try:
                mems = json.loads(r["memories"])
            except json.JSONDecodeError:
                mems = []
        messages.append({
            "id": r["id"],
            "role": r["role"],
            "content": r["content"],
            "memories": mems,
            "created_at": r["created_at"]
        })
    return messages


def save_chat_message(conn, role, content, memories=None):
    """Save a chat message to SQLite so it is preserved across reloads."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    mem_json = json.dumps(memories) if memories else None
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO chat_messages (role, content, memories, created_at) VALUES (?, ?, ?, ?)",
        (role, content, mem_json, now)
    )
    conn.commit()
    return cur.lastrowid


def clear_chat_history(conn):
    """Clear chat conversation from SQLite without deleting saved memories."""
    conn.execute("DELETE FROM chat_messages")
    conn.commit()


# ---------------------------------------------------------------------------
# Step 4: Prompt Construction & Personality
# ---------------------------------------------------------------------------
def load_personality(file_path=PERSONALITY_FILE):
    """Load assistant personality profile if available."""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "tone": "technical, enthusiastic, and warm",
        "style": "concise, articulate, and grounded",
        "interests": "AI, robotics, programming, electronics"
    }


def build_prompt(history, user_text, memories, max_turns=HISTORY_TURNS, personality=None):
    """
    Build a concise, grounded prompt:
    - Retains recent conversation turns for context.
    - Explicitly grounds personal user facts in retrieved memories.
    - Prohibits hallucinating personal details when memories don't answer.
    - Permits general knowledge for general inquiries without claiming it as memories.
    - Clearly marks when no relevant memories are found.
    """
    recent = history[-(max_turns * 2):] if history else []
    convo_lines = []
    for item in recent:
        speaker = "User" if item.get("role") == "user" else "PRISM"
        convo_lines.append(f"{speaker}: {item.get('content', '')}")
    conversation = "\n".join(convo_lines) if convo_lines else "(Start of conversation)"

    if memories:
        memory_section = "\n".join(f"- {text}" for text in memories)
    else:
        memory_section = "None found. Do NOT invent or extrapolate personal details about the user."

    persona_notes = ""
    if personality and isinstance(personality, dict):
        tone = personality.get("tone", "")
        style = personality.get("style", "")
        if tone or style:
            persona_notes = f"\nPersona Guidance: Speak with a {tone} tone and a {style} style."

    return f"""You are PRISM, a personalized digital twin and assistant.{persona_notes}

Core Instructions:
1. Grounding in Memories: Use the retrieved memories below ONLY as verified evidence for personal facts, preferences, background, and projects of the user.
2. No Hallucinated Facts: Never invent or assume personal details about the user. If the retrieved memories do not answer a personal question about the user, clearly state that you do not know or do not have that stored in your memories.
3. General Knowledge: You are free to answer general inquiries (such as math, science, code, history, general facts) using your general knowledge, but DO NOT claim that general knowledge represents a personal memory of the user.
4. Conversation Context: Use the recent conversation history to understand follow-up questions, pronouns, and references.

Retrieved Memories:
{memory_section}

Recent Conversation:
{conversation}

User: {user_text}
PRISM:"""


def ask_llm(prompt, model=DEFAULT_CHAT_MODEL, url=DEFAULT_LLM_URL):
    """Send prompt to Ollama generate endpoint."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))

        answer = data.get("response", "").strip()
        if not answer:
            raise OllamaServiceError("Ollama returned an empty response.")
        return answer

    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404 or "not found" in detail.lower():
            raise OllamaModelNotFoundError(
                f"Chat model '{model}' was not found in Ollama.\n"
                f"Setup hint: Run `ollama pull {model}` in your terminal to install it."
            ) from exc
        raise OllamaServiceError(
            f"Ollama chat request failed (HTTP {exc.code}): {detail or exc.reason}"
        ) from exc

    except (error.URLError, OSError) as exc:
        raise OllamaServiceError(
            f"Cannot reach Ollama at {url}.\n"
            f"Setup hint: Start Ollama by running `ollama serve` or launching the Ollama desktop app."
        ) from exc

    except json.JSONDecodeError as exc:
        raise OllamaServiceError(f"Ollama returned an unreadable response: {exc}") from exc

    except Exception as exc:
        if isinstance(exc, OllamaServiceError):
            raise
        raise OllamaServiceError(f"Unexpected error communicating with Ollama: {exc}") from exc


def generate_local_twin_response(user_text, memories, history, personality=None):
    """
    Intelligent embedded conversational response engine.
    Used when an Ollama chat model is not yet installed or offline, ensuring
    the user can always chat normally and get accurate answers based on their memories.
    """
    q_lower = user_text.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "howdy"]

    # Simple greeting check
    if any(q_lower == g or q_lower.startswith(g + " ") for g in greetings) and len(user_text.split()) <= 4:
        return (
            "Hello! I am PRISM, your digital twin assistant. I'm connected to your persistent "
            "memories. What would you like to discuss or work on today?"
        )

    # If relevant memories were retrieved, synthesize them directly
    if memories:
        bullets = "\n".join(f"• {m}" for m in memories)
        return (
            f"Here is what I remember about that from your personal profile:\n\n{bullets}\n\n"
            "Would you like me to elaborate on any of these, or update what I remember?"
        )

    # Personal question without memory
    personal_keywords = ["my", "i", "do i", "what do i", "am i", "my favorite", "where do i", "who am i"]
    if any(k in q_lower for k in personal_keywords):
        return (
            "I searched my memory store, but I don't have a record of that personal fact about you yet.\n\n"
            "💡 *Tip:* You can tell me right here, or switch to the **🧠 Memories** tab to add it!"
        )

    # General question fallback
    return (
        f"I received your inquiry: \"{user_text}\".\n\n"
        "I didn't find any specific personal memories related to this question. "
        "For full conversational depth, make sure a generative Ollama chat model (like `qwen2.5:0.5b` or `mistral`) "
        "is selected in the sidebar."
    )


def get_available_models(tags_url=DEFAULT_TAGS_URL):
    """Query Ollama tags endpoint to list installed models."""
    req = request.Request(tags_url, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def get_chat_models(installed_models):
    """Filter models to identify likely generative chat models (excluding pure embedding models)."""
    embed_keywords = ["embed", "bge-", "all-minilm", "arctic"]
    chat_models = [m for m in installed_models if not any(kw in m.lower() for kw in embed_keywords)]
    return chat_models


# ---------------------------------------------------------------------------
# Streamlit CSS Styling: Dynamic Dark & Light Themes with Aurora Accents
# ---------------------------------------------------------------------------
def apply_theme_styles(theme="Dark"):
    """Inject modern, responsive CSS styling for Dark or Light mode."""
    is_dark = (theme == "Dark")

    # Dark Theme Palette
    dark_css = """
        --bg-app: radial-gradient(circle at 50% 0%, #171d34 0%, #0a0d18 60%, #060810 100%);
        --bg-card: rgba(18, 24, 43, 0.85);
        --bg-hero: linear-gradient(135deg, rgba(29, 27, 75, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        --bg-metric: rgba(22, 30, 54, 0.7);
        --bg-sidebar: #0b0f1d;
        --border-card: rgba(99, 102, 241, 0.25);
        --border-metric: rgba(99, 102, 241, 0.2);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --shadow-card: 0 10px 25px rgba(0, 0, 0, 0.35);
        --user-bubble-bg: linear-gradient(135deg, rgba(79, 70, 229, 0.85) 0%, rgba(37, 99, 235, 0.85) 100%);
        --user-bubble-border: rgba(129, 140, 248, 0.4);
        --bot-bubble-bg: rgba(20, 28, 48, 0.85);
        --bot-bubble-border: rgba(51, 65, 85, 0.7);
        --accent-glow: rgba(99, 102, 241, 0.35);
    """

    # Light Theme Palette
    light_css = """
        --bg-app: radial-gradient(circle at 50% 0%, #edf2ff 0%, #f8fafc 55%, #f1f5f9 100%);
        --bg-card: rgba(255, 255, 255, 0.92);
        --bg-hero: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(243, 244, 246, 0.9) 100%);
        --bg-metric: rgba(255, 255, 255, 0.85);
        --bg-sidebar: #f8fafc;
        --border-card: rgba(99, 102, 241, 0.2);
        --border-metric: rgba(226, 232, 240, 0.9);
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #94a3b8;
        --shadow-card: 0 10px 30px rgba(99, 102, 241, 0.08);
        --user-bubble-bg: linear-gradient(135deg, #e0e7ff 0%, #dbeafe 100%);
        --user-bubble-border: #a5b4fc;
        --bot-bubble-bg: #ffffff;
        --bot-bubble-border: #e2e8f0;
        --accent-glow: rgba(99, 102, 241, 0.15);
    """

    selected_palette = dark_css if is_dark else light_css

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        {selected_palette}
        --aurora-cyan: #06b6d4;
        --aurora-indigo: #6366f1;
        --aurora-pink: #ec4899;
        --aurora-emerald: #10b981;
    }}

    html, body, [class*="st-"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: var(--bg-app) !important;
        color: var(--text-primary) !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-card) !important;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, .stTitle {{
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.025em;
        color: var(--text-primary) !important;
    }}

    /* Hero Header Card */
    .prism-hero {{
        background: var(--bg-hero);
        border: 1px solid var(--border-card);
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: var(--shadow-card);
        backdrop-filter: blur(12px);
        display: flex;
        flex-direction: column;
        gap: 8px;
        position: relative;
        overflow: hidden;
    }}
    .prism-hero::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--aurora-cyan), var(--aurora-indigo), var(--aurora-pink));
    }}

    .prism-title {{
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 45%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.2;
    }}

    .prism-subtitle {{
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin: 0;
    }}

    /* Aurora Badges */
    .badge-bar {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 4px;
    }}
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-cyan {{
        background: rgba(6, 182, 212, 0.15);
        border: 1px solid rgba(6, 182, 212, 0.4);
        color: var(--aurora-cyan);
    }}
    .badge-indigo {{
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.4);
        color: #818cf8;
    }}
    .badge-emerald {{
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: var(--aurora-emerald);
    }}

    /* Stat Tiles */
    .stat-tile {{
        background: var(--bg-metric);
        border: 1px solid var(--border-metric);
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }}
    .stat-tile:hover {{
        transform: translateY(-2px);
        border-color: var(--aurora-indigo);
    }}
    .stat-label {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        font-weight: 600;
    }}
    .stat-value {{
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        color: var(--text-primary);
        margin-top: 2px;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background: transparent;
        border-bottom: 2px solid var(--border-metric);
        padding-bottom: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        color: var(--aurora-indigo) !important;
        border-bottom: 2px solid var(--aurora-indigo) !important;
    }}

    /* Chat Messages */
    [data-testid="stChatMessage"] {{
        border-radius: 14px !important;
        margin-bottom: 12px !important;
        padding: 12px 18px !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
        background: var(--user-bubble-bg) !important;
        border: 1px solid var(--user-bubble-border) !important;
        border-bottom-right-radius: 4px !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
        background: var(--bot-bubble-bg) !important;
        border: 1px solid var(--bot-bubble-border) !important;
        border-bottom-left-radius: 4px !important;
    }}

    /* Primary Buttons with Aurora Gradient */
    .stButton button[kind="primary"], .stButton button[data-testid="baseButton-primary"] {{
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
        transition: all 0.2s ease !important;
    }}
    .stButton button[kind="primary"]:hover, .stButton button[data-testid="baseButton-primary"]:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
    }}

    /* Inputs */
    .stTextInput input, .stTextArea textarea {{
        border-radius: 10px !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Streamlit Application
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PRISM Digital Twin",
        page_icon="💎",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Database Initialization (Memories + Chat persistence)
    try:
        db = initialize_memory_store()
    except RuntimeError as exc:
        st.error(f"Database error: {exc}")
        st.stop()

    # Load personality
    personality = load_personality()

    # Session State Initialization
    if "theme" not in st.session_state:
        st.session_state.theme = "Dark"

    # Initialize persistent chat from SQLite once per session
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = list_chat_messages(db)

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 💎 PRISM Control")
        st.caption("Local digital twin with persistent SQLite memory & semantic retrieval.")

        st.divider()

        # Theme Switcher Option (Dark / Light)
        st.markdown("##### 🎨 Appearance & Theme")
        theme_choice = st.radio(
            "Select Theme:",
            options=["🌙 Dark", "☀️ Light"],
            index=0 if st.session_state.theme == "Dark" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        current_theme = "Dark" if "Dark" in theme_choice else "Light"
        st.session_state.theme = current_theme

        st.divider()

        # Model Selection & Status
        st.markdown("##### ⚙️ Models & Engine")
        installed_models = get_available_models()
        chat_model_candidates = get_chat_models(installed_models)

        embed_model = st.text_input(
            "Embedding Model",
            value=DEFAULT_EMBED_MODEL,
            help="Model used for semantic memory embeddings in Ollama"
        )

        model_options = []
        if chat_model_candidates:
            model_options.extend(chat_model_candidates)
        model_options.append("🧠 PRISM Cognitive Core (Local Engine)")

        chat_model_selection = st.selectbox(
            "Chat Model",
            options=model_options,
            index=0,
            help="Active language model for conversation"
        )

        # Connection status indicator
        if installed_models:
            st.success(f"🟢 Ollama Connected ({len(installed_models)} models)")
        else:
            st.warning("🟡 Ollama: Check server (`ollama serve`)")

        all_memories = list_memories(db)
        st.metric("Total Saved Memories", len(all_memories))

        st.divider()

        st.markdown("##### 💬 Chat Controls")
        if st.button("🧹 Clear Chat History", use_container_width=True,
                     help="Clears conversation messages from SQLite while leaving memories intact"):
            clear_chat_history(db)
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        with st.expander("🎭 Personality Profile", expanded=False):
            st.write(f"**Tone:** {personality.get('tone', 'N/A')}")
            st.write(f"**Style:** {personality.get('style', 'N/A')}")
            st.write(f"**Interests:** {personality.get('interests', 'N/A')}")

    # Apply CSS for chosen theme
    apply_theme_styles(st.session_state.theme)

    # -----------------------------------------------------------------------
    # Hero Header with Aurora Colors & Badges
    # -----------------------------------------------------------------------
    active_model_name = (
        chat_model_selection
        if chat_model_selection != "🧠 PRISM Cognitive Core (Local Engine)"
        else "PRISM Engine"
    )

    st.markdown(f"""
        <div class="prism-hero">
            <div class="prism-title">PRISM Digital Twin</div>
            <p class="prism-subtitle">
                Your intelligent, memory-grounded personal assistant with real-time SQLite persistence.
            </p>
            <div class="badge-bar">
                <span class="badge badge-indigo">💎 Twin Active</span>
                <span class="badge badge-cyan">🧠 {len(all_memories)} Memories Loaded</span>
                <span class="badge badge-emerald">⚡ {active_model_name}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Main Tabs: Chat & Memories View
    # -----------------------------------------------------------------------
    tab_chat, tab_memories = st.tabs(["💬 Chat with PRISM", "🧠 Memories Manager"])

    # -----------------------------------------------------------------------
    # Tab 1: Chat Interface
    # -----------------------------------------------------------------------
    with tab_chat:
        # Stat overview row
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.markdown(f"""
                <div class="stat-tile">
                    <div class="stat-label">Conversation Turns</div>
                    <div class="stat-value">{len(st.session_state.chat_history)}</div>
                </div>
            """, unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""
                <div class="stat-tile">
                    <div class="stat-label">Active Memories</div>
                    <div class="stat-value">{len(all_memories)}</div>
                </div>
            """, unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""
                <div class="stat-tile">
                    <div class="stat-label">Engine Mode</div>
                    <div class="stat-value" style="font-size: 1.1rem; color: #06b6d4; margin-top: 8px;">
                        {'Ollama (' + chat_model_selection + ')' if 'Cognitive Core' not in chat_model_selection else 'PRISM Core'}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Render conversation history (loaded from SQLite)
        if not st.session_state.chat_history:
            st.info("👋 Hello! Start chatting below. PRISM remembers your background and will ground its answers in your memories.")

        for item in st.session_state.chat_history:
            role = item.get("role", "user")
            content = item.get("content", "")
            memories_used = item.get("memories")

            with st.chat_message(role):
                st.markdown(content)
                if role == "assistant" and memories_used is not None:
                    if memories_used:
                        with st.expander(f"🔍 Memories used ({len(memories_used)})", expanded=False):
                            for mem in memories_used:
                                st.markdown(f"- {mem}")
                    else:
                        st.caption("ℹ️ No relevant memories were found or used for this response.")

        # Chat Input
        user_input = st.chat_input("Ask PRISM about your projects, preferences, or anything else...")
        if user_input:
            # 1. Save user message to SQLite and state
            save_chat_message(db, "user", user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # 2. Assistant Turn
            with st.chat_message("assistant"):
                with st.spinner("PRISM is thinking..."):
                    # Build retrieval query using immediate prior user turns for follow-ups
                    prior_user = [
                        m["content"] for m in st.session_state.chat_history[:-1]
                        if m.get("role") == "user"
                    ]
                    retrieval_query = "\n".join(prior_user[-2:] + [user_input]) if prior_user else user_input

                    # Retrieve relevant memories
                    try:
                        relevant = retrieve(
                            retrieval_query,
                            list_memories(db),
                            conn=db,
                            k=3,
                            threshold=SIMILARITY_THRESHOLD,
                            model=embed_model
                        )
                    except OllamaServiceError as emb_err:
                        st.warning(f"Memory retrieval note: {emb_err}")
                        relevant = []

                    # Generate reply
                    answer = ""
                    use_local_engine = ("Cognitive Core" in chat_model_selection)

                    if not use_local_engine:
                        try:
                            prompt = build_prompt(
                                history=st.session_state.chat_history[:-1],
                                user_text=user_input,
                                memories=relevant,
                                max_turns=HISTORY_TURNS,
                                personality=personality
                            )
                            answer = ask_llm(prompt, model=chat_model_selection)
                        except (OllamaModelNotFoundError, OllamaServiceError) as exc:
                            # Gracefully fall back to local twin engine so chat continues normally
                            st.info(f"💡 Note: {exc}\nSwitching to PRISM Core for this response.")
                            answer = generate_local_twin_response(
                                user_input,
                                relevant,
                                st.session_state.chat_history[:-1],
                                personality
                            )
                    else:
                        answer = generate_local_twin_response(
                            user_input,
                            relevant,
                            st.session_state.chat_history[:-1],
                            personality
                        )

                    st.markdown(answer)

                    # Show memories used under answer
                    if relevant:
                        with st.expander(f"🔍 Memories used ({len(relevant)})", expanded=False):
                            for mem in relevant:
                                st.markdown(f"- {mem}")
                    else:
                        st.caption("ℹ️ No relevant memories were found or used for this response.")

                    # Save assistant reply to SQLite and session state
                    save_chat_message(db, "assistant", answer, relevant)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "memories": relevant
                    })

    # -----------------------------------------------------------------------
    # Tab 2: Memories View & Manager
    # -----------------------------------------------------------------------
    with tab_memories:
        st.subheader("🧠 Memories Manager")
        st.caption("Review, add, edit, or delete personal facts and preferences stored in SQLite.")

        # Add Memory Form
        with st.form("add_memory_form", clear_on_submit=True):
            st.markdown("##### ➕ Add New Memory")
            new_text = st.text_area(
                "Enter a personal fact or preference PRISM should remember:",
                placeholder="e.g., I am designing a robotic arm using ROS 2 and ESP32.",
                height=90
            )
            add_submitted = st.form_submit_button("Save Memory", use_container_width=True, type="primary")

        if add_submitted:
            if not new_text.strip():
                st.warning("Please enter some text before saving.")
            else:
                try:
                    new_id = add_memory(db, new_text.strip(), model=embed_model)
                    st.success(f"Memory #{new_id} saved successfully with embedding!")
                    st.rerun()
                except OllamaServiceError as exc:
                    st.error(f"Could not generate embedding for memory: {exc}")
                except Exception as exc:
                    st.error(f"Failed to save memory: {exc}")

        st.divider()

        # Search & List Memories
        search_kw = st.text_input("🔍 Search Memories", placeholder="Filter memories by keyword...")
        memories = list_memories(db, search_query=search_kw)

        st.markdown(f"##### 📋 Saved Memories ({len(memories)})")

        if not memories:
            st.info("No memories match your query." if search_kw else "No memories saved yet. Add one above!")
        else:
            table_data = [
                {
                    "ID": row["id"],
                    "Created (UTC)": row["created_at"],
                    "Memory Content": row["text"],
                    "Has Embedding": "✅" if row["embedding"] else "❌"
                }
                for row in memories
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("##### ✏️ Edit or 🗑️ Delete a Memory")

            options = {row["id"]: f"#{row['id']}: {row['text'][:70]}..." for row in memories}
            selected_id = st.selectbox(
                "Choose a memory to modify:",
                options=list(options.keys()),
                format_func=lambda vid: options[vid]
            )

            selected_row = next(r for r in memories if r["id"] == selected_id)

            col_edit, col_del = st.columns([3, 1])

            with col_edit:
                with st.form("edit_memory_form"):
                    updated_text = st.text_area(
                        "Edit memory text:",
                        value=selected_row["text"],
                        key=f"edit_val_{selected_id}",
                        height=100
                    )
                    edit_submitted = st.form_submit_button("Update Memory (Regenerates Embedding)", use_container_width=True)

                if edit_submitted:
                    if not updated_text.strip():
                        st.warning("Memory text cannot be empty.")
                    else:
                        try:
                            update_memory(db, selected_id, updated_text.strip(), model=embed_model)
                            st.success(f"Memory #{selected_id} updated and re-embedded successfully!")
                            st.rerun()
                        except OllamaServiceError as exc:
                            st.error(f"Could not regenerate embedding: {exc}")
                        except Exception as exc:
                            st.error(f"Failed to update memory: {exc}")

            with col_del:
                st.markdown("**Danger Zone**")
                if st.button(f"🗑️ Delete Memory #{selected_id}", key=f"del_{selected_id}", use_container_width=True):
                    delete_memory(db, selected_id)
                    st.success(f"Memory #{selected_id} deleted.")
                    st.rerun()

            st.divider()
            st.markdown("##### ⚠️ Clear All Memories")
            confirm_clear = st.checkbox("I understand that this permanently deletes all saved memories from SQLite.")
            if st.button("Delete All Memories", disabled=not confirm_clear, type="primary"):
                clear_all_memories(db)
                st.success("All memories deleted.")
                st.rerun()


if __name__ == "__main__":
    main()
