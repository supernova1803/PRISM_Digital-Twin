"""PRISM: a local chat app with editable, persistent personal memories."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from urllib import error, request

import numpy as np
import streamlit as st


DB_FILE = "prism_memory.db"
LEGACY_MEMORY_FILE = "memory_store.json"
EMBED_URL = "http://localhost:11434/api/embeddings"
LLM_URL = "http://localhost:11434/api/generate"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "mistral"
HISTORY_TURNS = 6


def get_embedding(text):
    """Return an Ollama embedding or raise a readable error; never fake one."""
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = request.Request(EMBED_URL, data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        vector = np.asarray(data["embedding"], dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("Ollama returned an invalid embedding.")
        return vector.tolist()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama embedding request failed ({exc.code}): {detail or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError("Cannot reach Ollama at localhost:11434. Start Ollama and install the "
                           f"{EMBED_MODEL} model (`ollama pull {EMBED_MODEL}`).") from exc
    except (KeyError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Could not read an embedding from Ollama: {exc}") from exc


def connect_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL,
        embedding TEXT
    )""")
    conn.execute("CREATE TABLE IF NOT EXISTS app_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    return conn


def initialize_memory_store():
    """Create the SQLite store and migrate legacy memory text once, without losing it."""
    conn = connect_db()
    imported = conn.execute("SELECT value FROM app_state WHERE key='legacy_imported'").fetchone()
    if imported is None:
        try:
            if os.path.exists(LEGACY_MEMORY_FILE):
                with open(LEGACY_MEMORY_FILE, "r", encoding="utf-8") as source:
                    legacy = json.load(source)
                texts = legacy.get("texts", []) if isinstance(legacy, dict) else []
                now = datetime.now(timezone.utc).isoformat()
                conn.executemany("INSERT INTO memories(text, created_at) VALUES(?, ?)",
                                 [(str(text).strip(), now) for text in texts if str(text).strip()])
            conn.execute("INSERT INTO app_state(key, value) VALUES('legacy_imported', 'true')")
            conn.commit()
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            conn.close()
            raise RuntimeError(f"Could not import existing {LEGACY_MEMORY_FILE}: {exc}") from exc
    return conn


def list_memories(conn):
    return conn.execute("SELECT id, text, created_at, embedding FROM memories ORDER BY id").fetchall()


def add_memory(conn, text):
    vector = get_embedding(text)
    conn.execute("INSERT INTO memories(text, created_at, embedding) VALUES(?, ?, ?)",
                 (text, datetime.now(timezone.utc).isoformat(), json.dumps(vector)))
    conn.commit()


def update_memory(conn, memory_id, text):
    vector = get_embedding(text)
    conn.execute("UPDATE memories SET text=?, embedding=? WHERE id=?",
                 (text, json.dumps(vector), memory_id))
    conn.commit()


def delete_memory(conn, memory_id):
    conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
    conn.commit()


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.size != b.size:
        raise RuntimeError("Memory embeddings have inconsistent dimensions. Re-save the affected memory.")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieve(query, memories, k=3):
    if not memories:
        return []
    query_vector = get_embedding(query)
    scored = []
    for memory in memories:
        if not memory["embedding"]:
            vector = get_embedding(memory["text"])
            memory_id = memory["id"]
            # Cache vectors from legacy imports; if Ollama fails, report the error.
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("UPDATE memories SET embedding=? WHERE id=?", (json.dumps(vector), memory_id))
            memory = dict(memory)
            memory["embedding"] = json.dumps(vector)
        vector = json.loads(memory["embedding"])
        scored.append((cosine_similarity(query_vector, vector), memory["text"]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [text for score, text in scored[:min(k, len(scored))] if score > 0]


def ask_llm(prompt):
    payload = json.dumps({"model": CHAT_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = request.Request(LLM_URL, data=payload,
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        answer = data.get("response", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response.")
        return answer
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama chat request failed ({exc.code}): {detail or exc.reason}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach Ollama. Start it and install the {CHAT_MODEL} model "
                           f"(`ollama pull {CHAT_MODEL}`).") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned an unreadable response: {exc}") from exc


def build_prompt(history, user_text, memories):
    recent = history[-HISTORY_TURNS * 2:]
    conversation = "\n".join(
        f"{('User' if item['role'] == 'user' else 'PRISM')}: {item['content']}"
        for item in recent
    )
    memory_context = "\n".join(f"- {text}" for text in memories) or "No relevant memories found."
    return f"""You are PRISM, a helpful personal assistant.
Use the conversation to understand follow-up questions. Use memories only as evidence
about the user's personal facts. Do not invent personal details; if memories do not
answer a personal question, say you do not know. You may answer general questions
using general knowledge.

Relevant memories:
{memory_context}

Recent conversation:
{conversation or '(start of conversation)'}

User: {user_text}
PRISM:"""


st.set_page_config(page_title="PRISM Digital Twin", layout="wide")
st.title("PRISM Digital Twin")
st.caption("A local assistant with memories you can review and edit.")

try:
    db = initialize_memory_store()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.sidebar:
    st.header("Memory manager")
    with st.form("add_memory_form", clear_on_submit=True):
        new_text = st.text_area("Add a memory", placeholder="Something PRISM should remember")
        add_clicked = st.form_submit_button("Save memory")
    if add_clicked:
        if not new_text.strip():
            st.warning("Enter some text before saving.")
        else:
            try:
                add_memory(db, new_text.strip())
                st.success("Memory saved.")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))

    memories = list_memories(db)
    st.caption(f"{len(memories)} saved memor{'y' if len(memories) == 1 else 'ies'}")
    if memories:
        labels = {row["id"]: f"{row['id']}: {row['text'][:70]}" for row in memories}
        selected_id = st.selectbox("Choose a memory to edit", options=list(labels),
                                   format_func=lambda value: labels[value])
        selected = next(row for row in memories if row["id"] == selected_id)
        with st.form("edit_memory_form"):
            edited_text = st.text_area("Memory text", value=selected["text"], key=f"edit_{selected_id}")
            save_edit = st.form_submit_button("Update memory")
        if save_edit:
            if not edited_text.strip():
                st.warning("Memory text cannot be empty.")
            else:
                try:
                    update_memory(db, selected_id, edited_text.strip())
                    st.success("Memory updated.")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
        if st.button("Delete selected memory", key="delete_memory"):
            delete_memory(db, selected_id)
            st.rerun()
        if st.button("Clear all memories"):
            db.execute("DELETE FROM memories")
            db.commit()
            st.rerun()
    if st.button("Clear chat history"):
        st.session_state.chat = []
        st.rerun()

for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant" and item.get("memories"):
            with st.expander("Memories used"):
                for memory in item["memories"]:
                    st.write(f"- {memory}")

user_text = st.chat_input("Ask PRISM something...")
if user_text:
    st.session_state.chat.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)
    with st.chat_message("assistant"):
        try:
            # Include the immediate context when retrieving so short follow-ups resolve.
            prior_user = [item["content"] for item in st.session_state.chat[:-1]
                          if item["role"] == "user"]
            retrieval_query = "\n".join(prior_user[-2:] + [user_text])
            relevant = retrieve(retrieval_query, list_memories(db))
            prompt = build_prompt(st.session_state.chat[:-1], user_text, relevant)
            answer = ask_llm(prompt)
            st.markdown(answer)
            if relevant:
                with st.expander("Memories used"):
                    for memory in relevant:
                        st.write(f"- {memory}")
            st.session_state.chat.append({"role": "assistant", "content": answer,
                                          "memories": relevant})
        except RuntimeError as exc:
            message = f"I couldn't complete that request: {exc}"
            st.error(message)
            st.session_state.chat.append({"role": "assistant", "content": message, "memories": []})
