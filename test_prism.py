"""Comprehensive test suite for PRISM Steps 1 through 4."""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np


class TestPrismStep1(unittest.TestCase):
    """Step 1: Make memories reliable and editable."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_prism.db")
        self.legacy_path = os.path.join(self.temp_dir.name, "test_legacy.json")

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except (PermissionError, OSError):
            pass

    def test_schema_and_migration(self):
        import prism_demo

        # Create legacy file with 2 items and embeddings
        fake_emb = [0.1] * 768
        with open(self.legacy_path, "w", encoding="utf-8") as f:
            json.dump({
                "texts": ["I build electronics", "I love robotics"],
                "embeddings": [fake_emb, fake_emb]
            }, f)

        # Initialize store
        conn = prism_demo.initialize_memory_store(db_file=self.db_path, legacy_file=self.legacy_path)
        memories = prism_demo.list_memories(conn)
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["text"], "I build electronics")
        self.assertIsNotNone(memories[0]["id"])
        self.assertIsNotNone(memories[0]["created_at"])
        self.assertIsNotNone(memories[0]["embedding"])

        # Second init must not duplicate
        conn.close()
        conn2 = prism_demo.initialize_memory_store(db_file=self.db_path, legacy_file=self.legacy_path)
        memories2 = prism_demo.list_memories(conn2)
        self.assertEqual(len(memories2), 2)
        conn2.close()

    @patch("prism_demo.get_embedding", return_value=[0.2] * 768)
    def test_crud_operations(self, mock_embed):
        import prism_demo

        conn = prism_demo.connect_db(self.db_path)
        
        # Add memory
        new_id = prism_demo.add_memory(conn, "I play badminton")
        memories = prism_demo.list_memories(conn)
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["text"], "I play badminton")
        self.assertEqual(memories[0]["id"], new_id)

        # Edit memory
        mock_embed.return_value = [0.5] * 768
        prism_demo.update_memory(conn, new_id, "I play tennis")
        updated = prism_demo.list_memories(conn)[0]
        self.assertEqual(updated["text"], "I play tennis")
        self.assertEqual(json.loads(updated["embedding"]), [0.5] * 768)

        # Delete memory
        prism_demo.delete_memory(conn, new_id)
        self.assertEqual(len(prism_demo.list_memories(conn)), 0)

        # Re-add and Clear all
        prism_demo.add_memory(conn, "Mem 1")
        prism_demo.add_memory(conn, "Mem 2")
        self.assertEqual(len(prism_demo.list_memories(conn)), 2)
        prism_demo.clear_all_memories(conn)
        self.assertEqual(len(prism_demo.list_memories(conn)), 0)
        conn.close()


class TestPrismStep2(unittest.TestCase):
    """Step 2: Fix embedding and retrieval failures."""

    def test_no_random_fallback(self):
        import prism_demo

        with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
            with self.assertRaises(prism_demo.OllamaServiceError):
                prism_demo.get_embedding("test prompt")

    def test_empty_memories_returns_empty(self):
        import prism_demo

        result = prism_demo.retrieve("What are my hobbies?", [])
        self.assertEqual(result, [])

    def test_cosine_similarity_dimension_check(self):
        import prism_demo

        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.0, 2.0]
        with self.assertRaises(ValueError):
            prism_demo.cosine_similarity(vec_a, vec_b)

    @patch("prism_demo.get_embedding")
    def test_similarity_threshold_filtering(self, mock_embed):
        import prism_demo

        # Query vector: [1, 0]
        mock_embed.return_value = [1.0, 0.0]

        # Memory 1: very similar [0.99, 0.01] -> cos ~ 0.99
        # Memory 2: unrelated [0.0, 1.0] -> cos ~ 0.0
        memories = [
            {"id": 1, "text": "I love robotics", "embedding": json.dumps([0.99, 0.01])},
            {"id": 2, "text": "I eat pizza", "embedding": json.dumps([0.0, 1.0])}
        ]

        # With threshold 0.45, only Memory 1 should be returned
        results = prism_demo.retrieve("robotics project", memories, threshold=0.45)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "I love robotics")

        # When nothing meets threshold
        mock_embed.return_value = [0.0, -1.0]
        results_none = prism_demo.retrieve("astronomy", memories, threshold=0.45)
        self.assertEqual(results_none, [])

    def test_setup_hint_on_missing_model(self):
        import prism_demo
        from urllib.error import HTTPError
        import io

        mock_http_err = HTTPError(
            url="http://localhost:11434/api/embeddings",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b'{"error":"model \'nomic-embed-text\' not found"}')
        )

        with patch("urllib.request.urlopen", side_effect=mock_http_err):
            with self.assertRaises(prism_demo.OllamaModelNotFoundError) as ctx:
                prism_demo.get_embedding("hello", model="nomic-embed-text")
            self.assertIn("ollama pull nomic-embed-text", str(ctx.exception))


class TestPrismStep3(unittest.TestCase):
    """Step 3: Make the chat behave like a chat."""

    def test_conversation_turns_in_prompt(self):
        import prism_demo

        history = [
            {"role": "user", "content": "I like robots"},
            {"role": "assistant", "content": "Robots are fascinating!"},
            {"role": "user", "content": "What was the first thing I mentioned?"}
        ]
        prompt = prism_demo.build_prompt(history[:-1], history[-1]["content"], memories=["I like robots"], max_turns=4)
        self.assertIn("User: I like robots", prompt)
        self.assertIn("PRISM: Robots are fascinating!", prompt)
        self.assertIn("User: What was the first thing I mentioned?", prompt)

    def test_prompt_limits_history(self):
        import prism_demo

        long_history = []
        for i in range(20):
            long_history.append({"role": "user", "content": f"Turn {i}"})
            long_history.append({"role": "assistant", "content": f"Reply {i}"})

        prompt = prism_demo.build_prompt(long_history, "Latest query", memories=[], max_turns=3)
        # Should only include last 6 messages (3 turns * 2)
        self.assertNotIn("Turn 0", prompt)
        self.assertIn("Turn 19", prompt)


class TestPrismStep4(unittest.TestCase):
    """Step 4: Improve answer quality and trust."""

    def test_prompt_directives_and_evidence(self):
        import prism_demo

        # With memories
        prompt_with_mem = prism_demo.build_prompt(
            history=[],
            user_text="What do I work on?",
            memories=["I work on autonomous drones"],
            personality={"tone": "friendly", "style": "brief"}
        )
        self.assertIn("I work on autonomous drones", prompt_with_mem)
        self.assertIn("Grounding in Memories", prompt_with_mem)
        self.assertIn("No Hallucinated Facts", prompt_with_mem)
        self.assertIn("General Knowledge", prompt_with_mem)
        self.assertIn("friendly", prompt_with_mem)

        # Without memories
        prompt_no_mem = prism_demo.build_prompt(
            history=[],
            user_text="What is my favorite color?",
            memories=[]
        )
        self.assertIn("None found", prompt_no_mem)
        self.assertIn("Do NOT invent", prompt_no_mem)


class TestPrismChatPersistence(unittest.TestCase):
    """Test SQLite chat persistence and isolation."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_chat.db")

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except (PermissionError, OSError):
            pass

    def test_save_and_list_chat_messages(self):
        import prism_demo

        conn = prism_demo.connect_db(self.db_path)

        # Initially empty
        self.assertEqual(len(prism_demo.list_chat_messages(conn)), 0)

        # Save user message
        uid = prism_demo.save_chat_message(conn, "user", "What is my hobby?")
        self.assertIsNotNone(uid)

        # Save assistant message with memories
        aid = prism_demo.save_chat_message(conn, "assistant", "You love robotics.", ["I love robotics"])
        self.assertIsNotNone(aid)

        messages = prism_demo.list_chat_messages(conn)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "What is my hobby?")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["memories"], ["I love robotics"])

        # Also add a memory to verify isolation
        prism_demo.add_memory(conn, "Keep this memory")
        self.assertEqual(len(prism_demo.list_memories(conn)), 1)

        # Clear chat history: chat messages wiped, memories intact!
        prism_demo.clear_chat_history(conn)
        self.assertEqual(len(prism_demo.list_chat_messages(conn)), 0)
        self.assertEqual(len(prism_demo.list_memories(conn)), 1)
        conn.close()


class TestPrismLocalTwinEngine(unittest.TestCase):
    """Test built-in twin response fallback engine."""

    def test_greeting(self):
        import prism_demo
        resp = prism_demo.generate_local_twin_response("hello there", [], [])
        self.assertIn("digital twin", resp.lower())

    def test_with_memories(self):
        import prism_demo
        resp = prism_demo.generate_local_twin_response("What do I code in?", ["I usually write code in Python and Java."], [])
        self.assertIn("Python and Java", resp)

    def test_personal_question_without_memory(self):
        import prism_demo
        resp = prism_demo.generate_local_twin_response("What is my dog's name?", [], [])
        self.assertIn("don't have a record", resp.lower())


if __name__ == "__main__":
    unittest.main()
