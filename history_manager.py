# history_manager.py

import sqlite3
import json
import os
from datetime import datetime

# Import configuration from our config.py
import config

class HistoryManager:
    def __init__(self, db_path=config.CHAT_DB_PATH):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Initializes the SQLite database and creates the chat_history table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print(f"Chat history database initialized at: {self.db_path}")

    def save_message(self, session_id: str, role: str, content: str):
        """Saves a single chat message to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO chat_history (session_id, timestamp, role, content) VALUES (?, ?, ?, ?)",
            (session_id, timestamp, role, content)
        )
        conn.commit()
        conn.close()

    def load_history(self, session_id: str) -> list:
        """Loads all chat messages for a given session ID from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        history = []
        for row in cursor.fetchall():
            history.append({"role": row[0], "content": row[1]})
        conn.close()
        return history

    def clear_history(self, session_id: str):
        """Clears all chat messages for a given session ID from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        print(f"Chat history for session '{session_id}' cleared.")

    def get_all_session_ids(self) -> list:
        """Retrieves all unique session IDs from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM chat_history")
        session_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return session_ids