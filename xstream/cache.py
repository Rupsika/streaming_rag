"""Tiny sqlite key-value cache for VLM / LLM outputs (plan.md §8: "the single biggest time saver").

Keys are SHA-1 of the JSON-encoded key parts, e.g. (model, video, frame_id, prompt), so
ablations that change only retrieval or the LLM never re-run the VLM.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


class KVCache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, k: str):
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, k: str, v) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (k, json.dumps(v)))
            self._db.commit()
