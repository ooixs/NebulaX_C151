"""Small durable store for technician analysis records."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path


class HistoryStore:
    """Persist completed runs without changing their official prediction CSV."""

    def __init__(self, path: str | Path, limit: int = 500):
        self.path = Path(path)
        self.limit = limit
        self.lock = threading.Lock()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created TEXT NOT NULL,
                payload TEXT NOT NULL,
                csv BLOB
            )
            """
        )
        return connection

    @staticmethod
    def _unpack(row):
        if row is None:
            return None
        run = json.loads(row["payload"])
        if row["csv"] is not None:
            run["csv"] = bytes(row["csv"])
        return run

    def save(self, run):
        payload = {key: value for key, value in run.items() if key != "csv"}
        with self.lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs (id, created, payload, csv) VALUES (?, ?, ?, ?)",
                (run["id"], run["created"], json.dumps(payload, allow_nan=False), run.get("csv")),
            )
            connection.execute(
                """
                DELETE FROM runs
                WHERE id IN (
                    SELECT id FROM runs ORDER BY created DESC, rowid DESC LIMIT -1 OFFSET ?
                )
                """,
                (self.limit,),
            )

    def get(self, run_id):
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT payload, csv FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._unpack(row)

    def list(self, limit=None):
        count = self.limit if limit is None else min(int(limit), self.limit)
        with self.lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload, csv FROM runs ORDER BY created DESC, rowid DESC LIMIT ?",
                (count,),
            ).fetchall()
        return [self._unpack(row) for row in rows]

    def update_review(self, run_id, review):
        with self.lock, self._connect() as connection:
            row = connection.execute("SELECT payload, csv FROM runs WHERE id = ?", (run_id,)).fetchone()
            run = self._unpack(row)
            if run is None:
                return None
            run["review"] = review
            payload = {key: value for key, value in run.items() if key != "csv"}
            connection.execute(
                "UPDATE runs SET payload = ? WHERE id = ?",
                (json.dumps(payload, allow_nan=False), run_id),
            )
        return run
