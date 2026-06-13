"""SQLite registry for model runs and evaluation metadata."""
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional


class ExperimentRegistry:
    """Small SQLite registry; heavy artifacts stay on disk."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT,
                policy TEXT NOT NULL,
                model_path TEXT NOT NULL,
                goal_id INTEGER NOT NULL,
                capture TEXT NOT NULL,
                controller TEXT NOT NULL,
                started_at REAL NOT NULL,
                duration_requested REAL NOT NULL,
                conf_threshold REAL NOT NULL,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                finished_at REAL NOT NULL,
                total_steps INTEGER NOT NULL,
                executed_steps INTEGER NOT NULL,
                low_confidence_steps INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES runs(id)
            );
            """
        )
        self.conn.commit()

    def create_run(self, config, metadata: Optional[dict] = None) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO runs (
                run_name, policy, model_path, goal_id, capture, controller,
                started_at, duration_requested, conf_threshold, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config.run_name,
                config.policy,
                str(config.model_path),
                config.goal_id,
                config.capture,
                config.controller,
                time.time(),
                config.duration,
                config.conf_threshold,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, steps, metrics: Optional[dict] = None) -> None:
        total_steps = len(steps)
        executed_steps = sum(1 for step in steps if step.executed)
        low_confidence_steps = sum(1 for step in steps if step.reason == "low_confidence")
        payload = {
            "execution_rate": executed_steps / max(total_steps, 1),
            **(metrics or {}),
        }
        self.conn.execute(
            """
            INSERT INTO eval_results (
                run_id, finished_at, total_steps, executed_steps,
                low_confidence_steps, metrics_json
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                time.time(),
                total_steps,
                executed_steps,
                low_confidence_steps,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

