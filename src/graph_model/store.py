from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import NodeResult, RunState



try:  # pragma: no cover - Windows fallback is exercised only off POSIX
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_RUN_LOCKS: dict[str, threading.Lock] = {}
_RUN_LOCKS_GUARD = threading.RLock()


class RunAlreadyActive(RuntimeError):
    """Raised when another process or thread owns the same run execution lease."""


class SQLiteRunStore:
    """Local event/checkpoint store.

    It provides deterministic replay for node results. External side effects must still accept
    the supplied idempotency key. The optional Restate adapter provides stronger production
    durable-execution semantics.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def run_lock(self, run_id: str) -> Iterator[None]:
        """Hold a non-blocking per-run lease for the entire local execution call.

        The OS file lock is released automatically after crashes on POSIX systems. The in-process
        lock also prevents duplicate execution from concurrent asyncio entry points or threads.
        """

        digest = hashlib.sha256(run_id.encode("utf-8", errors="surrogatepass")).hexdigest()
        lock_dir = self.path.parent / f".{self.path.name}.run-locks"
        lock_path = lock_dir / f"{digest}.lock"
        key = str(lock_path)
        with _RUN_LOCKS_GUARD:
            thread_lock = _RUN_LOCKS.setdefault(key, threading.Lock())
        if not thread_lock.acquire(blocking=False):
            raise RunAlreadyActive(f"run {run_id!r} is already active")
        handle = None
        try:
            lock_dir.mkdir(parents=True, exist_ok=True)
            handle = lock_path.open("a+b")
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RunAlreadyActive(f"run {run_id!r} is already active") from exc
            yield
        finally:
            if handle is not None:
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                handle.close()
            thread_lock.release()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    node_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(run_id, seq)
                );
                CREATE TABLE IF NOT EXISTS step_results (
                    run_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(run_id, node_id, input_hash)
                );
                CREATE INDEX IF NOT EXISTS events_run_id_idx ON events(run_id, seq);
                """
            )

    def create_run(self, state: RunState) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO runs(run_id, state_json, updated_at) VALUES (?, ?, ?)",
                (state.run_id, state.model_dump_json(), time.time()),
            )
            connection.execute(
                "INSERT INTO events(run_id, seq, event_type, node_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    state.run_id,
                    0,
                    "run_started",
                    state.current_node,
                    json.dumps({"graph": state.graph_name, "version": state.graph_version}),
                    time.time(),
                ),
            )

    def load_run(self, run_id: str) -> RunState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return RunState.model_validate_json(row["state_json"]) if row else None

    def get_step_result(self, run_id: str, node_id: str, input_hash: str) -> NodeResult | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM step_results WHERE run_id = ? AND node_id = ? AND input_hash = ?",
                (run_id, node_id, input_hash),
            ).fetchone()
        return NodeResult.model_validate_json(row["result_json"]) if row else None

    def commit_step(
        self,
        *,
        state: RunState,
        node_id: str,
        input_hash: str,
        result: NodeResult,
        cached: bool,
        event_payload: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "cached": cached,
            "result": result.model_dump(),
            **(event_payload or {}),
        }
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if not cached:
                connection.execute(
                    "INSERT OR REPLACE INTO step_results(run_id, node_id, input_hash, result_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id := state.run_id, node_id, input_hash, result.model_dump_json(), time.time()),
                )
            else:
                run_id = state.run_id
            connection.execute(
                "UPDATE runs SET state_json = ?, updated_at = ? WHERE run_id = ?",
                (state.model_dump_json(), time.time(), run_id),
            )
            connection.execute(
                "INSERT INTO events(run_id, seq, event_type, node_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    state.step_count,
                    "node_completed",
                    node_id,
                    json.dumps(payload, sort_keys=True, default=str),
                    time.time(),
                ),
            )
            connection.commit()

    def save_terminal_event(self, state: RunState, event_type: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE runs SET state_json = ?, updated_at = ? WHERE run_id = ?",
                (state.model_dump_json(), time.time(), state.run_id),
            )
            next_seq = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS seq FROM events WHERE run_id = ?",
                (state.run_id,),
            ).fetchone()["seq"]
            connection.execute(
                "INSERT INTO events(run_id, seq, event_type, node_id, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    state.run_id,
                    next_seq,
                    event_type,
                    state.current_node,
                    json.dumps({"status": state.status, "error": state.error}, sort_keys=True),
                    time.time(),
                ),
            )
            connection.commit()

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT seq, event_type, node_id, payload_json, created_at "
                "FROM events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "event_type": row["event_type"],
                "node_id": row["node_id"],
                "payload": json.loads(row["payload_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def list_runs(self) -> list[RunState]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state_json FROM runs ORDER BY updated_at DESC"
            ).fetchall()
        return [RunState.model_validate_json(row["state_json"]) for row in rows]
