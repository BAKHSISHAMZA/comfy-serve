"""The job store: one SQLite file, one table.

Why SQLite: a job must survive the server dying. A Python list or an in-memory queue forgets everything on a crash;
a file on disk does not. SQLite needs no extra service on a Windows GPU box, and with WAL mode the API and the
worker can read and write at the same time.

A job moves through four states:
    queued  -> running -> done
                       -> failed
`running` is the dangerous one: if the process dies there, the job is stuck. On start, the worker looks at every
`running` job and decides (see worker.recover) whether to keep waiting for ComfyUI, collect a finished result,
or put the job back in the queue. Nothing is lost and nothing is rendered twice.
"""
import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    workflow      TEXT NOT NULL,
    inputs        TEXT NOT NULL,          -- JSON: the named inputs the client sent
    files         TEXT NOT NULL,          -- JSON: {input name: path of the uploaded file on disk}
    status        TEXT NOT NULL,          -- queued | running | done | failed
    prompt_id     TEXT,                   -- ComfyUI's id for this run, set once it is queued there
    outputs       TEXT,                   -- JSON: list of output file paths
    error         TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    created       REAL NOT NULL,
    started       REAL,
    finished      REAL,
    seconds       REAL,                   -- render time reported by ComfyUI (execution start -> success)
    peak_vram_mb  REAL                    -- highest GPU memory in use seen while this job ran
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created);
"""


class Store:
    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.Lock()          # one writer at a time inside this process
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def _db(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)   # autocommit; we use explicit BEGIN
        db.row_factory = sqlite3.Row
        return db

    # ------------------------------------------------------------------ writes
    def create(self, workflow, inputs, files=None, job_id=None):
        job_id = job_id or uuid.uuid4().hex[:12]
        with self._lock, self._db() as db:
            db.execute("INSERT INTO jobs (id, workflow, inputs, files, status, created) VALUES (?,?,?,?,?,?)",
                       (job_id, workflow, json.dumps(inputs), json.dumps(files or {}), "queued", time.time()))
        return job_id

    def claim_next(self):
        """Take the oldest queued job and mark it running, atomically, so two workers can never take the same job."""
        with self._lock, self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                db.execute("COMMIT")
                return None
            db.execute("UPDATE jobs SET status='running', started=?, attempts=attempts+1, prompt_id=NULL "
                       "WHERE id=?", (time.time(), row["id"]))
            db.execute("COMMIT")
        return self.get(row["id"])

    def set_prompt(self, job_id, prompt_id):
        self._update(job_id, prompt_id=prompt_id)

    def finish(self, job_id, outputs, seconds, peak_vram_mb):
        self._update(job_id, status="done", outputs=json.dumps(outputs), seconds=seconds,
                     peak_vram_mb=peak_vram_mb, finished=time.time(), error=None)

    def fail(self, job_id, error):
        self._update(job_id, status="failed", error=str(error)[:2000], finished=time.time())

    def requeue(self, job_id, reason, refund_attempt=False):
        with self._lock, self._db() as db:
            db.execute("UPDATE jobs SET status='queued', prompt_id=NULL, error=?, "
                       "attempts=CASE WHEN ? THEN MAX(attempts-1, 0) ELSE attempts END WHERE id=?",
                       (f"requeued: {reason}"[:2000], int(refund_attempt), job_id))

    def _update(self, job_id, **cols):
        keys = ", ".join(f"{k}=?" for k in cols)
        with self._lock, self._db() as db:
            db.execute(f"UPDATE jobs SET {keys} WHERE id=?", (*cols.values(), job_id))

    # ------------------------------------------------------------------ reads
    def get(self, job_id):
        with self._db() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _job(row) if row else None

    def by_status(self, status):
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status=? ORDER BY created", (status,)).fetchall()
        return [_job(r) for r in rows]

    def counts(self):
        with self._db() as db:
            rows = db.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}


def _job(row):
    j = dict(row)
    for k in ("inputs", "files", "outputs"):
        j[k] = json.loads(j[k]) if j.get(k) else ({} if k != "outputs" else [])
    return j
