"""Process-owned writer locks and a wall-clock training heartbeat."""

import json
import os
from pathlib import Path
import threading
from time import monotonic

from .data import DataError


def io_path(path):
    """Windows extended absolute path for deeply nested immutable artifacts."""
    path = Path(path).absolute()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        text = str(path)
        return Path("\\\\?\\UNC\\"+text[2:] if text.startswith("\\\\") else "\\\\?\\"+text)
    return path


class WriterLock:
    """Kernel lock, automatically released on process exit (including os._exit).

    Never unlink the lock file: unlinking would permit two locked inodes.
    The PID is diagnostic only, never used to steal a live lock.
    """
    def __init__(self, path, code="RUN_LOCKED"):
        self.path, self.handle = io_path(path), None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b", buffering=0)
        if self.path.stat().st_size == 0:
            handle.write(b"\0")
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            print(json.dumps({"event":"writer_conflict", "code":code,
                              "path":self.path.as_posix(), "pid":os.getpid()}), flush=True)
            raise DataError(code, "Another writer owns this run/cache", path=self.path.as_posix()) from exc
        self.handle = handle
        handle.seek(1)
        handle.write(json.dumps({"pid":os.getpid(), "path":self.path.as_posix()}).encode())
        handle.truncate()

    def close(self):
        if self.handle is not None:
            self.handle.close()  # OS releases the lock, even without an explicit unlock.
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class Progress:
    def __init__(self, output, experiment, variant, interval=30):
        if not 0 < interval <= 60:
            raise DataError("INVALID_VALUE", "Progress interval must be in (0,60] seconds")
        self.path = Path(output)/"progress.jsonl"
        self.state = {"experiment":experiment,"variant":variant,"epoch":0,"batch":0,
                      "batches":0,"phase":"initializing","cache_hits":0,"cache_misses":0,
                      "last_checkpoint":None}
        self.started = monotonic()
        self.counters = None
        self.interval = interval
        self.stop = threading.Event()
        self.mutex = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def update(self, **values):
        with self.mutex:
            self.state.update(values)

    def emit(self, event="progress", **extra):
        with self.mutex:
            if self.counters is not None:
                self.state.update(self.counters())
            record = {**self.state,"event":event,"elapsed_seconds":monotonic()-self.started,
                      "pid":os.getpid(), **extra}
            line = json.dumps(record,ensure_ascii=False)
            with self.path.open("a",encoding="utf-8") as out:
                out.write(line+"\n")
            print(line,flush=True)

    def _run(self):
        while not self.stop.wait(self.interval):
            self.emit()

    def __enter__(self):
        self.emit("started")
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.emit("process_finished")
