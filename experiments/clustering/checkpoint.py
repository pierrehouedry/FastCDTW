from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


class FileLock:
    def __init__(self, path: Path, timeout: float = 120.0):
        self.lock = path.with_suffix(path.suffix + ".lock")
        self.timeout, self.fd = timeout, None

    def __enter__(self):
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - t0 > self.timeout:
                    # assume the holder died; steal the lock and restart the clock
                    try:
                        self.lock.unlink()
                    except FileNotFoundError:
                        pass
                    t0 = time.time()
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            self.lock.unlink()
        except FileNotFoundError:
            pass


def read_locked(path: Path, default):
    with FileLock(path):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return default


def update_json(path: Path, mutate):
    """Read-modify-write under the lock, via an atomic replace."""
    with FileLock(path):
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                data = {}
        mutate(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent))
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
