"""Bounded-memory access to offline landmark geometry."""
from collections.abc import Mapping
import json
import os
from pathlib import Path
import sqlite3


class LandmarkStore(Mapping):
    def __init__(self, path):
        self.path = Path(path)
        self.connection, self.pid = None, None
        with self.path.open("rb") as stream:
            self.sqlite = stream.read(16) == b"SQLite format 3\x00"
        if self.sqlite:
            with sqlite3.connect(self.path) as db:
                self.metadata = json.loads(db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
                self.catalog = {row[0]: dict(sample_id=row[0], status=row[1], image_sha256=row[2], label=row[3])
                                for row in db.execute("SELECT sample_id,status,image_sha256,label FROM faces")}
            self.records = None
        else:
            if self.path.stat().st_size > 128*1024*1024:
                raise ValueError("Large landmark caches must use SQLite for bounded memory")
            value = json.loads(self.path.read_text())
            rows = value.pop("records")
            self.metadata = value
            self.records = {r["sample_id"]: r for r in rows}
            if len(rows) != len(self.records):
                raise ValueError("Duplicate landmark identity")
            self.catalog = self.records

    def __len__(self):
        return len(self.catalog)

    def __iter__(self):
        return iter(self.catalog)

    def __getitem__(self, key):
        if key not in self.catalog:
            raise KeyError(key)
        if not self.sqlite:
            return self.records[key]
        if self.pid != os.getpid():
            if self.connection is not None:
                self.connection.close()
            self.connection = sqlite3.connect(self.path)
            self.pid = os.getpid()
        return json.loads(self.connection.execute("SELECT record FROM faces WHERE sample_id=?", (key,)).fetchone()[0])

    def __getstate__(self):
        return {**self.__dict__, "connection": None, "pid": None}
