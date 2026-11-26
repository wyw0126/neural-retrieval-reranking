"""Build an immutable SQLite passage store from the MS MARCO collection."""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_document_store(collection: Path, output: Path, *,
                         expected_count: int = 8841823,
                         batch_size: int = 10000,
                         expected_sha256: str | None = None) -> dict:
    """Build in a temporary file; publish only after verification."""
    collection = Path(collection).resolve()
    output = Path(output).resolve()
    if expected_count <= 0 or batch_size <= 0:
        raise ValueError("expected_count and batch_size must be positive")
    if not collection.is_file():
        raise FileNotFoundError(collection)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".documents-", suffix=".sqlite3",
                                dir=output.parent)
    os.close(fd)
    temp = Path(name)
    start = time.monotonic()
    digest = hashlib.sha256()
    count = 0
    batch = []
    samples = []
    conn = None
    try:
        conn = sqlite3.connect(temp)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("CREATE TABLE documents (passage_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        with collection.open("rb") as source:
            for line_no, raw in enumerate(source, 1):
                digest.update(raw)
                parts = raw.decode("utf-8").rstrip("\r\n").split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid TSV at line {line_no}")
                try:
                    pid = int(parts[0])
                except ValueError as exc:
                    raise ValueError(f"Invalid passage ID at line {line_no}") from exc
                body = parts[1]
                if not 0 <= pid <= 2**63 - 1 or not body.strip():
                    raise ValueError(f"Invalid passage at line {line_no}")
                if line_no == 1 or line_no % 1000000 == 0 or line_no == expected_count:
                    samples.append((pid, body))
                batch.append((pid, body))
                if len(batch) >= batch_size:
                    with conn:
                        conn.executemany("INSERT INTO documents VALUES (?, ?)", batch)
                    count += len(batch)
                    batch.clear()
                    if count % (batch_size * 100) == 0:
                        print(f"Inserted {count:,} passages", flush=True)
        if batch:
            with conn:
                conn.executemany("INSERT INTO documents VALUES (?, ?)", batch)
            count += len(batch)
        checksum = digest.hexdigest()
        if count != expected_count:
            raise ValueError(f"Expected {expected_count} passages, got {count}")
        if expected_sha256 and checksum != expected_sha256.lower():
            raise ValueError(f"Collection SHA256 mismatch: {checksum}")
        actual = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if actual != count:
            raise RuntimeError("Database row count mismatch")
        for pid, body in samples:
            result = conn.execute("SELECT text FROM documents WHERE passage_id=?", (pid,)).fetchone()
            if result != (body,):
                raise RuntimeError(f"Sample verification failed: {pid}")
        metadata = {"schema_version": "1", "collection_sha256": checksum,
                    "document_count": str(count)}
        with conn:
            conn.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        if conn.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise RuntimeError("SQLite quick_check failed")
        conn.close()
        conn = None
        # Same-filesystem hard link publishes atomically without overwriting
        # an existing database, even if another builder finished first.
        os.link(temp, output)
        return {**metadata, "output": str(output), "document_count": count,
                "verified_samples": len(samples), "bytes": output.stat().st_size,
                "elapsed_seconds": round(time.monotonic() - start, 3)}
    finally:
        if conn is not None:
            conn.close()
        temp.unlink(missing_ok=True)
        Path(str(temp) + "-journal").unlink(missing_ok=True)


def main():
    from app.config import load_settings
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=settings.collection_path)
    parser.add_argument("--output", type=Path, default=settings.document_store)
    parser.add_argument("--expected-count", type=int, default=8841823)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    result = build_document_store(
        args.collection, args.output, expected_count=args.expected_count,
        batch_size=args.batch_size, expected_sha256=args.expected_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
