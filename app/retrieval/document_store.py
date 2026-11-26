"""Read-only passage lookup for the online retrieval engine."""
import sqlite3
from contextlib import closing
from pathlib import Path
from numbers import Integral
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    passage_id: int
    text: str


class MissingDocumentsError(LookupError):
    def __init__(self, passage_ids: list[int]):
        self.passage_ids = tuple(passage_ids)
        super().__init__(f"Missing passage IDs: {self.passage_ids}")


class DocumentStore:
    """Each operation owns its connection; safe to share across threads."""

    def __init__(self, path: Path, *, expected_count: int | None = None,
                 expected_sha256: str | None = None, chunk_size: int = 500):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if not 1 <= chunk_size <= 900:
            raise ValueError("chunk_size must be between 1 and 900")
        self.chunk_size = chunk_size
        self._uri = self.path.as_uri() + "?mode=ro"
        with closing(self._connect()) as conn:
            metadata = dict(conn.execute("SELECT key, value FROM metadata"))
            # Validate the schema without scanning all passages.
            conn.execute("SELECT passage_id, text FROM documents LIMIT 0")
        if metadata.get("schema_version") != "1":
            raise ValueError("Unsupported document store schema version")
        self.document_count = int(metadata["document_count"])
        self.collection_sha256 = metadata["collection_sha256"].lower()
        if self.document_count <= 0:
            raise ValueError("Invalid document count")
        if len(self.collection_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.collection_sha256
        ):
            raise ValueError("Invalid collection SHA256")
        if expected_count is not None and self.document_count != expected_count:
            raise ValueError("Document count mismatch")
        if expected_sha256 is not None and self.collection_sha256 != expected_sha256.lower():
            raise ValueError("Collection SHA256 mismatch")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._uri, uri=True, timeout=5)
        try:
            conn.execute("PRAGMA query_only=ON")
        except BaseException:
            conn.close()
            raise
        return conn

    @staticmethod
    def _validate_id(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("passage_id must be an integer")
        pid = int(value)
        if not 0 <= pid <= 2**63 - 1:
            raise ValueError("passage_id outside SQLite integer range")
        return pid

    def get_many(self, passage_ids: Iterable[int], *,
                 strict: bool = True) -> list[Document]:
        """Return in input order, preserving duplicates. Strict rejects missing IDs."""
        ids = [self._validate_id(pid) for pid in passage_ids]
        if not ids:
            return []
        unique_ids = list(dict.fromkeys(ids))
        found: dict[int, Document] = {}
        with closing(self._connect()) as conn:
            for start in range(0, len(unique_ids), self.chunk_size):
                chunk = unique_ids[start:start + self.chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT passage_id, text FROM documents WHERE passage_id IN ({placeholders})",
                    chunk,
                )
                for pid, body in rows:
                    found[pid] = Document(pid, body)
        missing = [pid for pid in unique_ids if pid not in found]
        if strict and missing:
            raise MissingDocumentsError(missing)
        return [found[pid] for pid in ids if pid in found]

    def get(self, passage_id: int) -> Document | None:
        docs = self.get_many([passage_id], strict=False)
        return docs[0] if docs else None
