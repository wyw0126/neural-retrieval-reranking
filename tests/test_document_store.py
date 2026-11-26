import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.retrieval.document_store import DocumentStore, MissingDocumentsError
from scripts.serving.build_document_store import build_document_store


@pytest.fixture
def store(tmp_path):
    source = tmp_path / "collection.tsv"
    source.write_text("8\thello\n2\tworld\n5\tthird\n", encoding="utf-8")
    path = tmp_path / "documents.sqlite3"
    build_document_store(source, path, expected_count=3)
    return DocumentStore(path, expected_count=3, chunk_size=2)


def test_get(store):
    assert store.get(8).text == "hello"
    assert store.get(99) is None


def test_batch_order_and_duplicates(store):
    assert [d.passage_id for d in store.get_many([8, 5, 2, 8])] == [8, 5, 2, 8]
    assert store.get_many([]) == []


def test_missing(store):
    with pytest.raises(MissingDocumentsError) as error:
        store.get_many([8, 99, 100, 99])
    assert error.value.passage_ids == (99, 100)
    assert [d.passage_id for d in store.get_many([99, 2], strict=False)] == [2]


@pytest.mark.parametrize("pid", [-1, True, "8", 8.0, 2**63])
def test_invalid_id(store, pid):
    with pytest.raises((ValueError, TypeError)):
        store.get(pid)


def test_metadata(store):
    with pytest.raises(ValueError, match="count"):
        DocumentStore(store.path, expected_count=4)
    with pytest.raises(ValueError, match="SHA256"):
        DocumentStore(store.path, expected_sha256="0"*64)


def test_read_only(store):
    conn = store._connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM documents")
    finally:
        conn.close()


def test_concurrent_reads(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(store.get, [8, 2, 5, 8]*10))
    assert [d.passage_id for d in results] == [8, 2, 5, 8]*10
