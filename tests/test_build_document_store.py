import hashlib
import sqlite3

import pytest

from scripts.serving.build_document_store import build_document_store


def test_build(tmp_path):
    source = tmp_path / "collection.tsv"
    raw = "8\thello\n2\t你好\tworld\n".encode()
    source.write_bytes(raw)
    output = tmp_path / "documents.sqlite3"
    result = build_document_store(source, output, expected_count=2,
                                  batch_size=1, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert result["document_count"] == 2
    with sqlite3.connect(output) as conn:
        assert conn.execute("SELECT * FROM documents ORDER BY passage_id").fetchall() == [(2, "你好\tworld"), (8, "hello")]
        assert dict(conn.execute("SELECT * FROM metadata"))["collection_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("raw", [b"1\ta\n1\tb\n", b"bad\n", b"1\t\n"])
def test_invalid_input(tmp_path, raw):
    source = tmp_path / "collection.tsv"
    source.write_bytes(raw)
    output = tmp_path / "documents.sqlite3"
    with pytest.raises((ValueError, sqlite3.IntegrityError)):
        build_document_store(source, output, expected_count=2, batch_size=1)
    assert not output.exists()
    assert not list(tmp_path.glob(".documents-*"))


def test_count_and_hash(tmp_path):
    source = tmp_path / "collection.tsv"
    source.write_bytes(b"1\ta\n")
    output = tmp_path / "documents.sqlite3"
    with pytest.raises(ValueError, match="Expected"):
        build_document_store(source, output, expected_count=2)
    with pytest.raises(ValueError, match="SHA256"):
        build_document_store(source, output, expected_count=1, expected_sha256="0"*64)
    assert not output.exists()


def test_no_overwrite(tmp_path):
    source = tmp_path / "collection.tsv"
    source.write_bytes(b"1\ta\n")
    output = tmp_path / "documents.sqlite3"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        build_document_store(source, output, expected_count=1)
    assert output.read_bytes() == b"existing"
