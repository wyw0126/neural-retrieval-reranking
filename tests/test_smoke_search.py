from types import SimpleNamespace
import json
import pytest

from scripts.serving import smoke_search as cli


def test_parse():
    args = cli.parse_args(["--query", "a", "--query", "b", "--no-rerank", "--top-k", "3"])
    assert args.query == ["a", "b"]
    assert args.no_rerank and args.top_k == 3


@pytest.mark.parametrize("argv", [[], ["--query", " "], ["--query", "a", "--interactive"],
                                  ["--query", "a", "--top-k", "0"],
                                  ["--query", "a", "--repeat", "0"]])
def test_invalid_args(argv):
    with pytest.raises(SystemExit):
        cli.parse_args(argv)


def test_summary():
    result = cli.summarize([{"total": 1}, {"total": 2}, {"total": 10}])
    assert result["requests"] == 3
    assert result["timings_ms"]["total"]["p95"] == 10
    assert result["timings_ms"]["total"]["p50"] == 2
    assert cli.summarize([])["requests"] == 0


def response():
    hit = SimpleNamespace(passage_id=1, text="example", rank=1, retrieval_rank=1,
                          retrieval_score=.8, rerank_score=4.)
    return SimpleNamespace(candidate_count=1, results=[hit], timings_ms={"total": 2.},
                           to_dict=lambda: {"results": [vars(hit)], "timings_ms":{"total":2.}})


def test_run_query(capsys):
    calls = []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return response()
    args = cli.parse_args(["--query", "a", "--warmup", "2", "--repeat", "3", "--json"])
    rows = cli.run_query(SimpleNamespace(search=search), "a", 3, 1, True, args)
    assert len(calls) == 5 and len(rows) == 3
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(payloads) == 3
    assert payloads[0]["results"][0]["text"] == "example"


def test_bad_response():
    result = response()
    result.results[0].rank = 2
    with pytest.raises(RuntimeError):
        cli.validate_response(result, 3, 1, True)
