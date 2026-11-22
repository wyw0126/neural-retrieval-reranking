
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, Optional, Set, Tuple


class MSMARCOLoader:
    """
    MS MARCO Passage Ranking v1 数据加载器。
    collection.tsv:
        passage_id \t passage_text
    queries.*.tsv:
        query_id \t query_text
    qrels.*.tsv:
        query_id iteration passage_id relevance
    """

    def __init__(self, data_dir="data/raw/msmarco"):
        self.data_dir = Path(data_dir)

        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"MS MARCO 数据目录不存在: {self.data_dir}"
            )

    def _get_path(self, filename: str) -> Path:
        path = self.data_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"数据文件不存在: {path}"
            )

        return path

    @staticmethod
    def _read_text_tsv(
        path: Path,
    ) -> Iterator[Tuple[int, str]]:
        """
        流式读取两列 TSV。
        只对第一个制表符进行切分，保留文本中的其他内容。
        """
        with path.open(
            "r",
            encoding="utf-8",
            errors="strict",
        ) as f:
            for line_no, line in enumerate(f, 1):
                parts = line.rstrip("\r\n").split("\t", 1)

                if len(parts) != 2:
                    raise ValueError(
                        f"{path}:{line_no} 格式错误，"
                        "预期为 ID<TAB>文本"
                    )

                item_id, text = parts

                try:
                    item_id = int(item_id)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_no} ID 不是整数"
                    ) from exc

                if not text.strip():
                    raise ValueError(
                        f"{path}:{line_no} 文本为空"
                    )

                yield item_id, text

    def iter_corpus(
        self,
        limit: Optional[int] = None,
    ) -> Iterator[Tuple[int, str]]:
        """
        流式读取 Passage。
        每次返回:
            (passage_id, passage_text)
        逐条构建索引
        """
        if limit is not None and limit < 0:
            raise ValueError("limit 不能小于 0")

        path = self._get_path("collection.tsv")

        for i, record in enumerate(
            self._read_text_tsv(path)
        ):
            if limit is not None and i >= limit:
                break

            yield record

    def load_queries(
        self,
        split: str = "dev",
        query_ids: Optional[Set[int]] = None,
    ) -> Dict[int, str]:
        """
        加载 Query。
        split:
            train / dev / eval / dev.small
        query_ids:
            可指定需要加载的 Query ID，
            避免读取无关查询。
        """
        if split not in {"train", "dev", "dev.small", "eval"}:
            raise ValueError(
                "split 必须是 train、dev 、dev.small或 eval"
            )

        path = self._get_path(f"queries.{split}.tsv")
        queries = {}

        for qid, text in self._read_text_tsv(path):
            if query_ids is not None and qid not in query_ids:
                continue

            if qid in queries:
                raise ValueError(f"重复 Query ID: {qid}")

            queries[qid] = text

        return queries

    def load_qrels(
        self,
        split: str = "dev",
    ) -> Dict[int, Dict[int, int]]:
        """
        加载 Query-Document 相关性标注。
        返回:
            {
                query_id: {passage_id: relevance}
            }
        """
        if split not in {"train", "dev", "dev.small"}:
            raise ValueError(
                "qrels 只支持 train 、dev.small 和 dev"
            )
        path = self._get_path(f"qrels.{split}.tsv")
        qrels = defaultdict(dict)
        duplicate_count = 0


        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                parts = line.split()

                if len(parts) != 4:
                    raise ValueError(
                        f"{path}:{line_no} "
                        "qrels 必须包含 4 列"
                    )

                qid, iteration, pid, relevance = parts

                try:
                    qid = int(qid)
                    pid = int(pid)
                    relevance = int(relevance)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{line_no} "
                        "数值字段解析失败"
                    ) from exc

                if relevance < 0:
                    raise ValueError(
                        f"{path}:{line_no} 相关性不能为负"
                    )

                
                if pid in qrels[qid]:
                    old_relevance = qrels[qid][pid]

                    # 同一个 Query-Passage 对有相同标签：
                    # 属于重复记录，忽略即可。
                    if old_relevance == relevance:
                        duplicate_count += 1
                        continue

                    # 同一配对出现不同标签：
                    # 不能直接覆盖，避免影响训练和评估。
                    raise ValueError(
                        f"Qrels 标注冲突: "
                        f"qid={qid}, pid={pid}, "
                        f"原标签={old_relevance}, "
                        f"新标签={relevance}, "
                        f"行号={line_no}"
                    )

                qrels[qid][pid] = relevance

        print(
            f"[Qrels] {split}: "
            f"忽略 {duplicate_count} 条相同标签的重复关系"
        )

        return dict(qrels)

    def iter_training_pairs(
        self,
    ) -> Iterator[Tuple[int, int, int]]:
        """
        生成已标注的训练关系。
        返回:
            (query_id, passage_id, relevance)
        """
        qrels = self.load_qrels("train")

        for qid, passages in qrels.items():
            for pid, relevance in passages.items():
                yield qid, pid, relevance

