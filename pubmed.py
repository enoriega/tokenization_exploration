from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal

import pubmed_parser
from datasets import Dataset


@dataclass
class PubMedRecord:
    pmid: str
    type: Literal["title", "abstract"]
    text: str


def parse_pubmed_xml(file_path: str | Path) -> Generator[PubMedRecord, None, None]:
    for article in pubmed_parser.parse_medline_xml(str(file_path)):
        pmid = article["pmid"]
        if article["title"]:
            yield PubMedRecord(pmid=pmid, type="title", text=article["title"])
        if article["abstract"]:
            yield PubMedRecord(pmid=pmid, type="abstract", text=article["abstract"])


def build_dataset(*file_paths: str | Path) -> Dataset:
    def _records():
        for path in file_paths:
            yield from parse_pubmed_xml(path)

    return Dataset.from_generator(lambda: ({"pmid": r.pmid, "type": r.type, "text": r.text} for r in _records()))
