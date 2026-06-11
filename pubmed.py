import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal

import pubmed_parser
from datasets import Dataset


# ---------------------------------------------------------------------------
# PubMed / MEDLINE abstracts
# ---------------------------------------------------------------------------

@dataclass
class PubMedRecord:
    pmid: str
    type: Literal["title", "abstract"]
    text: str


def parse_pubmed_abstracts(file_path: str | Path) -> Generator[PubMedRecord, None, None]:
    """Yield title and abstract records from a PubMed/MEDLINE XML file."""
    for article in pubmed_parser.parse_medline_xml(str(file_path)):
        pmid = article["pmid"]
        if article["title"]:
            yield PubMedRecord(pmid=pmid, type="title", text=article["title"])
        if article["abstract"]:
            yield PubMedRecord(pmid=pmid, type="abstract", text=article["abstract"])


def build_pubmed_dataset(*file_paths: str | Path) -> Dataset:
    """Build a HuggingFace Dataset of PubMed/MEDLINE title and abstract records."""
    def _records():
        for path in file_paths:
            yield from parse_pubmed_abstracts(path)

    return Dataset.from_generator(
        lambda: ({"pmid": r.pmid, "type": r.type, "text": r.text} for r in _records())
    )


# ---------------------------------------------------------------------------
# PMC (PubMed Central) full articles — JSON metadata + JATS XML
# ---------------------------------------------------------------------------

# All keys present across PMC Open Access JSON metadata files.
_PMC_JSON_KEYS = [
    "pmcid",
    "version",
    "pmid",
    "doi",
    "mid",
    "title",
    "citation",
    "is_pmc_openaccess",
    "is_manuscript",
    "is_historical_ocr",
    "is_retracted",
    "license_code",
    "xml_url",
    "text_url",
    "pdf_url",
    "media_urls",
]


def parse_pmc_directory(directory: str | Path) -> Generator[dict, None, None]:
    """Yield one record per article from a directory of paired PMC JSON + XML files.

    Each record contains all JSON metadata fields plus an ``xml`` key holding
    the raw JATS XML string for the article.
    """
    directory = Path(directory)
    for json_path in directory.glob("*.json"):
        xml_path = json_path.with_suffix(".xml")
        with open(json_path) as fh:
            metadata = json.load(fh)
        xml_content = xml_path.read_text() if xml_path.exists() else None
        record = {key: metadata.get(key) for key in _PMC_JSON_KEYS}
        record["xml"] = xml_content
        yield record


def build_pmc_dataset(*directories: str | Path) -> Dataset:
    """Build a HuggingFace Dataset from one or more PMC article directories.

    Columns mirror the PMC Open Access JSON metadata keys plus an ``xml``
    column containing the raw JATS XML for each article.
    """
    def _records():
        for directory in directories:
            yield from parse_pmc_directory(directory)

    return Dataset.from_generator(_records)
