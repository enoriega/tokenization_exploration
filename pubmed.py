import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, Literal

import pubmed_parser
from datasets import Dataset
from tqdm import tqdm


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

@dataclass
class PMCRecord:
    pmcid: str | None
    version: str | None
    pmid: str | None
    doi: str | None
    mid: str | None
    title: str | None
    citation: str | None
    is_pmc_openaccess: bool | None
    is_manuscript: bool | None
    is_historical_ocr: bool | None
    is_retracted: bool | None
    license_code: str | None
    xml_url: str | None
    text_url: str | None
    pdf_url: str | None
    media_urls: list | None
    xml: str | None


def parse_pmc_directory(directory: str | Path) -> Generator[PMCRecord, None, None]:
    """Yield one PMCRecord per article from a directory of paired PMC JSON + XML files."""
    directory = Path(directory)
    for json_path in directory.glob("*.json"):
        xml_path = json_path.with_suffix(".xml")
        with open(json_path) as fh:
            try:
                metadata = json.load(fh)
            except:
                # To avoid crashing the whole thing if the json is illegal and fails to parse
                metadata = dict()

        try:
            xml_content = xml_path.read_text() if xml_path.exists() else None
        except:
            # Similarly, fail silently if an xml didn't parse and don't crash the whole thing
            xml_content = None
            
        yield PMCRecord(
            pmcid=metadata.get("pmcid"),
            version=str(v) if (v := metadata.get("version")) is not None else None,
            pmid=metadata.get("pmid"),
            doi=metadata.get("doi"),
            mid=metadata.get("mid"),
            title=metadata.get("title"),
            citation=metadata.get("citation"),
            is_pmc_openaccess=metadata.get("is_pmc_openaccess"),
            is_manuscript=metadata.get("is_manuscript"),
            is_historical_ocr=metadata.get("is_historical_ocr"),
            is_retracted=metadata.get("is_retracted"),
            license_code=metadata.get("license_code"),
            xml_url=metadata.get("xml_url"),
            text_url=metadata.get("text_url"),
            pdf_url=metadata.get("pdf_url"),
            media_urls=metadata.get("media_urls"),
            xml=xml_content,
        )


def build_pmc_dataset(directory: str | Path) -> Dataset:
    """Build a HuggingFace Dataset from a directory with the same layout as the AWS bucket with PMC articles

    Columns mirror the PMC Open Access JSON metadata keys plus an ``xml``
    column containing the raw JATS XML for each article.
    """

    if isinstance(directory, str):
        directory = Path(directory)

    def _records():
        # Some papers have more than one version, in such cases, we want to use one.
        # This bookkeeping help us jump the duplicates
        seen = set()
        for paper_dir in directory.glob("PMC*/"):
            pmcid = paper_dir.name.split(".")[0]
            if pmcid not in seen:
                for record in parse_pmc_directory(paper_dir):
                    yield asdict(record)
                seen.add(pmcid)

    return Dataset.from_generator(_records)
