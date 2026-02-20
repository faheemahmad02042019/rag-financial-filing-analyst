"""
SEC 10-K filing document loader.

Handles loading from local files and the SEC EDGAR API, parsing standard
10-K sections (Risk Factors, MD&A, Business Description, Financial
Statements), cleaning/preprocessing text, and extracting document-level
metadata (company name, CIK, filing date, section type).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from src.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

# Canonical SEC 10-K section identifiers and their common header patterns.
SECTION_PATTERNS: dict[str, list[str]] = {
    "Business": [
        r"(?i)item\s+1[\.\s]*[-—]?\s*business",
        r"(?i)part\s+i\s*[\n\r]+\s*item\s+1\b",
    ],
    "Risk Factors": [
        r"(?i)item\s+1a[\.\s]*[-—]?\s*risk\s+factors",
    ],
    "Properties": [
        r"(?i)item\s+2[\.\s]*[-—]?\s*properties",
    ],
    "Legal Proceedings": [
        r"(?i)item\s+3[\.\s]*[-—]?\s*legal\s+proceedings",
    ],
    "MD&A": [
        r"(?i)item\s+7[\.\s]*[-—]?\s*management'?s?\s+discussion",
    ],
    "Financial Statements": [
        r"(?i)item\s+8[\.\s]*[-—]?\s*financial\s+statements",
    ],
    "Controls and Procedures": [
        r"(?i)item\s+9a[\.\s]*[-—]?\s*controls?\s+and\s+procedures",
    ],
}


@dataclass
class DocumentMetadata:
    """Metadata extracted from or assigned to a filing document."""

    company_name: str = ""
    cik: str = ""
    filing_date: str = ""
    filing_type: str = "10-K"
    section: str = ""
    source_file: str = ""
    accession_number: str = ""


@dataclass
class Document:
    """A single chunk of text with associated metadata."""

    text: str
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    @property
    def page_content(self) -> str:
        """Alias used by LangChain-compatible interfaces."""
        return self.text

    def __len__(self) -> int:
        return len(self.text)


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------


def clean_filing_text(raw_text: str) -> str:
    """Clean raw SEC filing text for downstream processing.

    Applies the following transformations in order:
    1. Strip HTML/XML tags that may remain from EDGAR SGML filings.
    2. Normalise Unicode whitespace and dashes.
    3. Collapse multiple blank lines into a single separator.
    4. Remove page-break artefacts and repeated separator lines.
    5. Strip leading/trailing whitespace.

    Args:
        raw_text: The raw text content of a filing.

    Returns:
        Cleaned text suitable for chunking and embedding.
    """
    # Remove HTML/XML tags
    text = re.sub(r"<[^>]+>", " ", raw_text)

    # Decode common HTML entities
    replacements = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&nbsp;": " ",
        "&quot;": '"',
        "&#8217;": "'",
        "&#8220;": '"',
        "&#8221;": '"',
        "&#8212;": " -- ",
    }
    for entity, replacement in replacements.items():
        text = text.replace(entity, replacement)

    # Normalise whitespace
    text = re.sub(r"\xa0", " ", text)  # non-breaking space
    text = re.sub(r"\t", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse excessive blank lines (3+ -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove page-break lines (e.g., "---", "===", "___" repeated)
    text = re.sub(r"\n[-=_]{3,}\n", "\n", text)

    return text.strip()


def extract_metadata_from_text(text: str, source_path: str = "") -> DocumentMetadata:
    """Heuristically extract metadata from the header of a 10-K filing.

    Looks for common EDGAR header fields such as COMPANY CONFORMED NAME,
    CENTRAL INDEX KEY, and FILED AS OF DATE.

    Args:
        text: The first portion of the filing text.
        source_path: Path to the source file (stored in metadata).

    Returns:
        A populated DocumentMetadata instance.
    """
    metadata = DocumentMetadata(source_file=source_path)

    # Company name
    match = re.search(
        r"COMPANY CONFORMED NAME:\s*(.+?)(?:\n|$)", text, re.IGNORECASE
    )
    if match:
        metadata.company_name = match.group(1).strip()

    # CIK
    match = re.search(r"CENTRAL INDEX KEY:\s*(\d+)", text, re.IGNORECASE)
    if match:
        metadata.cik = match.group(1).strip()

    # Filing date
    match = re.search(r"FILED AS OF DATE:\s*(\d{8})", text, re.IGNORECASE)
    if match:
        raw = match.group(1)
        metadata.filing_date = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    # Accession number
    match = re.search(r"ACCESSION NUMBER:\s*([\d-]+)", text, re.IGNORECASE)
    if match:
        metadata.accession_number = match.group(1).strip()

    return metadata


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


def parse_sections(text: str) -> dict[str, str]:
    """Split a 10-K filing into its standard sections.

    Uses regex patterns to locate section headers and extracts the text
    between consecutive headers.

    Args:
        text: Cleaned full-text of a 10-K filing.

    Returns:
        A dict mapping section names (e.g., "Risk Factors") to their text.
        Sections that could not be located are omitted.
    """
    section_boundaries: list[tuple[int, str]] = []

    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                section_boundaries.append((match.start(), section_name))
                break  # use first matching pattern per section

    if not section_boundaries:
        logger.warning("No standard 10-K sections detected; returning full text.")
        return {"Full Document": text}

    # Sort by position in the document
    section_boundaries.sort(key=lambda x: x[0])

    sections: dict[str, str] = {}
    for idx, (start, name) in enumerate(section_boundaries):
        end = (
            section_boundaries[idx + 1][0]
            if idx + 1 < len(section_boundaries)
            else len(text)
        )
        section_text = text[start:end].strip()
        if section_text:
            sections[name] = section_text

    logger.info(
        "Parsed %d sections: %s",
        len(sections),
        ", ".join(sections.keys()),
    )
    return sections


# ---------------------------------------------------------------------------
# EDGAR API loader
# ---------------------------------------------------------------------------


class EDGARLoader:
    """Load 10-K filings from the SEC EDGAR full-text search API.

    Respects the SEC's fair-access guidelines by setting a descriptive
    User-Agent header and rate-limiting requests.

    Args:
        settings: Application settings containing EDGAR configuration.
    """

    BASE_URL = "https://efts.sec.gov/LATEST/search-index"
    FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    FULL_TEXT_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": settings.sec_edgar_user_agent}
        )
        self._min_interval = 1.0 / settings.sec_edgar_rate_limit
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def get_filing_urls(
        self,
        cik: str,
        filing_type: str = "10-K",
        count: int = 5,
    ) -> list[dict[str, str]]:
        """Retrieve filing index URLs from the EDGAR filing search.

        Args:
            cik: Central Index Key for the company.
            filing_type: Filing type to search (default "10-K").
            count: Maximum number of filings to return.

        Returns:
            A list of dicts with keys 'accession_number', 'filing_date',
            and 'index_url'.
        """
        self._rate_limit()
        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": filing_type,
            "dateb": "",
            "owner": "include",
            "count": str(count),
            "search_text": "",
            "output": "atom",
        }
        try:
            response = self._session.get(self.FILING_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("EDGAR filing search failed for CIK %s: %s", cik, exc)
            return []

        # Parse the Atom XML feed for filing entries
        filings: list[dict[str, str]] = []
        entries = re.findall(
            r"<entry>(.+?)</entry>", response.text, re.DOTALL
        )
        for entry in entries[:count]:
            accession_match = re.search(
                r"accession-number.*?>([\d-]+)<", entry
            )
            date_match = re.search(r"<filing-date>(\d{4}-\d{2}-\d{2})</filing-date>", entry)
            link_match = re.search(r'<link[^>]+href="([^"]+)"', entry)
            if accession_match and link_match:
                filings.append(
                    {
                        "accession_number": accession_match.group(1),
                        "filing_date": date_match.group(1) if date_match else "",
                        "index_url": link_match.group(1),
                    }
                )

        logger.info("Found %d %s filings for CIK %s", len(filings), filing_type, cik)
        return filings

    def download_filing_text(self, cik: str, accession_number: str) -> Optional[str]:
        """Download the full-text content of a specific filing.

        Args:
            cik: Central Index Key (without leading zeros).
            accession_number: Accession number with dashes (e.g. "0000320193-23-000106").

        Returns:
            The raw text of the filing, or ``None`` on failure.
        """
        self._rate_limit()
        cik_padded = cik.lstrip("0") if cik else cik
        accession_clean = accession_number.replace("-", "")
        url = (
            f"{self.FULL_TEXT_URL}/{cik_padded}/{accession_clean}/"
            f"{accession_number}.txt"
        )
        try:
            response = self._session.get(url, timeout=60)
            response.raise_for_status()
            logger.info(
                "Downloaded filing %s for CIK %s (%d bytes)",
                accession_number,
                cik,
                len(response.text),
            )
            return response.text
        except requests.RequestException as exc:
            logger.error(
                "Failed to download filing %s for CIK %s: %s",
                accession_number,
                cik,
                exc,
            )
            return None


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------


class SECFilingLoader:
    """High-level loader that produces ``Document`` objects from 10-K filings.

    Supports two ingestion modes:
    1. **Local files** — read from disk (plain-text or cleaned HTML).
    2. **EDGAR API** — download filings by CIK number.

    In both cases the loader cleans the text, parses sections, extracts
    metadata, and returns one ``Document`` per section.

    Args:
        settings: Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._edgar = EDGARLoader(settings)

    def load_from_file(
        self,
        file_path: str | Path,
        company_name: str = "",
        filing_date: str = "",
    ) -> list[Document]:
        """Load and parse a local 10-K filing.

        Args:
            file_path: Path to the plain-text filing.
            company_name: Override company name (used if header parsing fails).
            filing_date: Override filing date.

        Returns:
            A list of ``Document`` objects, one per detected section.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", path)
            raise FileNotFoundError(f"Filing not found at {path}")

        raw_text = path.read_text(encoding="utf-8", errors="replace")
        logger.info("Loaded %d characters from %s", len(raw_text), path.name)

        cleaned = clean_filing_text(raw_text)
        metadata = extract_metadata_from_text(cleaned, source_path=str(path))

        # Allow caller overrides
        if company_name:
            metadata.company_name = company_name
        if filing_date:
            metadata.filing_date = filing_date

        sections = parse_sections(cleaned)

        documents: list[Document] = []
        for section_name, section_text in sections.items():
            doc_meta = DocumentMetadata(
                company_name=metadata.company_name,
                cik=metadata.cik,
                filing_date=metadata.filing_date,
                filing_type=metadata.filing_type,
                section=section_name,
                source_file=metadata.source_file,
                accession_number=metadata.accession_number,
            )
            documents.append(Document(text=section_text, metadata=doc_meta))

        logger.info(
            "Produced %d section documents from %s", len(documents), path.name
        )
        return documents

    def load_from_edgar(
        self,
        cik: str,
        company_name: str = "",
        count: int = 1,
    ) -> list[Document]:
        """Download and parse 10-K filings from EDGAR.

        Args:
            cik: Central Index Key for the target company.
            company_name: Human-readable company name.
            count: Number of most-recent 10-K filings to fetch.

        Returns:
            A list of ``Document`` objects across all downloaded filings.
        """
        filing_refs = self._edgar.get_filing_urls(cik, count=count)
        if not filing_refs:
            logger.warning("No filings found on EDGAR for CIK %s", cik)
            return []

        all_documents: list[Document] = []
        for ref in filing_refs:
            raw = self._edgar.download_filing_text(cik, ref["accession_number"])
            if raw is None:
                continue

            cleaned = clean_filing_text(raw)
            metadata = extract_metadata_from_text(cleaned)
            metadata.company_name = company_name or metadata.company_name
            metadata.cik = cik
            metadata.filing_date = ref.get("filing_date", metadata.filing_date)
            metadata.accession_number = ref["accession_number"]

            sections = parse_sections(cleaned)
            for section_name, section_text in sections.items():
                doc_meta = DocumentMetadata(
                    company_name=metadata.company_name,
                    cik=metadata.cik,
                    filing_date=metadata.filing_date,
                    filing_type="10-K",
                    section=section_name,
                    source_file=f"edgar://{cik}/{ref['accession_number']}",
                    accession_number=metadata.accession_number,
                )
                all_documents.append(
                    Document(text=section_text, metadata=doc_meta)
                )

        logger.info(
            "Loaded %d section documents from EDGAR for CIK %s",
            len(all_documents),
            cik,
        )
        return all_documents

    def load_directory(
        self,
        directory: str | Path,
        glob_pattern: str = "*.txt",
        company_name: str = "",
    ) -> list[Document]:
        """Batch-load all filings from a local directory.

        Args:
            directory: Path to the directory containing filing files.
            glob_pattern: File-matching pattern (default ``*.txt``).
            company_name: Default company name if not extractable from headers.

        Returns:
            A combined list of ``Document`` objects from all files.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        files = sorted(dir_path.glob(glob_pattern))
        logger.info("Found %d files matching '%s' in %s", len(files), glob_pattern, dir_path)

        all_documents: list[Document] = []
        for file_path in files:
            try:
                docs = self.load_from_file(
                    file_path, company_name=company_name
                )
                all_documents.extend(docs)
            except Exception as exc:
                logger.error("Failed to load %s: %s", file_path, exc)

        return all_documents
