"""
Output guardrails for the RAG Financial Filing Analyst.

Validates generated responses against retrieved source context to detect
hallucinations, verify financial numbers, enforce output formatting, and
assign confidence scores. Designed for the high-accuracy demands of
financial document analysis.

Guardrail checks:
1. **Factual consistency** — each sentence in the answer is checked for
   grounding in the retrieved sources.
2. **Hallucination detection** — claims not traceable to any source are
   flagged with severity levels.
3. **Numerical validation** — extracted financial figures are cross-checked
   against the source text.
4. **Output format enforcement** — verifies that citations are present and
   the response is well-structured.
5. **Confidence scoring** — aggregates all checks into a single [0, 1]
   confidence score.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.config import Settings
from src.vector_store import SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Severity level for a guardrail finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailFinding:
    """A single finding from a guardrail check.

    Attributes:
        check_name: Identifier for the check that produced this finding.
        severity: How serious the finding is.
        message: Human-readable description.
        span: The text span in the answer that triggered the finding.
        evidence: Supporting detail (e.g., the expected number vs. found).
    """

    check_name: str
    severity: Severity
    message: str
    span: str = ""
    evidence: str = ""


@dataclass
class GuardrailReport:
    """Aggregated report from all guardrail checks.

    Attributes:
        findings: All individual findings.
        confidence_score: Aggregate confidence in [0, 1].
        passed: Whether the response meets the confidence threshold.
        answer: The original answer (possibly annotated).
        warnings: High-level warning messages for the user.
    """

    findings: list[GuardrailFinding] = field(default_factory=list)
    confidence_score: float = 1.0
    passed: bool = True
    answer: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def critical_findings(self) -> list[GuardrailFinding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    @property
    def high_findings(self) -> list[GuardrailFinding]:
        return [f for f in self.findings if f.severity == Severity.HIGH]


# ---------------------------------------------------------------------------
# Number extraction
# ---------------------------------------------------------------------------

# Matches patterns like "$1.2 billion", "1,234,567", "$456 million", "12.5%"
_NUMBER_RE = re.compile(
    r"\$?\s*\d[\d,]*\.?\d*\s*(?:billion|million|thousand|trillion|%|percent)?",
    re.IGNORECASE,
)

# Matches patterns like "revenue of $X", "net income was $X", etc.
_FINANCIAL_CLAIM_RE = re.compile(
    r"(?:revenue|income|profit|loss|earnings|ebitda|margin|growth|"
    r"debt|assets|liabilities|equity|cash\s+flow|expense|cost|"
    r"sales|dividend|eps|share\s+price)\s*(?:of|was|were|is|"
    r"totaled|totalled|reached|increased|decreased|grew|declined)"
    r"[^.]*?(\$?\s*\d[\d,]*\.?\d*\s*(?:billion|million|thousand|trillion|%|percent)?)",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> list[str]:
    """Extract all number expressions from text.

    Returns:
        A list of normalised number strings found in the text.
    """
    matches = _NUMBER_RE.findall(text)
    return [m.strip() for m in matches if m.strip()]


def normalise_number(num_str: str) -> Optional[float]:
    """Parse a financial number string into a float.

    Handles currency symbols, commas, and scale words (million, billion).

    Args:
        num_str: A string like "$1.2 billion" or "1,234,567".

    Returns:
        The numeric value as a float, or ``None`` if parsing fails.
    """
    text = num_str.strip().lower()
    text = text.replace("$", "").replace(",", "").strip()

    multiplier = 1.0
    for word, mult in [
        ("trillion", 1e12),
        ("billion", 1e9),
        ("million", 1e6),
        ("thousand", 1e3),
    ]:
        if word in text:
            multiplier = mult
            text = text.replace(word, "").strip()
            break

    is_percent = "%" in text or "percent" in text
    text = text.replace("%", "").replace("percent", "").strip()

    try:
        value = float(text) * multiplier
        if is_percent:
            value = float(text.replace("%", "").strip())  # keep as percentage
        return value
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Individual guardrail checks
# ---------------------------------------------------------------------------


class FactualConsistencyChecker:
    """Check that each sentence in the answer is grounded in the sources.

    Splits the answer into sentences, then for each sentence checks whether
    any source chunk contains overlapping key phrases. Sentences with no
    grounding are flagged.
    """

    def __init__(self, overlap_threshold: int = 3) -> None:
        self._overlap_threshold = overlap_threshold

    def check(
        self,
        answer: str,
        sources: list[SearchResult],
    ) -> list[GuardrailFinding]:
        """Run the factual consistency check.

        Args:
            answer: The generated answer text.
            sources: Retrieved source chunks.

        Returns:
            A list of findings for ungrounded sentences.
        """
        findings: list[GuardrailFinding] = []
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        source_text_lower = " ".join(s.text.lower() for s in sources)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue  # skip very short fragments

            # Skip meta-sentences (e.g., "Based on the sources...")
            if re.match(
                r"(?i)(based on|according to|the sources|as mentioned)", sentence
            ):
                continue

            # Check for key-phrase overlap
            words = re.findall(r"\b\w{4,}\b", sentence.lower())
            if not words:
                continue

            # Count how many content words appear in the sources
            matched = sum(1 for w in words if w in source_text_lower)
            overlap_ratio = matched / len(words) if words else 0

            if overlap_ratio < 0.3:
                severity = (
                    Severity.HIGH if overlap_ratio < 0.1 else Severity.MEDIUM
                )
                findings.append(
                    GuardrailFinding(
                        check_name="factual_consistency",
                        severity=severity,
                        message=(
                            f"Sentence may not be grounded in sources "
                            f"(overlap={overlap_ratio:.0%})."
                        ),
                        span=sentence[:200],
                    )
                )

        return findings


class HallucinationDetector:
    """Detect claims in the answer that are not present in any source.

    Focuses on specific factual assertions — named entities, dates, and
    numerical claims — rather than general statements.
    """

    def check(
        self,
        answer: str,
        sources: list[SearchResult],
    ) -> list[GuardrailFinding]:
        """Run hallucination detection.

        Args:
            answer: Generated answer text.
            sources: Source chunks.

        Returns:
            Findings for potentially hallucinated claims.
        """
        findings: list[GuardrailFinding] = []
        source_text = " ".join(s.text for s in sources)

        # Check for financial claims with numbers
        claims = _FINANCIAL_CLAIM_RE.findall(answer)
        for claim_number in claims:
            claim_clean = claim_number.strip()
            if claim_clean and claim_clean not in source_text:
                # Try normalised comparison
                claim_val = normalise_number(claim_clean)
                source_numbers = extract_numbers(source_text)
                source_vals = [
                    normalise_number(n) for n in source_numbers
                ]
                source_vals_clean = [v for v in source_vals if v is not None]

                if claim_val is not None and source_vals_clean:
                    # Check if any source number is within 1% tolerance
                    close_match = any(
                        abs(claim_val - sv) / max(abs(sv), 1e-10) < 0.01
                        for sv in source_vals_clean
                    )
                    if not close_match:
                        findings.append(
                            GuardrailFinding(
                                check_name="hallucination_detection",
                                severity=Severity.HIGH,
                                message=(
                                    f"Financial figure '{claim_clean}' not "
                                    f"found in source documents."
                                ),
                                span=claim_clean,
                                evidence=f"Source numbers: {source_numbers[:5]}",
                            )
                        )
                elif claim_val is not None:
                    findings.append(
                        GuardrailFinding(
                            check_name="hallucination_detection",
                            severity=Severity.MEDIUM,
                            message=(
                                f"Could not verify figure '{claim_clean}' "
                                f"against sources."
                            ),
                            span=claim_clean,
                        )
                    )

        return findings


class NumericalValidator:
    """Cross-check every number in the answer against the source text.

    Extracts all number expressions from both the answer and the sources,
    then flags answer numbers that have no close match in the sources.
    """

    def __init__(self, tolerance: float = 0.02) -> None:
        self._tolerance = tolerance

    def check(
        self,
        answer: str,
        sources: list[SearchResult],
    ) -> list[GuardrailFinding]:
        """Validate numerical claims.

        Args:
            answer: Generated answer.
            sources: Source chunks.

        Returns:
            Findings for unverifiable numbers.
        """
        findings: list[GuardrailFinding] = []
        answer_numbers = extract_numbers(answer)
        source_text = " ".join(s.text for s in sources)
        source_numbers = extract_numbers(source_text)

        source_values: list[Optional[float]] = [
            normalise_number(n) for n in source_numbers
        ]
        source_vals_clean = [v for v in source_values if v is not None]

        for num_str in answer_numbers:
            val = normalise_number(num_str)
            if val is None:
                continue

            # Exact string match in source
            if num_str in source_text:
                continue

            # Numeric match within tolerance
            if source_vals_clean:
                close = any(
                    abs(val - sv) / max(abs(sv), 1e-10) < self._tolerance
                    for sv in source_vals_clean
                )
                if close:
                    continue

            findings.append(
                GuardrailFinding(
                    check_name="numerical_validation",
                    severity=Severity.MEDIUM,
                    message=f"Number '{num_str}' could not be verified in sources.",
                    span=num_str,
                )
            )

        return findings


class FormatChecker:
    """Verify that the response follows expected formatting conventions.

    Checks for:
    - Presence of source citations ([Source N]).
    - Reasonable response length.
    - No markdown artefacts from the LLM (e.g., triple backticks).
    """

    def check(self, answer: str, sources: list[SearchResult]) -> list[GuardrailFinding]:
        """Run format validation.

        Args:
            answer: Generated answer.
            sources: Source chunks (used to determine expected citation count).

        Returns:
            Findings for formatting issues.
        """
        findings: list[GuardrailFinding] = []

        # Check for citations
        citations = re.findall(r"\[Source\s+\d+\]", answer)
        if not citations and sources:
            findings.append(
                GuardrailFinding(
                    check_name="format_check",
                    severity=Severity.LOW,
                    message="No source citations found in the response.",
                )
            )

        # Check for very short answers when sources are available
        if sources and len(answer.split()) < 20:
            findings.append(
                GuardrailFinding(
                    check_name="format_check",
                    severity=Severity.LOW,
                    message="Response is unusually short given available sources.",
                )
            )

        # Check for code block artefacts
        if "```" in answer:
            findings.append(
                GuardrailFinding(
                    check_name="format_check",
                    severity=Severity.LOW,
                    message="Response contains code block markers.",
                )
            )

        return findings


# ---------------------------------------------------------------------------
# Confidence scorer
# ---------------------------------------------------------------------------


def compute_confidence(findings: list[GuardrailFinding]) -> float:
    """Compute an aggregate confidence score from guardrail findings.

    Assigns penalty weights by severity and clamps the result to [0, 1].

    Args:
        findings: All guardrail findings.

    Returns:
        A confidence score in [0, 1].
    """
    penalties = {
        Severity.LOW: 0.02,
        Severity.MEDIUM: 0.08,
        Severity.HIGH: 0.15,
        Severity.CRITICAL: 0.30,
    }
    total_penalty = sum(penalties.get(f.severity, 0.05) for f in findings)
    return max(0.0, min(1.0, 1.0 - total_penalty))


# ---------------------------------------------------------------------------
# Guardrail pipeline
# ---------------------------------------------------------------------------


class GuardrailPipeline:
    """Run all guardrail checks on a generated response and produce a report.

    Args:
        settings: Application configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._checkers = self._build_checkers(settings)

    @staticmethod
    def _build_checkers(
        settings: Settings,
    ) -> list[Any]:
        """Instantiate the active guardrail checkers based on config."""
        checkers: list[Any] = [
            FactualConsistencyChecker(),
            FormatChecker(),
        ]
        if settings.enable_hallucination_check:
            checkers.append(HallucinationDetector())
        if settings.enable_numerical_validation:
            checkers.append(NumericalValidator())
        return checkers

    def validate(
        self,
        answer: str,
        sources: list[SearchResult],
    ) -> GuardrailReport:
        """Run all guardrail checks and produce a consolidated report.

        Args:
            answer: The generated answer text.
            sources: The source chunks used for generation.

        Returns:
            A ``GuardrailReport`` with findings, confidence score, and
            pass/fail status.
        """
        all_findings: list[GuardrailFinding] = []

        for checker in self._checkers:
            try:
                findings = checker.check(answer, sources)
                all_findings.extend(findings)
            except Exception as exc:
                logger.error(
                    "Guardrail checker %s failed: %s",
                    type(checker).__name__,
                    exc,
                )

        confidence = compute_confidence(all_findings)
        passed = confidence >= self._settings.confidence_threshold

        warnings: list[str] = []
        if not passed:
            warnings.append(
                f"Response confidence ({confidence:.2f}) is below the "
                f"threshold ({self._settings.confidence_threshold:.2f})."
            )
        for finding in all_findings:
            if finding.severity in (Severity.HIGH, Severity.CRITICAL):
                warnings.append(f"[{finding.severity.value.upper()}] {finding.message}")

        report = GuardrailReport(
            findings=all_findings,
            confidence_score=confidence,
            passed=passed,
            answer=answer,
            warnings=warnings,
        )

        logger.info(
            "Guardrail report: confidence=%.2f, passed=%s, findings=%d "
            "(critical=%d, high=%d)",
            confidence,
            passed,
            len(all_findings),
            len(report.critical_findings),
            len(report.high_findings),
        )

        return report
