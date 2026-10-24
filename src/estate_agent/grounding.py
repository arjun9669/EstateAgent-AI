"""Check that an answer's figures and citations come from the evidence.

The agent may only state numbers that appear in a tool result or the user's
question. A number counts as grounded if some evidence value, scaled by the
same suffix (k, million...) and rounded to the same decimal places, equals it:
"AED 2.3 million" is grounded by 2,277,500 and "2,139" by 2138.9.
"""

import re
from dataclasses import dataclass, field

NUMBER = re.compile(
    r"(?<![A-Za-z#.\d])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?:\s?(k|m|million|bn|billion)\b)?",
    re.IGNORECASE,
)
SCALES = {"k": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
# Small integers are list markers, counts and ordinals far more often than facts,
# so they are not checked, unless they are percentages ("5%").
ALWAYS_ALLOWED_MAX = 10
PASSAGE_ID = re.compile(r"\b[a-z0-9-]+#art\d+\b")
LISTING_ID = re.compile(r"\bL\d{4}\b")


@dataclass(frozen=True)
class Number:
    text: str
    raw: float  # as written, before the suffix
    scale: float
    decimals: int
    percent: bool = False

    @property
    def value(self) -> float:
        return self.raw * self.scale


def numbers_in(text: str) -> list[Number]:
    found = []
    for match in NUMBER.finditer(text):
        whole, fraction, suffix = match.groups()
        raw = float(whole.replace(",", "") + (f".{fraction}" if fraction else ""))
        scale = SCALES[suffix.lower()] if suffix else 1.0
        percent = text[match.end():match.end() + 2].lstrip()[:1] == "%"
        found.append(Number(match.group(0), raw, scale, len(fraction or ""), percent))
    return found


def _matches(number: Number, evidence: float) -> bool:
    return abs(round(evidence / number.scale, number.decimals) - number.raw) < 1e-9


@dataclass
class GroundingReport:
    ungrounded_numbers: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.ungrounded_numbers and not self.unknown_citations

    def as_dict(self) -> dict:
        return {"ok": self.ok, "ungrounded_numbers": self.ungrounded_numbers,
                "unknown_citations": self.unknown_citations}


def check(answer: str, evidence: list[str]) -> GroundingReport:
    joined = "\n".join(evidence)
    values = {n.value for n in numbers_in(joined)}
    report = GroundingReport()
    # Digits inside IDs (L0435, law-26-2007#art14) are checked as citations, not numbers.
    prose = LISTING_ID.sub(" ", PASSAGE_ID.sub(" ", answer))
    for number in numbers_in(prose):
        if (number.scale == 1 and number.decimals == 0 and not number.percent
                and number.raw <= ALWAYS_ALLOWED_MAX):
            continue
        if not any(_matches(number, v) for v in values) and number.text not in report.ungrounded_numbers:
            report.ungrounded_numbers.append(number.text)
    for pattern in (PASSAGE_ID, LISTING_ID):
        for citation in pattern.findall(answer):
            if citation not in joined and citation not in report.unknown_citations:
                report.unknown_citations.append(citation)
    return report
