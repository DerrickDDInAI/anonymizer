"""
Regex extractor: finds entities that follow a fixed, predictable shape.

Emails, phone numbers, URLs, IBANs, social security numbers, amounts and ISO
dates all look the same every time. A pattern ("regular expression", regex)
finds them more reliably than a language model, and it never misses one because
the sentence is unusual. Names, companies, places and jobs do NOT have a fixed
shape, so those are left to the ML models.

This extractor is applied on top of every model, so all models get the same
help and the comparison between them stays fair.
"""

import re

from anonymizer.extractors.base import Extractor
from anonymizer.spans import Span, overlaps

# Patterns in priority order: when two patterns claim the same characters, the
# one higher in this list wins. Example: an IBAN also looks like a phone number
# (lots of digits), so IBAN is listed before PHONE_NUMBER.
PATTERNS: list[tuple[str, re.Pattern]] = [
    # something@something.something
    ("EMAIL_ADDRESS", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    # http://... or https://... (stops before spaces; last char may not be punctuation)
    # or www.something.something
    (
        "URL",
        re.compile(r"""https?://[^\s<>"']*[^\s<>"'.,;:!?)]|www\.[\w-]+(?:\.[\w-]+)+""", re.IGNORECASE),
    ),
    # 2 letters, 2 digits, then groups of 4 letters/digits: BE68 5390 0754 7034
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")),
    # Belgian national number 82.05.30-025.56  or  US social security number 123-45-6789
    ("SSN", re.compile(r"\b\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}\b|\b\d{3}-\d{2}-\d{4}\b")),
    # ISO date 2022-12-27, optionally followed by a time 08:26:49.21
    ("DATE_TIME", re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?\b")),
    # Optional +country code, then digits with spaces / dots / dashes / brackets in
    # between: 015 29 58 58   +1 (202) 555-0199   (digit count is checked afterwards)
    ("PHONE_NUMBER", re.compile(r"(?<![\w.-])\+?\(?\d[\d ().-]{6,}\d(?![\w-])")),
    # Currency before the number ($1.65 billion, €50.000) or after it (50 euros)
    (
        "AMOUNT",
        re.compile(
            r"(?:[$€£]|\b(?:EUR|USD|GBP)\b)\s?\d(?:[\d.,]*\d)?(?:\s?(?:thousand|million|billion|trillion|k|m|bn))?\b"
            r"|\b\d(?:[\d.,]*\d)?\s?(?:thousand|million|billion|trillion)?\s?(?:€|£|\$|EUR|USD|GBP|euros?|dollars?|pounds?)\b",
            re.IGNORECASE,
        ),
    ),
]


def iban_is_valid(iban: str) -> bool:
    """
    Check an IBAN with the official "mod 97" rule, so random letter+digit
    combinations are not mistaken for a bank account.

    Steps: remove spaces, move the first 4 characters to the end, turn letters
    into numbers (A=10, B=11, ... Z=35), and the result must be divisible by 97
    with remainder 1.
    """
    compact = iban.replace(" ", "").upper()
    rearranged = compact[4:] + compact[:4]
    as_digits = "".join(str(int(ch, 36)) for ch in rearranged)  # int("B", 36) == 11
    return int(as_digits) % 97 == 1


def looks_like_phone(candidate: str) -> bool:
    """A phone number has between 8 and 15 digits (avoids years, zip codes, ...)."""
    digits = sum(ch.isdigit() for ch in candidate)
    return 8 <= digits <= 15


# Extra checks that a raw pattern match must pass before we accept it.
VALIDATORS = {
    "IBAN": iban_is_valid,
    "PHONE_NUMBER": looks_like_phone,
}


class RegexExtractor(Extractor):
    """Finds the structured entity types with the patterns above."""

    name = "regex"

    def extract(self, text: str) -> list[Span]:
        found: list[Span] = []
        for label, pattern in PATTERNS:
            validator = VALIDATORS.get(label)
            for match in pattern.finditer(text):
                if validator and not validator(match.group()):
                    continue  # matched the shape, but failed the extra check
                span = Span(match.start(), match.end(), label, source=self.name)
                # Skip if a higher-priority pattern already claimed these characters.
                if any(overlaps(span, other) for other in found):
                    continue
                found.append(span)
        return sorted(found, key=lambda s: s.start)
