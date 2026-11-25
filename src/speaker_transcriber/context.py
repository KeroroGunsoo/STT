from __future__ import annotations

import difflib
import html
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXTS = {
    ".txt", ".md", ".tex", ".bib", ".py", ".r", ".json", ".yaml", ".yml",
    ".csv", ".html", ".xml", ".docx", ".odt",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*")

STOPWORDS = frozenset(
    "a an the and or but if then else when where while of at by for with about against between into "
    "through during before after above below to from up down in out on off over under again further "
    "once here there all any both each few more most other some such no nor not only own same so than "
    "too very can will just should now is are was were be been being have has had having do does did "
    "doing would could shall may might must it its this that these those i you he she we they them his "
    "her their our your my me him us as et al fig figure table section chapter eq ref refs see show "
    "shown using use used based new et cetera vs per via within without".split()
)


@dataclass
class ContextProfile:
    source_dir: str
    terms: list[str]
    prompt: str

    def to_dict(self) -> dict:
        return {"source_dir": self.source_dir, "num_terms": len(self.terms), "terms": self.terms, "prompt": self.prompt}


def extract_text_from_file(path: str | Path, max_bytes: int = 2_000_000) -> str:
    path = Path(path)
    if path.stat().st_size > max_bytes:
        return ""
    suffix = path.suffix.lower()
    if suffix in (".docx", ".odt"):
        member = "word/document.xml" if suffix == ".docx" else "content.xml"
        try:
            with zipfile.ZipFile(path) as z:
                xml = z.read(member).decode("utf-8", errors="ignore")
        except Exception:
            return ""
        return html.unescape(re.sub(r"<[^>]+>", " ", xml))
    return path.read_bytes().decode("utf-8", errors="ignore")


def iter_context_files(folder: str | Path, extensions: set[str] = DEFAULT_EXTS, max_bytes: int = 2_000_000):
    folder = Path(folder)
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        if path.suffix.lower() not in extensions:
            continue
        if path.stat().st_size > max_bytes:
            continue
        yield path


def build_vocabulary(folder: str | Path, max_terms: int = 300, min_freq: int = 2) -> list[str]:
    """Extract domain terms (technical words, acronyms, names) from a project folder.

    Counting is case-insensitive (so "retrieval"/"Retrieval" count together);
    the most frequent casing wins, preferring capitalized forms on ties.
    Acronyms (ALL-CAPS, 2-8 chars) are kept even at frequency 1.
    """
    counts: Counter[str] = Counter()
    casings: dict[str, Counter[str]] = {}
    for path in iter_context_files(folder):
        for tok in TOKEN_RE.findall(extract_text_from_file(path)):
            low = tok.lower()
            if low in STOPWORDS or tok.isdigit():
                continue
            counts[low] += 1
            casings.setdefault(low, Counter())[tok] += 1

    def best_casing(low: str) -> str:
        return max(casings[low].items(), key=lambda kv: (kv[1], kv[0] != kv[0].lower()))[0]

    terms: list[str] = []
    seen: set[str] = set()
    for low, _ in counts.most_common():
        tok = best_casing(low)
        if tok.isupper() and 2 <= len(tok) <= 8 and low not in seen:
            seen.add(low)
            terms.append(tok)
    for low, count in counts.most_common():
        if len(terms) >= max_terms:
            break
        if count < min_freq:
            break
        if low in seen:
            continue
        seen.add(low)
        terms.append(best_casing(low))
    return terms[:max_terms]


def build_prompt(terms: list[str], max_chars: int = 400) -> str:
    head = "Transcript of an academic/research discussion. Domain terms that may appear: "
    out = head
    for term in terms:
        piece = (", " if out != head else "") + term
        if len(out) + len(piece) > max_chars:
            break
        out += piece
    return out


def build_profile(folder: str | Path, max_terms: int = 300) -> ContextProfile:
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Context folder not found: {folder}")
    terms = build_vocabulary(folder, max_terms=max_terms)
    return ContextProfile(source_dir=str(folder), terms=terms, prompt=build_prompt(terms))


def correct_text(text: str, terms: list[str], threshold: float = 0.92, min_len: int = 5) -> str:
    """Replace likely mis-heard words with the closest known domain term.

    Fixes casing for known terms and fuzzy-matches unknown tokens (ratio >=
    ``threshold``, length >= ``min_len``) against the vocabulary, preserving
    punctuation and spacing.
    """
    if not terms or not text:
        return text
    lower_map: dict[str, str] = {}
    for term in terms:
        lower_map.setdefault(term.lower(), term)
    vocab = set(lower_map)

    def repl(match: re.Match) -> str:
        token = match.group(0)
        low = token.lower()
        if low in vocab:
            return lower_map[low]
        if len(low) < min_len or low in STOPWORDS:
            return token
        close = difflib.get_close_matches(low, vocab, n=1, cutoff=threshold)
        return lower_map[close[0]] if close else token

    return TOKEN_RE.sub(repl, text)
