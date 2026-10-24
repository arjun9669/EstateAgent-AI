"""Download official Dubai tenancy legislation and convert it to article text.

    python scripts/fetch_rules.py

Raw HTML is cached in data/raw/rules_html/ (git-ignored). Output goes to
data/rules/ as one Markdown file per document, one "## Article (N)" section per
article, which search_rules indexes as citable passages.

Law No. 33 of 2008 supersedes eleven articles of Law No. 26 of 2007. The output
for Law 26 is consolidated mechanically: each superseded article is replaced by
the Law 33 text and labelled as such. Law 33's own list of superseded articles
is checked against the articles actually replaced.
"""

import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "rules_html"
OUT = ROOT / "data" / "rules"
PORTAL = "https://dlp.dubai.gov.ae/Legislation%20Reference/"
SOURCES = {
    "law-26-2007": PORTAL + "2007/Law%20No.%20(26)%20of%202007%20Regarding%20regulation%20between%20the%20lessor%20and%20the%20lessee.html",
    "law-33-2008": PORTAL + "2009/Law%20No.%20(33)%20of%202008%20Amending%20Law%20No.%20(26)%20of%202007.html",
    "decree-43-2013": PORTAL + "2013/Decree%20No.%20(43)%20of%202013%20Determining%20Rent%20Increase%20for%20Real%20Property.html",
}
TITLES = {
    "law-26-2007": "Law No. 26 of 2007 Regulating the Relationship between Landlords and Tenants in the Emirate of Dubai (as amended by Law No. 33 of 2008)",
    "decree-43-2013": "Decree No. 43 of 2013 Determining Rent Increase for Real Property in the Emirate of Dubai",
}
RETRIEVED = "2026-10-24"
BLOCK_TAGS = {"p", "div", "li", "td", "h1", "h2", "h3", "h4", "h5", "h6", "br"}
ARTICLE = re.compile(r"^\W*Article \((\d+)\)\s*(.*)$", re.S)


class Blocks(HTMLParser):
    """Collect the visible text of block-level elements, one string per block."""

    def __init__(self):
        super().__init__()
        self.blocks, self._parts, self._skip = [], [], 0

    def _flush(self):
        text = " ".join("".join(self._parts).replace("�", " ").split())
        if text:
            self.blocks.append(text)
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1
        elif tag in BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def close(self):
        super().close()
        self._flush()


def fetch(name: str) -> str:
    path = RAW / f"{name}.html"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCES[name], timeout=60) as response:
            path.write_bytes(response.read())
    parser = Blocks()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    parser.close()
    return parser.blocks


def articles(blocks: list[str]) -> list[dict]:
    """Split the operative text into articles: number, optional heading, paragraphs.

    The text runs from the first Article (1) up to the Ruler's signature. A short
    block directly before an article number is that article's heading.
    """
    start = next(i for i, b in enumerate(blocks) if ARTICLE.match(b) and ARTICLE.match(b)[1] == "1")
    result, heading = [], None
    for block in blocks[start - 1:]:
        if block.startswith("Mohammed bin Rashid"):
            break
        match = ARTICLE.match(block)
        if match:
            result.append({"number": int(match[1]), "heading": heading, "paragraphs": []})
            heading = None
            if match[2]:
                result[-1]["paragraphs"].append(match[2])
        elif result and not heading and len(block.split()) <= 6 and not block.endswith((".", ";", ":")) \
                and result[-1]["paragraphs"]:
            # Candidate heading for the next article; keep it if no article follows.
            heading = block
        elif heading:
            result[-1]["paragraphs"].extend([heading, block])
            heading = None
        elif result:
            result[-1]["paragraphs"].append(block)
        else:
            heading = block if len(block.split()) <= 6 else None
    return result


def superseded_numbers(intro: str) -> list[int]:
    """Article numbers listed in Law 33's 'Articles (2), (3) ... are superseded' sentence."""
    if "superseded" not in intro:
        raise SystemExit("Law 33 Article (1) no longer says which articles it supersedes")
    return [int(n) for n in re.findall(r"\((\d+)\)", intro.split("superseded")[0])]


def consolidate(original: list[dict], amending: list[dict]) -> list[dict]:
    intro = amending[0]
    numbers = superseded_numbers(" ".join(intro["paragraphs"]))
    # Law 33 = Article (1) intro, the replacement articles, then its own Article (2).
    replacements = amending[1:-1]
    if [a["number"] for a in replacements] != numbers:
        raise SystemExit(f"Replacement articles {[a['number'] for a in replacements]} != listed {numbers}")
    by_number = {a["number"]: a for a in replacements}
    merged = []
    for article in original:
        if article["number"] in by_number:
            new = dict(by_number[article["number"]], heading=article["heading"])
            new["amended"] = True
            merged.append(new)
        else:
            merged.append(article)
    if [a["number"] for a in merged] != list(range(1, len(merged) + 1)):
        raise SystemExit("Consolidated articles are not numbered 1..N")
    return merged


def write(name: str, items: list[dict], sources: list[str]) -> None:
    lines = [f"# {TITLES[name]}", ""]
    lines += [f"Source: {url}" for url in sources]
    lines += [
        f"Retrieved: {RETRIEVED}",
        "",
        "Official English version from the Dubai Legislation Portal. The Arabic text "
        "prevails in case of conflict. Laws are not protected by copyright under UAE "
        "Federal Decree-Law No. 38 of 2021, Article 3(2).",
        "",
    ]
    for article in items:
        title = f"Article ({article['number']})"
        if article.get("heading"):
            title += f" {article['heading']}"
        lines.append(f"## {title}")
        lines.append("")
        if article.get("amended"):
            lines += ["*Text as superseded by Law No. 33 of 2008.*", ""]
        for paragraph in article["paragraphs"]:
            lines += [paragraph, ""]
    (OUT / f"{name}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote data/rules/{name}.md ({len(items)} articles)")


def main():
    law26 = articles(fetch("law-26-2007"))
    law33 = articles(fetch("law-33-2008"))
    decree43 = articles(fetch("decree-43-2013"))
    write("law-26-2007", consolidate(law26, law33), [SOURCES["law-26-2007"], SOURCES["law-33-2008"]])
    write("decree-43-2013", decree43, [SOURCES["decree-43-2013"]])


if __name__ == "__main__":
    sys.exit(main())
