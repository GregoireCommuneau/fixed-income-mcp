"""
scripts/build_wiki.py
Builds the GitHub wiki pages from docs/var-backtest-guide.md, the source of truth.

The guide is split into one page per chapter (section numbers are kept, so
"section 5.4" means the same place everywhere), with a Home page, a sidebar,
a footer, and the figures copied into the wiki's images/ folder.

Usage:
    git clone https://github.com/GregoireCommuneau/fixed-income-mcp.wiki.git ../fixed-income-mcp.wiki
    python scripts/build_wiki.py ../fixed-income-mcp.wiki
    # then commit and push from the wiki clone
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "var-backtest-guide.md"
README = ROOT / "README.md"

# (page file, page title, guide sections it holds)
PAGES: list[tuple[str, str, list[int]]] = [
    ("Value-at-Risk", "Value at Risk and the three models", [1, 2]),
    ("Why-Backtest", "Why backtest, and just enough statistics", [3, 4]),
    ("Kupiec-Test", "Kupiec: is the number of exceptions right?", [5]),
    ("Christoffersen-Tests", "Christoffersen: do exceptions come in bunches?", [6, 7]),
    ("Basel-Traffic-Light", "The Basel traffic light", [8]),
    ("Real-Results", "Results on 10 years of bond ETF data", [9]),
    ("Using-the-Backtest", "How to use the system", [10]),
    ("Pitfalls-Glossary-References", "Pitfalls, glossary, references", [11, 12, 13]),
]


def _wiki_paths(text: str) -> str:
    """Image paths relative to docs/ become paths inside the wiki's images/ folder."""
    return text.replace("img/guide/", "images/").replace("img/var_backtest_portfolio.png", "images/var_backtest_portfolio.png")


def _split(guide: str) -> tuple[str, dict[int, str]]:
    parts = re.split(r"(?m)^(?=## \d+\. )", guide)
    sections = {}
    for p in parts[1:]:
        num = int(re.match(r"## (\d+)\.", p).group(1))  # type: ignore[union-attr]
        sections[num] = p.replace("\n---\n", "\n").rstrip() + "\n"
    return parts[0], sections


def _results_section(section9: str) -> str:
    """Section 9 of the guide, with the setup and the tables from the README inserted."""
    readme = README.read_text(encoding="utf-8")
    block = readme[readme.index("### VaR backtest: results"):readme.index("\n---\n\n# Part II")]
    setup = block[block.index("**Setup, fixed"):block.index("**What the numbers say.**")]
    setup = re.sub(r"!\[[^\]]*\]\(docs/img/var_backtest_portfolio\.png\)\n\n", "", setup)
    body = re.sub(r"The README section \[VaR backtest: results\]\([^)]*\) applies", "The backtest applies", section9)
    head, rest = body.split("\n", 1)
    return f"{head}\n\n{setup.strip()}\n\n### What the chart shows\n{rest}"


def build(wiki: Path) -> None:
    guide = GUIDE.read_text(encoding="utf-8")
    intro, sections = _split(guide)
    sections[9] = _results_section(sections[9])

    for i, (name, _title, nums) in enumerate(PAGES):
        text = "".join(sections[n] + "\n" for n in nums).rstrip() + "\n"
        nav = []
        if i > 0:
            nav.append(f"Previous: [{PAGES[i - 1][1]}]({PAGES[i - 1][0]})")
        if i < len(PAGES) - 1:
            nav.append(f"Next: [{PAGES[i + 1][1]}]({PAGES[i + 1][0]})")
        text += "\n---\n\n" + " | ".join(nav) + "\n"
        (wiki / f"{name}.md").write_text(_wiki_paths(text), encoding="utf-8")

    home = intro.split("**Contents**")[0].replace("# VaR backtesting, from zero", "# MacroQuant wiki: VaR backtesting, from zero")
    home = home.rstrip() + "\n\n## Reading path\n\n"
    home += ("Read the pages in order: each one builds on the previous. If you only have ten minutes, read "
             "[Value at Risk](Value-at-Risk), then jump to [Results](Real-Results) and come back to the tests "
             "when a column of the tables is unclear.\n\n")
    home += "| # | Page | Covers | Sections |\n|---|---|---|---|\n"
    for i, (name, title, nums) in enumerate(PAGES, 1):
        covers = ", ".join(re.match(r"## \d+\. (.+)", sections[n].split("\n")[0]).group(1) for n in nums)  # type: ignore[union-attr]
        span = f"{nums[0]}-{nums[-1]}" if len(nums) > 1 else str(nums[0])
        home += f"| {i} | [{title}]({name}) | {covers} | {span} |\n"
    home += ('\nSection numbers are kept across pages, so "section 5.4" always means the same place.\n\n'
             "The code lives in `macro_quant/analytics/backtest.py`; the MCP tool is `backtest_var`.\n")
    (wiki / "Home.md").write_text(home, encoding="utf-8")

    sidebar = "**[Home](Home)**\n\n" + "\n".join(f"{i}. [{t}]({n})" for i, (n, t, _) in enumerate(PAGES, 1)) + "\n"
    (wiki / "_Sidebar.md").write_text(sidebar, encoding="utf-8")
    (wiki / "_Footer.md").write_text(
        "Source of truth: `docs/var-backtest-guide.md` in the repository, rendered here by `scripts/build_wiki.py`. "
        "Every number on these pages was produced by the project's code.\n", encoding="utf-8")

    images = wiki / "images"
    images.mkdir(exist_ok=True)
    for svg in (ROOT / "docs" / "img" / "guide").glob("*.svg"):
        shutil.copy2(svg, images / svg.name)
    shutil.copy2(ROOT / "docs" / "img" / "var_backtest_portfolio.png", images / "var_backtest_portfolio.png")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not (Path(sys.argv[1]) / ".git").exists():
        sys.exit("usage: python scripts/build_wiki.py <path to a clone of the .wiki.git repository>")
    build(Path(sys.argv[1]))
    print(f"Wiki pages written to {sys.argv[1]}")
