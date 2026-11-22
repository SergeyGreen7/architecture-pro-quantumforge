#!/usr/bin/env python3
"""Download direct article members of selected Fandom categories as clean TXT."""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


API_URL = "https://rickandmorty.fandom.com/ru/api.php"
USER_AGENT = "QuantumForgeCategoryParser/1.0 (educational project)"
CATEGORIES = (
    "Люди",
    "Альтернативные версии персонажей",
    "Пришельцы",
    "Животные",
    "Расы",
    "Группы",
)


def api_request(params: dict[str, str], retries: int = 5) -> dict:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            if "error" in payload:
                raise RuntimeError(f"MediaWiki API error: {payload['error']}")
            return payload
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Request failed after {retries} attempts: {exc}") from exc
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def category_articles(category: str) -> list[str]:
    titles: list[str] = []
    continuation: str | None = None
    while True:
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": f"Категория:{category}",
            "cmnamespace": "0",
            "cmlimit": "500",
        }
        if continuation:
            params["cmcontinue"] = continuation
        payload = api_request(params)
        titles.extend(member["title"] for member in payload["query"]["categorymembers"])
        continuation = payload.get("continue", {}).get("cmcontinue")
        if not continuation:
            return titles


def parse_article(title: str) -> tuple[str, str]:
    payload = api_request(
        {
            "action": "parse",
            "format": "json",
            "page": title,
            "prop": "text",
            "disabletoc": "1",
            "disableeditsection": "1",
            "redirects": "1",
        }
    )
    parsed = payload["parse"]
    return parsed["title"], parsed["text"]["*"]


def normalize_text(value: str) -> str:
    value = html.unescape(value).replace("\xa0", " ").replace("\u200b", "")
    value = re.sub(r"\[(?:\d+|citation needed|источник\?)\]", "", value, flags=re.I)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


@dataclass
class Block:
    tag: str
    text: str


class CleanArticleParser(HTMLParser):
    """Keep article prose/headings; discard HTML chrome and formatting."""

    kept_tags = {"h2", "h3", "h4", "p"}
    ignored_tags = {
        "table",
        "style",
        "script",
        "figure",
        "gallery",
        "sup",
        "noscript",
        "audio",
        "video",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ignore_depth = 0
        self.current_tag: str | None = None
        self.buffer: list[str] = []
        self.blocks: list[Block] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ignored_tags:
            self.ignore_depth += 1
        if self.ignore_depth:
            return
        if tag in self.kept_tags and self.current_tag is None:
            self.current_tag = tag
            self.buffer = []
        elif tag == "br" and self.current_tag:
            self.buffer.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored_tags and self.ignore_depth:
            self.ignore_depth -= 1
            return
        if self.ignore_depth:
            return
        if tag == self.current_tag:
            text = normalize_text("".join(self.buffer))
            if text:
                self.blocks.append(Block(tag, text))
            self.current_tag = None
            self.buffer = []

    def handle_data(self, data: str) -> None:
        if not self.ignore_depth and self.current_tag:
            self.buffer.append(data)


IGNORED_SECTIONS = {
    "галерея",
    "ссылки",
    "примечания",
    "источники",
    "навигация",
    "смотрите также",
    "внешние ссылки",
    "цитаты",
    "больше",
    "все персонажи",
}


def clean_article(raw_html: str) -> str:
    parser = CleanArticleParser()
    parser.feed(raw_html)
    parser.close()

    selected: list[Block] = []
    skip_section_level: int | None = None
    seen: set[tuple[str, str]] = set()
    for block in parser.blocks:
        level = int(block.tag[1]) if block.tag.startswith("h") else None
        if level is not None:
            if skip_section_level is not None and level <= skip_section_level:
                skip_section_level = None
            normalized_heading = block.text.casefold().strip(": ")
            if normalized_heading in IGNORED_SECTIONS:
                skip_section_level = level
                continue
        elif skip_section_level is not None:
            continue
        if skip_section_level is not None:
            continue
        key = (block.tag, block.text)
        if key in seen:
            continue
        if block.text.startswith("Эта статья ") and len(block.text) < 180:
            continue
        seen.add(key)
        selected.append(block)

    lines: list[str] = []
    for index, block in enumerate(selected):
        if block.tag.startswith("h"):
            current_level = int(block.tag[1])
            has_prose = False
            for following in selected[index + 1 :]:
                if following.tag.startswith("h") and int(following.tag[1]) <= current_level:
                    break
                if following.tag == "p":
                    has_prose = True
                    break
            if not has_prose:
                continue
            lines.extend((block.text, "-" * min(len(block.text), 72), ""))
        else:
            lines.extend((block.text, ""))
    return "\n".join(lines).strip()


def safe_filename(title: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", title)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return f"{name}.txt"


def validate(directory: Path, expected_count: int) -> list[str]:
    errors: list[str] = []
    files = sorted(directory.glob("*.txt"))
    if len(files) != expected_count:
        errors.append(f"Expected {expected_count} TXT files, found {len(files)}")
    for path in files:
        text = path.read_text(encoding="utf-8")
        if len(text.split()) < 15:
            errors.append(f"Too little readable text: {path.name} ({len(text.split())} words)")
        if re.search(r"</?[a-z][^>]*>", text, flags=re.I):
            errors.append(f"HTML tag remains: {path.name}")
        if "javascript:" in text.casefold() or "data:image" in text.casefold():
            errors.append(f"Web artifact remains: {path.name}")
    return errors


def build(output_dir: Path) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    category_results: dict[str, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CATEGORIES)) as executor:
        futures = {executor.submit(category_articles, category): category for category in CATEGORIES}
        for future in concurrent.futures.as_completed(futures):
            category = futures[future]
            titles = future.result()
            category_results[category] = titles
            print(f"category {category}: {len(titles)} articles", file=sys.stderr)

    memberships: dict[str, list[str]] = {}
    for category in CATEGORIES:
        for title in category_results[category]:
            memberships.setdefault(title, []).append(category)

    parsed_articles: dict[str, tuple[str, str]] = {}
    failures: list[str] = []
    titles = sorted(memberships, key=str.casefold)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(parse_article, title): title for title in titles}
        for future in concurrent.futures.as_completed(futures):
            title = futures[future]
            try:
                parsed_articles[title] = future.result()
                print(f"downloaded: {title}", file=sys.stderr)
            except Exception as exc:
                failures.append(f"{title}: {exc}")

    canonical_articles: dict[str, str] = {}
    canonical_categories: dict[str, set[str]] = {}
    for requested_title in titles:
        parsed = parsed_articles.get(requested_title)
        if parsed is None:
            continue
        canonical_title, raw_html = parsed
        canonical_articles.setdefault(canonical_title, raw_html)
        canonical_categories.setdefault(canonical_title, set()).update(memberships[requested_title])

    written: list[Path] = []
    for canonical_title in sorted(canonical_articles, key=str.casefold):
        raw_html = canonical_articles[canonical_title]
        body = clean_article(raw_html)
        if not body:
            failures.append(f"{canonical_title}: no readable article prose extracted")
            continue
        categories = ", ".join(
            category for category in CATEGORIES if category in canonical_categories[canonical_title]
        )
        document = f"{canonical_title}\n{'=' * min(len(canonical_title), 72)}\n\nКатегории: {categories}\n\n{body}\n"
        path = output_dir / safe_filename(canonical_title)
        path.write_text(document, encoding="utf-8")
        written.append(path)

    errors = validate(output_dir, len(canonical_articles))
    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
    if errors:
        print("\nValidation errors:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    summary = {
        "categories": {category: len(category_results[category]) for category in CATEGORIES},
        "category_entries_after_title_deduplication": len(memberships),
        "canonical_entities": len(canonical_articles),
        "redirect_aliases_collapsed": len(memberships) - len(canonical_articles),
        "written_txt_files": len(written),
        "failures": len(failures),
        "validation_errors": len(errors),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures or errors else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        return build(args.output_dir.resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
