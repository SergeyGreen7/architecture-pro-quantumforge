#!/usr/bin/env python3
"""Create a coherent pseudonymized copy of the Rick and Morty text corpus."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path


# Hand-authored mappings define the vocabulary and preserve the logic of the
# fictional world. Longer phrases are replaced before shorter roots.
EXPLICIT_TERMS: dict[str, str] = {
    # Universe and principal characters
    "Рик и Морти": "Аркон и Лио",
    "Rick and Morty": "Arkon and Lio",
    "Рик Санчез": "Аркон Вейр",
    "Морти Смит": "Лио Вейл",
    "Бет Смит": "Нэйра Вейл",
    "Саммер Смит": "Сайра Вейл",
    "Джерри Смит": "Гарен Вейл",
    "Дайен Санчез": "Дайна Вейр",
    "Злой Морти": "Тёмный Лио",
    "Злой Рик Санчез": "Тёмный Аркон Вейр",
    "Рик Прайм": "Аркон Прайм",
    "Прайм Рик": "Прайм Аркон",
    "Мистер Мисикс": "Куратор Эйдос",
    "Мистер Жопосранчик": "Господин Лучезар",
    "Птичья личность": "Авиан Нокс",
    "Сквончи": "Кваркс",
    "Юнити": "Конкорда",
    "Джессика": "Элара",
    "Снафлс": "Криос",
    "Снежок": "Криос",
    "Бета-Семь": "Пара-Нокс",
    "The Flesh Curtains": "Багровый Занавес",
    "MC Хапс": "MC Кайро",
    # Institutions, groups, and locations
    "Галактическая Федерация": "Орбитальная Коалиция",
    "Президент Галактической Федерации": "Консул Орбитальной Коалиции",
    "Цитадель Риков": "Бастион Арконов",
    "Совет Риков": "Собрание Арконов",
    "Теневой Совет Риков": "Теневой Конклав Арконов",
    "Виндикаторы": "Стражи Предела",
    "Заступники": "Стражи Предела",
    "Сталкеры Смерти": "Кочевники Пепла",
    "Центральная конечная кривая": "Осевой предел",
    # Species and races
    "Газорпианцы": "Талорианцы",
    "Громфломиты": "Зерониды",
    "Кромулонцы": "Астракроны",
    "Зигерионцы": "Зигариды",
    "Плутонианцы": "Крионцы",
    "Кроненберги": "Мутагениды",
    "Корблокианцы": "Корвалианцы",
    "Гарбловиане": "Гарволиды",
    "Древолюди": "Арбориане",
    "Инопланетные паразиты": "Межмировые мимикры",
    # Planets, dimensions, and worlds
    "Громфлом Прайм": "Зерон Прайм",
    "Газорпазорп": "Талорис",
    "Плутон": "Крион",
    "Птичий мир": "Авиара",
    "Мир Кроненбергов": "Мир Мутагенидов",
    "Измерение C-137": "Континуум K-42",
    "Измерение C-131": "Континуум K-17",
    "C-137": "K-42",
    "C-131": "K-17",
    "C-500A": "K-500V",
    # Technology, weapons, and artifacts
    "Портальная пушка": "Фазовая пушка",
    "портальная пушка": "фазовая пушка",
    "Портальная жидкость": "Фазовый раствор",
    "Коробка с Мисиксами": "Куб Эйдоса",
    "Микровселенный аккумулятор": "Карманный реактор",
    "Микровселенная": "Карманный космос",
    "Плюмбус": "Ворпус",
    "Брейналайзер": "Нейроскоп",
    "Замораживающий луч": "Криоизлучатель",
    "Уменьшающий луч": "Миниатюризатор",
    "Увеличивающий луч": "Макроизлучатель",
    "Мегасемена": "Кванто-зёрна",
    "Крис (оружие)": "Клинок Ардис",
    # Russian inflections and recurring roots
    "Риками": "Арконами",
    "Рикам": "Арконам",
    "Риках": "Арконах",
    "Риков": "Арконов",
    "Рики": "Арконы",
    "Джессикой": "Эларой",
    "Джессику": "Элару",
    "Джессике": "Эларе",
    "Джессики": "Элары",
    "Мисикс": "Эйдос",
    "Кроненберг": "Мутаген",
    "Сквонч": "Кваркс",
    "Морти": "Лио",
    "Санчез": "Вейр",
    "Смит": "Вейл",
    "Саммер": "Сайра",
    "Джерри": "Гарен",
    "Джессик": "Элар",
    "Дайен": "Дайна",
    "Жопосранчик": "Лучезар",
    "Плюмбус": "Ворпус",
    "Рик": "Аркон",
    "Бет": "Нэйра",
    # English aliases and spelling variants found in article prose
    "SEAL Team Ricks": "Aegis Team Arkons",
    "SWAT Ricks": "Aegis Arkons",
    "Rick Sanchez": "Arkon Veir",
    "Morty Smith": "Lio Veil",
    "Galactic Federation": "Orbital Coalition",
    "Mr. Meeseeks": "Curator Eidos",
    "Birdperson": "Avian Nox",
    "Mr. Poopybutthole": "Mister Lumen",
    "Meeseeks": "Eidos",
    "Sanchez": "Veir",
    "Morty": "Lio",
    "Smith": "Veil",
    "Ricks": "Arkons",
    "Rick": "Arkon",
    "Санчес": "Вейр",
    "Скванч": "Кваркс",
    "Громфломит": "Зеронид",
    "Галактическ": "Орбитальн",
    "Федераци": "Коалици",
    "Соларикс": "Гелиосдвиг",
    "Солярикс": "Гелиосдвиг",
    "Возврикщение": "Возвращение",
    "Мортпех": "Лиопех",
    "Виндикаторами": "Стражами",
    "Виндикаторов": "Стражей",
    "Виндикаторы": "Стражи",
    "Злого Морти": "Тёмного Лио",
    "Злому Морти": "Тёмному Лио",
    "Злым Морти": "Тёмным Лио",
    "Злого Рика": "Тёмного Аркона",
    "Злому Рику": "Тёмному Аркону",
    "Злым Риком": "Тёмным Арконом",
    "Злой Рик": "Тёмный Аркон",
    "безриков": "безарконов",
    "Близкие риконтакты риковой степени": "Схождение Арконов",
    "Сказки из Цитадели": "Хроники Бастиона",
    "Пора Швифтануться": "Время резонанса",
    "Аутоэротическая ассимиляция": "Протокол Единства",
    "Огурчик Рик": "Капсульный Аркон",
    "Джастин Ройланд": "Илар Восс",
    "Дэн Хармон": "Кейл Оррен",
    "Ройланд": "Восс",
    "Хармон": "Оррен",
    "Бетани": "Нэйрана",
    "Даен": "Дайна",
    "С-137": "K-42",
    "С-131": "K-17",
    "С-500A": "K-500V",
}


STOP_TOKENS = {
    "the", "mc", "и", "в", "с", "из", "как", "ранее", "известный",
    "единственный", "истинный", "злой", "токсичный", "мелкий", "ковбой",
    "мистер", "миссис", "доктор", "директор", "президент", "король", "леди",
    "клон", "ученый", "оружие", "говорящий", "кот", "девочка", "подруга",
    "совет", "теневой", "полиция", "времени", "глаза", "муравьиные", "миллион",
    "муравьев", "звездный", "солдат", "ренегад", "пришелец", "кальяном",
    "инопланетные", "паразиты", "альфа", "зета", "квантум", "прайм", "семь",
}

SYLLABLE_A = (
    "Аэ", "Ве", "Кай", "Дар", "Зе", "Ир", "Кор", "Лай", "Мер", "Нор",
    "Ор", "Пар", "Рей", "Сар", "Тал", "Ур", "Фей", "Хал", "Цер", "Эл",
)
SYLLABLE_B = (
    "лар", "вис", "кор", "ден", "зар", "ис", "кар", "лор", "мир", "нор",
    "ор", "пис", "ран", "сир", "тар", "ус", "фир", "хар", "цис", "эн",
)


def synthetic_token(index: int) -> str:
    return SYLLABLE_A[index % len(SYLLABLE_A)] + SYLLABLE_B[(index // len(SYLLABLE_A)) % len(SYLLABLE_B)]


def term_pattern(term: str, allow_suffix: bool) -> re.Pattern[str]:
    # Word-start matching handles Russian inflections but avoids corrupting
    # ordinary words such as «приключение» when replacing «Рик».
    ending = "" if allow_suffix else r"(?![A-Za-zА-Яа-яЁё])"
    return re.compile(
        r"(?<![A-Za-zА-Яа-яЁё])" + re.escape(term) + ending,
        re.IGNORECASE,
    )


def preserve_initial_case(source: str, replacement: str) -> str:
    if source and source[0].islower() and replacement:
        return replacement[0].lower() + replacement[1:]
    return replacement


def replace_terms(text: str, terms: dict[str, str], prefix_terms: set[str] | None = None) -> str:
    prefix_terms = prefix_terms or set()
    for source, target in sorted(terms.items(), key=lambda item: len(item[0]), reverse=True):
        text = term_pattern(source, source in prefix_terms).sub(
            lambda match, value=target: preserve_initial_case(match.group(0), value), text
        )
    return text


def title_from_file(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        return handle.readline().strip() or path.stem


def categories_from_file(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Категории:"):
                return {item.strip() for item in line.partition(":")[2].split(",") if item.strip()}
    return set()


def build_dynamic_maps(files: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    titles = [title_from_file(path) for path in files]
    explicit_sources = {token.casefold() for phrase in EXPLICIT_TERMS for token in re.findall(r"[A-Za-zА-Яа-яЁё]+", phrase)}
    candidate_tokens: set[str] = set()
    for title in titles:
        for token in re.findall(r"[A-Za-zА-Яа-яЁё]+", title):
            folded = token.casefold()
            if len(token) >= 4 and folded not in STOP_TOKENS and folded not in explicit_sources:
                candidate_tokens.add(token)

    token_map: dict[str, str] = {}
    for index, token in enumerate(sorted(candidate_tokens, key=str.casefold)):
        token_map[token] = synthetic_token(index)

    generation_terms = dict(EXPLICIT_TERMS)
    generation_terms.update(token_map)
    title_map: dict[str, str] = {}
    used_targets: set[str] = set()
    for index, path in enumerate(files):
        source_title = title_from_file(path)
        target_title = replace_terms(source_title, generation_terms, set(EXPLICIT_TERMS))
        if target_title.casefold() == source_title.casefold():
            categories = categories_from_file(path)
            label = (
                "Народ" if "Расы" in categories
                else "Союз" if "Группы" in categories
                else "Существо" if "Животные" in categories
                else "Персона"
            )
            target_title = f"{label} {synthetic_token(index + len(token_map))}"
        base_target = target_title
        suffix = 2
        while target_title.casefold() in used_targets:
            target_title = f"{base_target} {suffix}"
            suffix += 1
        used_targets.add(target_title.casefold())
        title_map[source_title] = target_title
    return title_map, token_map


def safe_filename(title: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", title)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return f"{name}.txt"


def strip_source_aliases(text: str) -> str:
    # Original English aliases reveal the source universe even after Russian
    # names have been replaced. They are metadata rather than article prose.
    text = re.sub(r"\s*\(англ\.[^()]*(?:\([^()]*\)[^()]*)*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text


def validate(output_dir: Path, expected_count: int, terms: dict[str, str]) -> list[str]:
    errors: list[str] = []
    files = sorted(output_dir.glob("*.txt"))
    if len(files) != expected_count:
        errors.append(f"Expected {expected_count} TXT files, found {len(files)}")
    if not (output_dir / "terms_map.json").exists():
        errors.append("terms_map.json is missing")

    iconic = re.compile(
        r"(?<![A-Za-zА-Яа-яЁё])(?:Рик|Морти|Санчез|Смит|Мисикс|"
        r"Сквонч|Плюмбус|Кроненберг|Rick|Morty|Sanchez|Smith|Meeseeks|Plumbus)",
        re.IGNORECASE,
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        if len(text.split()) < 15:
            errors.append(f"Document too short: {path.name}")
        match = iconic.search(text)
        if match:
            errors.append(f"Iconic source term {match.group(0)!r} remains in {path.name}")
        if re.search(r"\(англ\.", text, flags=re.IGNORECASE):
            errors.append(f"English source alias remains in {path.name}")
        if re.search(r"</?[a-z][^>]*>", text, flags=re.IGNORECASE):
            errors.append(f"HTML remains in {path.name}")
    return errors


def build(source_dir: Path, output_dir: Path) -> int:
    source_files = sorted(source_dir.glob("*.txt"), key=lambda path: path.name.casefold())
    if not source_files:
        raise RuntimeError(f"No TXT files found in {source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    title_map, token_map = build_dynamic_maps(source_files)
    terms = dict(title_map)
    terms.update(EXPLICIT_TERMS)
    terms.update(token_map)
    dimension_terms: dict[str, str] = {}
    for path in source_files:
        source_text = path.read_text(encoding="utf-8")
        for dimension_id in re.findall(r"(?<![A-Za-zА-Яа-яЁё])[CС]-(\d+[A-ZА-Я]?)", source_text):
            latin_source = f"C-{dimension_id}"
            cyrillic_source = f"С-{dimension_id}"
            target = f"K-{dimension_id}"
            dimension_terms.setdefault(latin_source, target)
            dimension_terms.setdefault(cyrillic_source, target)
    # Preserve the deliberately chosen aliases for the main continuums.
    dimension_terms.update({key: value for key, value in EXPLICIT_TERMS.items() if re.fullmatch(r"[CС]-\w+", key)})
    terms.update(dimension_terms)

    filenames: Counter[str] = Counter()
    for path in source_files:
        source_title = title_from_file(path)
        target_title = title_map[source_title]
        text = strip_source_aliases(path.read_text(encoding="utf-8"))
        text = replace_terms(text, terms, set(EXPLICIT_TERMS))
        filename = safe_filename(target_title)
        filenames[filename.casefold()] += 1
        if filenames[filename.casefold()] > 1:
            raise RuntimeError(f"Generated filename collision: {filename}")
        (output_dir / filename).write_text(text, encoding="utf-8")

    (output_dir / "terms_map.json").write_text(
        json.dumps(terms, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = validate(output_dir, len(source_files), terms)
    summary = {
        "source_documents": len(source_files),
        "output_documents": len(list(output_dir.glob("*.txt"))),
        "mapping_entries": len(terms),
        "validation_errors": len(errors),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print("\nValidation errors:", file=sys.stderr)
        for error in errors[:100]:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        return build(args.source_dir.resolve(), args.output_dir.resolve())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
