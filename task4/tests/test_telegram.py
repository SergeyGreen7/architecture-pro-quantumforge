from quantumforge_bot.telegram import split_message


def test_short_message_is_unchanged() -> None:
    assert list(split_message("короткий ответ", limit=50)) == ["короткий ответ"]


def test_long_message_is_split_without_data_loss() -> None:
    text = "первая строка\n" + "слово " * 20
    parts = list(split_message(text, limit=40))

    assert len(parts) > 1
    assert all(len(part) <= 40 for part in parts)
    assert " ".join(" ".join(parts).split()) == " ".join(text.split())

