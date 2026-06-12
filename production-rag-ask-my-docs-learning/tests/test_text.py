from ask_my_docs.text import ensure_line_citations, extract_citation_numbers, sliding_word_chunks


def test_sliding_chunks_overlap() -> None:
    text = "one two three four five six seven eight"
    chunks = sliding_word_chunks(text=text, chunk_size=4, overlap=2)
    assert chunks == ["one two three four", "three four five six", "five six seven eight"]


def test_ensure_line_citations() -> None:
    lines = ["- fact one", "- fact two [2]"]
    output = ensure_line_citations(lines, default_citation=1)
    assert output[0].endswith("[1]")
    assert output[1].endswith("[2]")


def test_extract_citation_numbers() -> None:
    answer = "line [1]\nline [3][4]"
    assert extract_citation_numbers(answer) == [1, 3, 4]
