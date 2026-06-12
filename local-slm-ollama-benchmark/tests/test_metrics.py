from local_slm_ollama_benchmark.metrics import ns_to_sec, percentile, tokens_per_second


def test_ns_to_sec() -> None:
    assert ns_to_sec(2_000_000_000) == 2.0
    assert ns_to_sec(None) is None


def test_tokens_per_second() -> None:
    score = tokens_per_second(eval_count=100, eval_duration_ns=2_000_000_000)
    assert score == 50


def test_percentile_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert round(percentile(values, 95), 4) == 3.85
