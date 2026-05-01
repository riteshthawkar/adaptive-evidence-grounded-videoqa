from scripts.run_vlm_answerer import (
    frame_times_for_segment,
    parse_answer_index,
    sanitize_id,
)


def test_parse_answer_index_handles_common_vlm_outputs() -> None:
    assert parse_answer_index("C") == 2
    assert parse_answer_index("Answer: B.") == 1
    assert parse_answer_index("The answer is (E)") == 4
    assert parse_answer_index("option D") == 3


def test_parse_answer_index_falls_back_to_option_text() -> None:
    options = ["opening the door", "sitting down", "running away"]

    assert parse_answer_index("The person is sitting down.", options) == 1


def test_frame_times_for_segment_samples_inside_interval() -> None:
    assert frame_times_for_segment(10.0, 16.0, 3) == [11.0, 13.0, 15.0]
    assert frame_times_for_segment(5.0, 5.0, 3) == [5.0]
    assert frame_times_for_segment(8.0, 2.0, 1) == [2.0 + 3.0]


def test_sanitize_id_keeps_paths_cache_safe() -> None:
    assert sanitize_id("nextgqa:4882821564:1:segment:4") == "nextgqa_4882821564_1_segment_4"
