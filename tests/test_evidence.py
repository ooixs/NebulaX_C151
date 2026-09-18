from common.evidence import comparison


def test_comparison_marks_relative_outlier_without_fault_claim():
    result = comparison("Example", 10, [1] * 9 + [10])
    assert result["status"] == "outlier_high"
    assert result["average"] == 1.9
    assert "fault" not in result


def test_comparison_handles_single_item_batch():
    result = comparison("Example", 4.2, [4.2], "m/s")
    assert result == {
        "label": "Example",
        "value": 4.2,
        "average": 4.2,
        "unit": "m/s",
        "status": "only",
        "z_score": 0.0,
        "peer_count": 1,
    }
