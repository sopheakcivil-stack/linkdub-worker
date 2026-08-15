from linkdub.main import failure_update_fields


def test_failure_is_requeued_when_attempts_remain():
    fields = failure_update_fields(
        {"attempts": 1, "max_attempts": 3},
        "temporary failure",
    )

    assert fields == {
        "status": "queued",
        "stage": "Retry scheduled (1/3)",
        "progress": 0,
        "error": "temporary failure",
    }


def test_failure_is_terminal_after_last_attempt():
    fields = failure_update_fields(
        {"attempts": 3, "max_attempts": 3},
        "permanent failure",
    )

    assert fields == {
        "status": "failed",
        "stage": "Failed",
        "error": "permanent failure",
    }

