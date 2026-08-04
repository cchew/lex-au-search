import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from colab_driver import _parse_verify_output, _parse_pid_output, _parse_poll_output


def test_parse_verify_output_true_on_success_marker():
    assert _parse_verify_output(0, "CUDA available, providers: [...]\nCUDA_VERIFY_OK\n") is True


def test_parse_verify_output_false_on_nonzero_exit():
    assert _parse_verify_output(1, "CUDA_VERIFY_OK") is False


def test_parse_verify_output_false_when_marker_missing():
    assert _parse_verify_output(0, "AssertionError: CUDA not available") is False


def test_parse_pid_output_extracts_pid():
    assert _parse_pid_output("PID 12345\n") == 12345


def test_parse_pid_output_extracts_from_noisy_output():
    assert _parse_pid_output("some banner text\nPID 987\nmore text\n") == 987


def test_parse_pid_output_raises_when_no_pid_present():
    import pytest
    with pytest.raises(ValueError):
        _parse_pid_output("no pid here")


def test_parse_poll_output_running_when_alive_and_not_done():
    assert _parse_poll_output("ALIVE PENDING \n") == "running"


def test_parse_poll_output_done_on_zero_exit_code():
    assert _parse_poll_output("DEAD DONE 0\n") == "done"


def test_parse_poll_output_failed_on_nonzero_exit_code():
    assert _parse_poll_output("DEAD DONE 1\n") == "failed"


def test_parse_poll_output_failed_when_process_gone_without_exit_code():
    assert _parse_poll_output("DEAD PENDING \n") == "failed"
