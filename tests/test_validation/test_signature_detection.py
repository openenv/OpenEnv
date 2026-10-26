import pytest
from conftest import FIXTURES
from openenv.validation.signature import (
    detect_signature,
    SignatureError,
    WELL_KNOWN_FILES,
)
from openenv.validation.types import SignatureKind


def test_openenv_yaml_detects_the_served_format():
    assert (
        detect_signature(FIXTURES / "served_min_pass") is SignatureKind.OPENENV_SERVED
    )


def test_the_table_lists_exactly_the_implemented_parsers():
    assert set(WELL_KNOWN_FILES) == {SignatureKind.OPENENV_SERVED}


@pytest.mark.parametrize("fixture", ["harbor_task_min", "posttrain_task_min"])
def test_formats_without_parsers_are_unrecognized_not_guessed(fixture):
    with pytest.raises(SignatureError, match="unrecognized"):
        detect_signature(FIXTURES / fixture)


def test_no_well_known_file_is_unrecognized():
    with pytest.raises(SignatureError, match="unrecognized"):
        detect_signature(FIXTURES / "unrecognized_package")


def test_two_well_known_files_are_ambiguous_never_a_guess():
    with pytest.raises(SignatureError, match="ambiguous"):
        detect_signature(FIXTURES / "ambiguous_package")


def test_task_md_without_frontmatter_is_not_a_named_format(tmp_path):
    (tmp_path / "openenv.yaml").write_text("spec_version: 1\nname: x\n")
    (tmp_path / "task.md").write_text("# just a readme\n")
    assert detect_signature(tmp_path) is SignatureKind.OPENENV_SERVED


def test_non_utf8_task_md_is_not_a_named_format(tmp_path):
    (tmp_path / "openenv.yaml").write_text("spec_version: 1\nname: x\n")
    (tmp_path / "task.md").write_bytes(b"\xff\xfe")
    assert detect_signature(tmp_path) is SignatureKind.OPENENV_SERVED


def test_missing_directory_is_a_signature_error(tmp_path):
    with pytest.raises(SignatureError, match="not a package directory"):
        detect_signature(tmp_path / "does_not_exist")
