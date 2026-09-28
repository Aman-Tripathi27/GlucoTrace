"""Published protocol files must match their declared SHA-256 values."""

import shutil
from pathlib import Path

import pytest

from glucofm.protocols import PROTOCOL_FILES, verify_protocol

REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "processed"


@pytest.mark.parametrize("version", sorted(PROTOCOL_FILES))
def test_published_protocol_files_match_declarations(version: str) -> None:
    report = verify_protocol(version, REPO_DATA)
    assert report["passed"], report["files"]


def test_tampered_or_missing_files_fail(tmp_path: Path) -> None:
    for relative in PROTOCOL_FILES["1.3"]["files"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_DATA / relative, target)
    split = tmp_path / "colas_midnight" / "splits_protocol_1_3.json"
    split.write_text(split.read_text(encoding="utf-8") + " ", encoding="utf-8")
    (tmp_path / "big_ideas_midnight" / "manifest.json").unlink()

    report = verify_protocol("1.3", tmp_path)
    statuses = {item["file"]: item["status"] for item in report["files"]}
    assert not report["passed"]
    assert statuses["colas_midnight/splits_protocol_1_3.json"] == "MISMATCH"
    assert statuses["big_ideas_midnight/manifest.json"] == "missing"
    assert statuses["colas_midnight/manifest.json"] == "ok"


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown protocol"):
        verify_protocol("9.9", REPO_DATA)
