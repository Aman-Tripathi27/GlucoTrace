import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch

from glucofm import probes
from glucofm.cli import COMMANDS
from glucofm.cli import main as cli_main
from glucofm.data import MG_DL_PER_MMOL_L, load_cgm_csv
from glucofm.inference import ResearchEncoder
from glucofm.pretrain import (
    LatentPretrainer,
    PretrainingConfig,
    gradient_reversal,
    symmetric_contrastive_loss,
)
from glucofm.report import build_report, main as report_main
from test_inference import make_checkpoint, make_manifest, write_day
from test_pretrain import small_encoder


def write_csv(path: Path, values: list[float]) -> None:
    start = datetime(2026, 1, 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "glucose"))
        for index, value in enumerate(values):
            writer.writerow(
                ((start + timedelta(minutes=5 * index)).isoformat(), value)
            )


def test_mmol_file_is_rejected_as_mg_dl_and_converted_when_declared(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mmol.csv"
    write_csv(path, [5.5, 6.0, 6.5, 7.0])
    with pytest.raises(ValueError, match="looks like mmol/L"):
        load_cgm_csv(path)
    series = load_cgm_csv(path, unit="mmol/L")
    assert series.glucose[0].item() == pytest.approx(5.5 * MG_DL_PER_MMOL_L)


def test_mg_dl_file_is_rejected_when_declared_mmol(tmp_path: Path) -> None:
    path = tmp_path / "mgdl.csv"
    write_csv(path, [100, 110, 120])
    with pytest.raises(ValueError, match="looks like mg/dL"):
        load_cgm_csv(path, unit="mmol/L")
    with pytest.raises(ValueError, match="unit must be"):
        load_cgm_csv(path, unit="mg")


def test_within_source_negatives_ignore_other_sources() -> None:
    torch.manual_seed(0)
    first = torch.randn(4, 8)
    second = torch.randn(4, 8)
    labels = torch.tensor([0, 0, 1, 1])
    restricted = symmetric_contrastive_loss(
        first, second, temperature=0.2, source_labels=labels
    )
    # Equal-sized sources: the restricted loss is the mean of per-source losses,
    # so other sources' days never act as negatives.
    per_source = 0.5 * (
        symmetric_contrastive_loss(first[:2], second[:2], temperature=0.2)
        + symmetric_contrastive_loss(first[2:], second[2:], temperature=0.2)
    )
    assert float(restricted) == pytest.approx(float(per_source))
    unrestricted = symmetric_contrastive_loss(first, second, temperature=0.2)
    assert float(restricted) < float(unrestricted)


def test_gradient_reversal_negates_gradient() -> None:
    value = torch.tensor([2.0], requires_grad=True)
    (3.0 * gradient_reversal(value)).sum().backward()
    assert value.grad.item() == pytest.approx(-3.0)


def test_source_adversary_trains_and_requires_labels() -> None:
    torch.manual_seed(1)
    config = PretrainingConfig(
        within_source_negatives=True, source_adversary_weight=1.0
    )
    pretrainer = LatentPretrainer(small_encoder(), config, num_sources=2)
    glucose = 100 + 20 * torch.rand(4, 24)
    mask = torch.ones(4, 24, dtype=torch.bool)
    gap = torch.zeros(4, 24)
    tod = torch.zeros(4, 24, 2)
    with pytest.raises(ValueError, match="source_labels"):
        pretrainer(glucose, mask, gap, tod)
    output = pretrainer(
        glucose, mask, gap, tod, source_labels=torch.tensor([0, 1, 0, 1])
    )
    output["loss"].backward()
    assert output["source_adversary_loss"] > 0
    adversary_grad = pretrainer.source_adversary[0].weight.grad
    assert adversary_grad is not None and adversary_grad.abs().sum() > 0
    with pytest.raises(ValueError, match="two sources"):
        LatentPretrainer(small_encoder(), config, num_sources=1)


def test_linear_source_probe_detects_and_misses_source() -> None:
    torch.manual_seed(2)
    sources = ["a"] * 20 + ["b"] * 20
    separable = torch.randn(40, 4)
    separable[20:, 0] += 6.0
    result = probes.linear_source_probe(separable, sources, separable, sources)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    noise_train = torch.randn(40, 4)
    noise_test = torch.randn(40, 4)
    result = probes.linear_source_probe(noise_train, sources, noise_test, sources)
    assert result["balanced_accuracy"] < 0.8


def test_same_participant_retrieval_and_chance() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, -1.0]]
    )
    participants = [("s", "p1"), ("s", "p1"), ("s", "p2"), ("s", "p2"), ("s", "p3")]
    result = probes.same_participant_retrieval(features, features, participants)
    assert result["eligible_days"] == 4
    assert result["top1_same_participant"] == pytest.approx(1.0)
    assert result["chance_top1_same_participant"] == pytest.approx(0.25)


def test_hidden_window_never_exposes_hidden_values() -> None:
    glucose = torch.arange(1, 13, dtype=torch.float32).repeat(2, 1)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 8:] = False
    visible, visible_mask, target, eligible = probes.hide_final_window(
        glucose, mask, positions=4
    )
    assert not visible_mask[:, 8:].any()
    assert torch.equal(visible[:, 8:], torch.zeros(2, 4))
    assert target[0].item() == pytest.approx(10.5)
    assert eligible.tolist() == [True, False]
    persistence = probes.last_visible_hour_mean(visible, visible_mask, positions=2)
    assert persistence[0].item() == pytest.approx(7.5)


def test_ridge_regression_recovers_linear_signal() -> None:
    torch.manual_seed(4)
    features = torch.randn(200, 3)
    targets = 100 + 10 * features[:, 0]
    result = probes.ridge_regression(
        features, targets, features, targets, penalty=1e-3
    )
    assert result["r_squared"] > 0.999


def test_report_is_self_contained_and_shows_gaps(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path / "corpus")
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    query = tmp_path / "query.csv"
    write_day(query, start=datetime(2026, 2, 1), offset=3.0)
    document = build_report(
        ResearchEncoder.load(checkpoint), query, [manifest], top_k=3
    )
    assert document.startswith("<!doctype html>")
    assert "Research software only" in document
    assert document.count("<svg") == 1 + 3
    assert "<script" not in document and "http" not in document

    output = tmp_path / "out.html"
    argv = sys.argv
    sys.argv = [
        "glucofm-report", str(query), "--manifest", str(manifest),
        "--checkpoint", str(checkpoint), "--output", str(output), "--top-k", "2",
    ]
    try:
        report_main()
    finally:
        sys.argv = argv
    assert output.read_text(encoding="utf-8").count("<svg") == 3


def test_cli_lists_and_dispatches(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main([]) == 0
    listing = capsys.readouterr().out
    assert all(name in listing for name in COMMANDS)
    assert cli_main(["nope"]) == 2
    assert cli_main(["--version"]) == 0
    from glucofm import __version__

    assert f"glucotrace {__version__}" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        cli_main(["encode", "--help"])
    assert exit_info.value.code == 0
    assert cli_main(["encode", "missing.csv", "--checkpoint", "missing.pt"]) == 1
    assert "download-model" in capsys.readouterr().err


def test_glucotrace_alias_exposes_public_api() -> None:
    import glucotrace

    assert glucotrace.GlucoTrace is glucotrace.GlucoFM
    import glucofm

    assert glucotrace.__version__ == glucofm.__version__


def test_checkpoint_resolution_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from glucofm import inference

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GLUCOTRACE_CHECKPOINT", raising=False)
    monkeypatch.setenv("GLUCOTRACE_HOME", str(tmp_path / "home"))
    assert inference.default_checkpoint_path() == tmp_path / "home" / "glucofm-research.pt"
    local = tmp_path / "checkpoints" / "glucofm-research.pt"
    local.parent.mkdir()
    local.write_bytes(b"x")
    assert inference.default_checkpoint_path() == Path("checkpoints/glucofm-research.pt")
    monkeypatch.setenv("GLUCOTRACE_CHECKPOINT", "/elsewhere/model.pt")
    assert inference.default_checkpoint_path() == Path("/elsewhere/model.pt")
    with pytest.raises(FileNotFoundError, match="download-model"):
        ResearchEncoder.load(tmp_path / "missing.pt")


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk


def test_download_verifies_checksum_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from glucofm import inference

    payload = b"model-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        inference.urllib.request, "urlopen", lambda url, timeout: _FakeResponse(payload)
    )
    target = tmp_path / "cache" / "model.pt"
    with pytest.raises(ValueError, match="does not match"):
        inference.download_checkpoint(
            target, url="https://example.test/m.pt", expected_sha256="0" * 64
        )
    assert not target.exists()
    assert list(target.parent.iterdir()) == []
    path = inference.download_checkpoint(
        target, url="https://example.test/m.pt", expected_sha256=digest
    )
    assert path.read_bytes() == payload
    with pytest.raises(ValueError, match="https"):
        inference.download_checkpoint(
            tmp_path / "x.pt", url="http://example.test/m.pt", force=True
        )
