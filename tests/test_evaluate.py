import torch

from glucofm.evaluate import (
    MissingnessScenario,
    embedding_diagnostics,
    perturb_missingness,
    predeclared_engineering_checks,
    robustness_metrics,
    source_separability,
    summary_baseline,
)


def test_missingness_perturbation_is_deterministic_and_mask_only() -> None:
    glucose = torch.arange(48, dtype=torch.float32).reshape(2, 24)
    observed = torch.ones(2, 24, dtype=torch.bool)
    observed[0, 3] = False
    scenario = MissingnessScenario("test_random", "random", 0.30, 2)

    first = perturb_missingness(
        glucose, observed, scenario, repeat=0, seed=7
    )
    second = perturb_missingness(
        glucose, observed, scenario, repeat=0, seed=7
    )
    corrupted, mask, removed_fraction = first

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert not (mask & ~observed).any()
    assert torch.equal(corrupted[~mask], torch.zeros_like(corrupted[~mask]))
    assert (removed_fraction > 0).all()


def test_block_and_cadence_perturbations_remove_expected_positions() -> None:
    glucose = torch.ones(1, 24)
    observed = torch.ones(1, 24, dtype=torch.bool)
    block = MissingnessScenario("test_block", "block", 6, 1)
    cadence = MissingnessScenario("test_cadence", "cadence", 3, 3)

    _, block_mask, _ = perturb_missingness(
        glucose, observed, block, repeat=0, seed=7
    )
    _, cadence_mask, _ = perturb_missingness(
        glucose, observed, cadence, repeat=1, seed=7
    )

    assert block_mask.sum() == 18
    assert cadence_mask.sum() == 8
    assert cadence_mask[0, 1::3].all()


def test_summary_baseline_is_finite_with_missing_data() -> None:
    glucose = torch.tensor(
        [[100.0, 0.0, 110.0, 120.0], [90.0, 95.0, 0.0, 0.0]]
    )
    observed = glucose != 0

    features = summary_baseline(glucose, observed)

    assert features.shape == (2, 11)
    assert torch.isfinite(features).all()
    assert torch.equal(features[:, 9], torch.tensor([0.75, 0.5]))


def test_embedding_diagnostics_exposes_collapsed_features() -> None:
    collapsed = torch.ones(8, 6)
    varied = torch.eye(8)

    collapsed_report = embedding_diagnostics(collapsed)
    varied_report = embedding_diagnostics(varied)

    assert collapsed_report["mean_feature_std"] == 0.0
    assert collapsed_report["effective_rank"] == 1.0
    assert varied_report["effective_rank"] > collapsed_report["effective_rank"]


def test_source_centroid_probe_reports_separable_sources() -> None:
    validation = torch.tensor(
        [[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]]
    )
    test = torch.tensor([[-1.5, -1.5], [1.5, 1.5]])
    validation_sources = ["a", "a", "b", "b"]
    test_sources = ["a", "b"]

    report = source_separability(
        validation, validation_sources, test, test_sources
    )

    assert report["accuracy"] == 1.0
    assert report["balanced_accuracy"] == 1.0


def test_identical_vectors_have_perfect_robustness_metrics() -> None:
    clean = torch.eye(6)

    report = robustness_metrics(clean, [clean.clone(), clean.clone()])

    assert report["mean_cosine_similarity"] == 1.0
    assert report["median_relative_l2_drift"] == 0.0
    assert report["top1_neighbor_agreement"] == 1.0
    assert report["top5_neighbor_overlap"] == 1.0


def test_predeclared_checks_use_frozen_metric_schema() -> None:
    diagnostics = {"mean_feature_std": 0.10, "effective_rank": 9.0}
    stable = {
        "median_cosine_similarity": 0.90,
        "p10_cosine_similarity": 0.60,
        "top5_neighbor_overlap": 0.70,
    }
    missingness = {
        name: {"model": dict(stable)}
        for name in (
            "random_30_percent",
            "contiguous_60_minutes",
            "cadence_15_minutes",
        )
    }

    report = predeclared_engineering_checks(diagnostics, missingness)

    assert report["all_passed"] is True
    assert len(report["checks"]) == 5
    assert (
        report["checks"]["stability_random_30_percent"]["values"]
        ["top5_neighbor_overlap"]
        == 0.70
    )
