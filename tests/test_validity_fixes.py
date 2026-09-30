"""Regression tests for the evaluation-validity fixes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eeg_motor_imagery_classification.evaluation import (
    FoldMetrics,
    SubjectResult,
    aggregate_fold_metrics,
    aggregate_transfer_seed_runs,
    holm_correction,
    paired_permutation_test,
    summarize_subject_results,
    summarize_transfer_across_targets,
)

SETTINGS = ["zero_shot", "5_shot", "10_shot", "20_shot", "30_shot"]


def _lateralised_volts(n_trials: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic epochs at the real MOABB scale (~5 uV std, in volts) with lateralised power."""

    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n_trials // 2)
    X = rng.standard_normal((n_trials, 8, 500))
    X[y == 0, 5] *= 0.7
    X[y == 1, 2] *= 0.7
    return (X * 5e-6).astype(np.float32), y


def test_log_variance_features_are_not_clipped_at_volt_scale() -> None:
    from eeg_motor_imagery_classification.features import LogVarianceVectorizer

    X, _ = _lateralised_volts()
    features = LogVarianceVectorizer().transform(X)
    assert np.unique(features).size > 1
    np.testing.assert_allclose(features, np.log(np.var(X.astype(np.float64), axis=-1)))


def test_raw_power_pipeline_learns_lateralised_power_at_volt_scale() -> None:
    from sklearn.model_selection import cross_val_score

    from eeg_motor_imagery_classification.models import build_raw_power_pipeline

    X, y = _lateralised_volts()
    assert cross_val_score(build_raw_power_pipeline(), X, y, cv=5).mean() > 0.9


def test_aggregate_uses_sample_std_and_t_interval() -> None:
    from scipy import stats

    values = np.array([0.6, 0.7, 0.8, 0.9])
    summary = aggregate_fold_metrics([FoldMetrics(v, v, v, np.eye(2)) for v in values])
    assert summary["accuracy_std"] == pytest.approx(values.std(ddof=1))
    half = stats.t.ppf(0.975, 3) * values.std(ddof=1) / 2.0
    assert summary["accuracy_ci95_high"] == pytest.approx(values.mean() + half)


def test_exact_sign_flip_floor_for_nine_units() -> None:
    diffs = np.array([0.08, 0.12, 0.03, 0.10, 0.06, 0.09, 0.11, 0.05, 0.07])
    result = paired_permutation_test(diffs, np.zeros(9))
    assert result["exact"] is True
    assert result["p_value"] == pytest.approx(2 / 2**9)


def test_holm_correction_matches_reference() -> None:
    adjusted = holm_correction([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])


def _transfer_seed_run(offset: float) -> dict[str, object]:
    targets = {}
    for subject in range(1, 10):
        rows = [
            SubjectResult(setting, FoldMetrics(a, a, a, np.array([[36, 36], [0, 72]])))
            for setting, a in zip(SETTINGS, 0.5 + 0.04 * subject + offset + 0.01 * np.arange(5), strict=True)
        ]
        targets[f"S{subject}"] = summarize_subject_results(rows)
    placeholder = summarize_subject_results([SubjectResult(s, FoldMetrics(0, 0, 0, np.zeros((2, 2)))) for s in SETTINGS])
    return {"aggregate_by_setting": placeholder, "targets": targets}


def test_transfer_summary_uses_targets_as_units() -> None:
    repeated = aggregate_transfer_seed_runs({"seed_42": _transfer_seed_run(0.0), "seed_43": _transfer_seed_run(0.02)})
    per_setting = summarize_transfer_across_targets(repeated)
    assert set(per_setting) == set(SETTINGS)
    zero = per_setting["zero_shot"]
    assert zero["summary"]["n"] == 9
    expected = 0.5 + 0.04 * np.arange(1, 10) + 0.01  # seed-averaged per-target accuracy
    assert zero["summary"]["accuracy_mean"] == pytest.approx(expected.mean())
    assert zero["summary"]["accuracy_std"] == pytest.approx(expected.std(ddof=1))
    # Two seeds x 144 evaluation trials per target
    assert int(np.sum(zero["summary"]["confusion_matrix_sum"])) == 9 * 2 * 144


def test_erds_normalisation_is_unbiased_without_task_effect() -> None:
    from eeg_motor_imagery_classification.eda import baseline_percent_change

    rng = np.random.default_rng(0)
    times = np.linspace(-1.5, 4.5, 301)
    power = rng.chisquare(df=4, size=(200, times.size))  # noisy single-trial power, no task effect
    change = baseline_percent_change(power, times)
    task = (times >= 0.5) & (times <= 3.5)
    assert abs(change[task].mean()) < 5.0
    baseline_mask = (times >= -1.0) & (times <= -0.2)
    per_trial_ratio = 100 * (power / power[:, baseline_mask].mean(axis=1, keepdims=True) - 1).mean(axis=0)
    assert per_trial_ratio[task].mean() > change[task].mean() + 1.0  # the old recipe is biased upwards


def test_eegnet_repeated_sweep_varies_training_seed(monkeypatch) -> None:
    from eeg_motor_imagery_classification.experiments import transfer

    seen: list[tuple[int, int]] = []

    def fake_sweep(*_args, random_state, training_config, **_kwargs):
        seen.append((random_state, training_config.seed))
        return {"aggregate_by_setting": {}, "targets": {}, "runtime_seconds": 0.0}

    monkeypatch.setattr(transfer, "run_eegnet_transfer_sweep", fake_sweep)
    monkeypatch.setattr(transfer, "aggregate_transfer_seed_runs", lambda runs: {"n_seeds": len(runs)})
    transfer.run_eegnet_transfer_repeated_sweep(np.zeros((1, 1, 1)), np.zeros(1), np.zeros(1), seeds=(42, 43))
    assert seen == [(42, 42), (43, 43)]


def test_export_statistics_from_saved_outputs(tmp_path: Path) -> None:
    from eeg_motor_imagery_classification.stats_report import export_statistics

    def write(name: str, payload: dict) -> None:
        path = tmp_path / "outputs" / name / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload))

    def subject_rows(base: float) -> dict:
        rows = [{"label": f"S{s}", "accuracy": base + 0.01 * s, "balanced_accuracy": 0, "macro_f1": 0} for s in range(1, 10)]
        return {"summary": {"accuracy_mean": base}, "rows": rows}

    write("loso_riemann", subject_rows(0.60))
    write("loso_eegnet", subject_rows(0.70))
    for name, base in (("transfer_riemann_all_targets_seed42_43_v2", 0.0), ("transfer_eegnet_all_targets_seed42_43", 0.05)):
        run = aggregate_transfer_seed_runs({"seed_42": _transfer_seed_run(base), "seed_43": _transfer_seed_run(base)})
        write(name, json.loads(json.dumps(run, default=lambda o: o.tolist())))

    result = export_statistics(project_root=tmp_path, output_dir=tmp_path / "stats")
    stats = json.loads(Path(result["statistics_json"]).read_text())
    loso = stats["loso"]["comparisons"][0]
    assert loso["n_pairs"] == 9 and loso["p_value"] == pytest.approx(2 / 2**9)
    transfer_tests = stats["transfer"]["comparisons"]
    assert len(transfer_tests) == 5 and all(item["n_pairs"] == 9 for item in transfer_tests)
    assert "Holm" in Path(result["statistics_md"]).read_text()
