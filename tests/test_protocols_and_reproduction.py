"""Tests for cross-session evaluation, repeated EEGNet seeds, ERD/ERS summaries, and the runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _two_session_data(n_subjects: int = 3, n_per_session: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y, groups, sessions = [], [], [], []
    for subject in range(1, n_subjects + 1):
        for session in (0, 1):
            labels = np.repeat([0, 1], n_per_session // 2)
            trials = rng.standard_normal((n_per_session, 6, 128))
            trials[labels == 0, 4] *= 0.6
            trials[labels == 1, 1] *= 0.6
            X.append(trials * 5e-6)
            y.append(labels)
            groups.append(np.full(n_per_session, subject))
            sessions.append(np.full(n_per_session, session))
    return (np.concatenate(X).astype(np.float32), np.concatenate(y), np.concatenate(groups), np.concatenate(sessions))


def test_cross_session_splits_train_on_first_and_test_on_second_session() -> None:
    from eeg_motor_imagery_classification.experiments.cross_session import cross_session_splits

    _, _, groups, sessions = _two_session_data()
    splits = cross_session_splits(groups, sessions)
    assert [subject for subject, _, _ in splits] == [1, 2, 3]
    for subject, train_idx, test_idx in splits:
        assert set(sessions[train_idx]) == {0} and set(sessions[test_idx]) == {1}
        assert set(groups[train_idx]) == {subject} == set(groups[test_idx])
    with pytest.raises(ValueError):
        cross_session_splits(groups, np.zeros_like(sessions))


def test_classical_and_riemann_cross_session_learn_synthetic_lateralisation() -> None:
    from eeg_motor_imagery_classification.experiments import run_classical_cross_session, run_riemann_cross_session

    X, y, groups, sessions = _two_session_data()
    classical = run_classical_cross_session(X, y, groups, sessions, sfreq=128.0)
    assert classical["raw_power"]["summary"]["n"] == 3
    assert classical["raw_power"]["summary"]["accuracy_mean"] > 0.8
    riemann = run_riemann_cross_session(X, y, groups, sessions)
    assert riemann["summary"]["accuracy_mean"] > 0.8
    assert [row["label"] for row in riemann["rows"]] == ["S1", "S2", "S3"]


def test_eegnet_cross_session_over_training_seeds_averages_per_subject() -> None:
    from eeg_motor_imagery_classification.experiments import run_eegnet_cross_session, run_over_training_seeds
    from eeg_motor_imagery_classification.train import TrainingConfig

    X, y, groups, sessions = _two_session_data(n_subjects=2, n_per_session=24)
    cfg = TrainingConfig(epochs=2, batch_size=16, device="cpu", validation_split=0.25, min_epochs=1, patience=1)
    result = run_over_training_seeds(run_eegnet_cross_session, X, y, groups, sessions, seeds=(1, 2), training_config=cfg)
    assert result["seeds"] == [1, 2] and set(result["seed_runs"]) == {"seed_1", "seed_2"}
    assert result["summary"]["n"] == 2
    for row in result["rows"]:
        per_seed = [next(r["accuracy"] for r in run["rows"] if r["label"] == row["label"]) for run in result["seed_runs"].values()]
        assert row["accuracy"] == pytest.approx(np.mean(per_seed))
        assert np.sum(row["confusion_matrix"]) == 2 * 24
    assert all(history["label"].startswith("seed_") for history in result["training_histories"])


def test_resolve_device_rejects_unavailable_or_unknown_devices(monkeypatch) -> None:
    import torch

    from eeg_motor_imagery_classification.train import resolve_device

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_text_outputs_are_utf8(tmp_path: Path) -> None:
    from eeg_motor_imagery_classification.utils import read_json, write_json, write_text

    write_text(tmp_path / "a.md", "µV² ± SEM")
    assert (tmp_path / "a.md").read_bytes().decode("utf-8") == "µV² ± SEM"
    write_json(tmp_path / "a.json", {"unit": "µV"})
    assert read_json(tmp_path / "a.json") == {"unit": "µV"}


def _synthetic_curves(contra_drop: float, ipsi_drop: float):
    from eeg_motor_imagery_classification.eda import CONTRALATERAL_CHANNEL

    times = np.linspace(-1.0, 4.0, 251)
    task = (times >= 0.4) & (times <= 3.0)
    curves = {(-1, "times", "times"): times}
    for class_id in (0, 1):
        for channel in ("C3", "C4"):
            drop = contra_drop if CONTRALATERAL_CHANNEL[class_id] == channel else ipsi_drop
            for band in ("mu", "beta"):
                curves[(class_id, channel, band)] = np.where(task, -drop, 0.0)
    return curves


def test_erds_summary_rows_report_window_mean_and_peak() -> None:
    from eeg_motor_imagery_classification.eda import erds_summary_rows

    rows = erds_summary_rows(_synthetic_curves(30.0, 10.0), "S1")
    contra = [r for r in rows if r["side"] == "contralateral"]
    ipsi = [r for r in rows if r["side"] == "ipsilateral"]
    assert len(contra) == len(ipsi) == 4
    assert all(r["window_mean_pct"] == pytest.approx(-30.0) and r["peak_erd_pct"] == pytest.approx(-30.0) for r in contra)
    assert all(0.4 <= r["peak_latency_s"] <= 3.0 for r in contra)
    assert all(r["window_mean_pct"] == pytest.approx(-10.0) for r in ipsi)


def test_lateralization_tests_use_subjects_as_units() -> None:
    from eeg_motor_imagery_classification.eda import _lateralization_tests, erds_summary_rows

    rows = [row for s in range(1, 10) for row in erds_summary_rows(_synthetic_curves(30.0 + s, 10.0), f"S{s}")]
    tests = _lateralization_tests(rows)
    assert len(tests) == 4
    for test in tests:
        assert test["n_pairs"] == 9 and test["wins_y"] == 9  # contralateral more negative in every subject
        assert test["p_value"] == pytest.approx(2 / 2**9)
        assert test["p_holm"] == pytest.approx(4 * 2 / 2**9)


def test_primary_comparisons_are_tested_as_one_holm_family(tmp_path: Path) -> None:
    from eeg_motor_imagery_classification.stats_report import PRIMARY_COMPARISONS, export_statistics

    def write(name: str, payload: dict) -> None:
        path = tmp_path / "outputs" / name / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def rows(base: float) -> dict:
        return {"summary": {}, "rows": [{"label": f"S{s}", "accuracy": base + 0.01 * s, "balanced_accuracy": 0, "macro_f1": 0} for s in range(1, 10)]}

    write("cross_session_classical", {"raw_power": rows(0.6), "csp": rows(0.65), "fbcsp": rows(0.70)})
    write("cross_session_riemann", rows(0.72))
    write("cross_session_eegnet", {**rows(0.75), "seeds": [42, 43], "per_subject_seed_sd": {f"S{s}": 0.01 for s in range(1, 10)}})
    write("loso_riemann", rows(0.60))
    write("loso_eegnet", rows(0.70))
    result = export_statistics(project_root=tmp_path, output_dir=tmp_path / "stats")
    stats = json.loads(Path(result["statistics_json"]).read_text(encoding="utf-8"))
    assert len(stats["primary"]) == len(PRIMARY_COMPARISONS) == 3
    for item in stats["primary"]:
        assert item["p_value"] == pytest.approx(2 / 2**9)
        assert item["p_holm"] == pytest.approx(3 * 2 / 2**9)
    text = Path(result["statistics_md"]).read_text(encoding="utf-8")
    assert "Cross-Session (session 1 -> 2)" in text and "between-seed SD" in text


def test_runner_skips_current_results_and_reruns_stale_ones(tmp_path: Path, monkeypatch) -> None:
    from eeg_motor_imagery_classification import reproduce

    monkeypatch.setattr(reproduce, "PROJECT_ROOT", tmp_path)
    eegnet_stage = next(s for s in reproduce.STAGES if s.name == "loso_eegnet")
    riemann_stage = next(s for s in reproduce.STAGES if s.name == "loso_riemann")
    assert reproduce._result_status(eegnet_stage) == "missing"

    def write(stage, config) -> None:
        path = tmp_path / "outputs" / stage.output / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": [], **({"config": config} if config is not None else {})}), encoding="utf-8")

    write(riemann_stage, None)
    assert reproduce._result_status(riemann_stage) == "current"  # deterministic pipeline, legacy result
    write(eegnet_stage, None)
    assert reproduce._result_status(eegnet_stage) == "stale"  # EEGNet result without recorded settings
    write(eegnet_stage, reproduce._parse_cli_settings(eegnet_stage.args))
    assert reproduce._result_status(eegnet_stage) == "current"
    write(eegnet_stage, {**reproduce._parse_cli_settings(eegnet_stage.args), "epochs": 50})
    assert reproduce._result_status(eegnet_stage) == "stale"
    assert [s.name for s in reproduce.select_stages("primary")] == [
        "cross_session_classical", "cross_session_riemann", "cross_session_eegnet", "loso_riemann", "loso_eegnet"]


def test_cli_accepts_device_and_cross_session_experiments() -> None:
    import eeg_motor_imagery_classification.cli as cli

    args = cli.build_parser().parse_args(["--experiment", "eegnet_cross_session", "--device", "cuda", "--seed-list", "42,43"])
    assert args.device == "cuda" and args.experiment == "eegnet_cross_session"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--experiment", "eegnet_loso", "--device", "tpu"])
